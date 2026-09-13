#!/usr/bin/env python3
"""Benchmark Pi3 and GeoWeave-Pi3 forward latency on synthetic image tensors."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["pi3_base", "geoweave_pi3"], required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--height", type=int, default=392)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--skip-empty-cache-after-case",
        action="store_true",
        help="Skip torch.cuda.empty_cache() after each timed case; useful for CUDA graph replay benchmarks.",
    )
    return parser.parse_args()


def add_repo_imports() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    tools_root = repo_root / "tools"
    pi3_native_root = repo_root / "aidi" / "third_party" / "pi3_training"
    for path in (repo_root, tools_root, pi3_native_root):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return repo_root


def load_named_model(model_name: str, device: torch.device):
    from reprodata_pi3_protocol_infer import DEFAULT_GEOWEAVE_CONFIG, default_checkpoint, load_model

    if model_name == "pi3_base":
        return load_model("base", default_checkpoint("base"), None, device)
    if model_name == "geoweave_pi3":
        return load_model("geoweave", default_checkpoint("geoweave"), DEFAULT_GEOWEAVE_CONFIG, device)
    raise ValueError(f"Unknown model: {model_name}")


def output_shapes(output: Any) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    if isinstance(output, dict):
        for key, value in output.items():
            if torch.is_tensor(value):
                shapes[str(key)] = [int(dim) for dim in value.shape]
    return shapes


def make_synthetic_input(*, frames: int, height: int, width: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(frames) * 1009 + int(height) * 17 + int(width))
    cpu_tensor = torch.rand((1, int(frames), 3, int(height), int(width)), generator=generator, dtype=torch.float32)
    return cpu_tensor.to(device=device, non_blocking=False)


def run_one_case(
    *,
    model: Any,
    model_name: str,
    frames: int,
    height: int,
    width: int,
    warmup: int,
    repeats: int,
    device: torch.device,
    seed: int,
    skip_empty_cache_after_case: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model_name,
        "frames": int(frames),
        "height": int(height),
        "width": int(width),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "status": "ok",
    }
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    imgs = make_synthetic_input(frames=frames, height=height, width=width, device=device, seed=seed)

    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        with torch.no_grad():
            for _ in range(max(int(warmup), 0)):
                with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                    output = model(imgs)
                del output
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

            timings_ms: list[float] = []
            last_shapes: dict[str, list[int]] = {}
            for _ in range(max(int(repeats), 1)):
                if device.type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(device)
                    start.record()
                    with torch.amp.autocast(device_type="cuda", dtype=dtype):
                        output = model(imgs)
                    end.record()
                    torch.cuda.synchronize(device)
                    elapsed_ms = float(start.elapsed_time(end))
                else:
                    started = time.perf_counter()
                    output = model(imgs)
                    elapsed_ms = float((time.perf_counter() - started) * 1000.0)
                timings_ms.append(elapsed_ms)
                last_shapes = output_shapes(output)
                del output

        row["timings_ms"] = timings_ms
        row["mean_ms"] = float(statistics.mean(timings_ms))
        row["median_ms"] = float(statistics.median(timings_ms))
        row["min_ms"] = float(min(timings_ms))
        row["max_ms"] = float(max(timings_ms))
        row["mean_seconds"] = row["mean_ms"] / 1000.0
        row["median_seconds"] = row["median_ms"] / 1000.0
        row["fps_by_frames_per_mean_second"] = float(int(frames) / row["mean_seconds"])
        row["output_shapes"] = last_shapes
        if device.type == "cuda":
            row["cuda_peak_allocated_gib"] = float(torch.cuda.max_memory_allocated(device) / (1024**3))
            row["cuda_peak_reserved_gib"] = float(torch.cuda.max_memory_reserved(device) / (1024**3))
    except torch.cuda.OutOfMemoryError as exc:
        row["status"] = "oom"
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc)
        if device.type == "cuda":
            row["cuda_peak_allocated_gib"] = float(torch.cuda.max_memory_allocated(device) / (1024**3))
            row["cuda_peak_reserved_gib"] = float(torch.cuda.max_memory_reserved(device) / (1024**3))
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            row["status"] = "oom"
        else:
            row["status"] = "failed"
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc)
        if device.type == "cuda":
            row["cuda_peak_allocated_gib"] = float(torch.cuda.max_memory_allocated(device) / (1024**3))
            row["cuda_peak_reserved_gib"] = float(torch.cuda.max_memory_reserved(device) / (1024**3))
    finally:
        del imgs
        gc.collect()
        if device.type == "cuda" and not skip_empty_cache_after_case:
            torch.cuda.empty_cache()
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo_root = add_repo_imports()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    metadata: dict[str, Any] = {
        "repo_root": str(repo_root),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": "Direct model(imgs) forward on synthetic tensor; excludes image I/O, resize, point interpolation, and file writing.",
        "skip_empty_cache_after_case": bool(args.skip_empty_cache_after_case),
    }
    if device.type == "cuda":
        metadata["cuda_device_name"] = torch.cuda.get_device_name(device)
        metadata["cuda_device_count"] = int(torch.cuda.device_count())
        metadata["cuda_benchmark_dtype"] = "bfloat16" if torch.cuda.get_device_capability(device)[0] >= 8 else "float16"

    rows: list[dict[str, Any]] = []
    for model_name in args.models:
        print(f"[synthetic-runtime] loading model={model_name}", flush=True)
        repo_root_loaded, model, loaded_checkpoint = load_named_model(model_name, device)
        model.eval()
        for frames in args.frames:
            print(
                f"[synthetic-runtime] model={model_name} frames={frames} shape=(1,{frames},3,{args.height},{args.width})",
                flush=True,
            )
            row = run_one_case(
                model=model,
                model_name=model_name,
                frames=int(frames),
                height=int(args.height),
                width=int(args.width),
                warmup=int(args.warmup),
                repeats=int(args.repeats),
                device=device,
                seed=int(args.seed),
                skip_empty_cache_after_case=bool(args.skip_empty_cache_after_case),
            )
            row["repo_root"] = str(repo_root_loaded)
            row["loaded_checkpoint"] = str(loaded_checkpoint)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        del model
        gc.collect()
        if device.type == "cuda" and not args.skip_empty_cache_after_case:
            torch.cuda.empty_cache()

    metadata["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (output_dir / "synthetic_runtime_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "synthetic_runtime_rows.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "synthetic_runtime_rows.csv", rows)
    print(f"[synthetic-runtime] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
