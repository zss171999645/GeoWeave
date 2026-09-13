#!/usr/bin/env python3
"""Benchmark Pi3 and GeoWeave-Pi3 forward/backward step latency."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import statistics
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


TRAINING_OUTPUT_KEYS = ("camera_poses", "local_points", "points")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["pi3_base", "geoweave_pi3"], required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--height", type=int, default=392)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output-dir", required=True)
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
    from aidi.scripts.baselines.pi3_checkpoint_loader import (
        build_native_pi3_kwargs,
        filter_native_pi3_kwargs_for_class,
        import_native_pi3_class,
    )
    from reprodata_pi3_protocol_infer import DEFAULT_GEOWEAVE_CONFIG, default_checkpoint, load_model

    if model_name == "pi3_base":
        from safetensors.torch import load_file

        repo_root = Path(__file__).resolve().parents[1]
        pi3_class, _ = import_native_pi3_class(repo_root / "aidi" / "third_party" / "pi3_training")
        kwargs = make_native_base_kwargs(build_native_pi3_kwargs(DEFAULT_GEOWEAVE_CONFIG))
        kwargs = filter_native_pi3_kwargs_for_class(kwargs, pi3_class)
        model = pi3_class(**kwargs).to(device).eval()
        checkpoint = default_checkpoint("base")
        state_dict = load_file(str(checkpoint), device="cpu")
        load_result = model.load_state_dict(state_dict, strict=False)
        print(f"[training-time] loaded fair native Pi3 base result={load_result}", flush=True)
        return repo_root, model, str(checkpoint)
    if model_name == "geoweave_pi3":
        return load_model("geoweave", default_checkpoint("geoweave"), DEFAULT_GEOWEAVE_CONFIG, device)
    raise ValueError(f"Unknown model: {model_name}")


def make_native_base_kwargs(source: Mapping[str, Any]) -> dict[str, Any]:
    kwargs = copy.deepcopy(dict(source))
    indexer_cfg = copy.deepcopy(dict(kwargs.get("indexer_cfg", {})))
    indexer_cfg["enabled"] = False
    kwargs["indexer_cfg"] = indexer_cfg
    return kwargs


def count_trainable_parameters(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def configure_training_state(model: Any) -> dict[str, Any]:
    model.train()
    if not hasattr(model, "set_indexer_state_by_step"):
        return {}
    state = model.set_indexer_state_by_step(100_000, training=True)
    if isinstance(state, Mapping):
        return {str(key): value for key, value in state.items() if isinstance(value, (str, int, float, bool))}
    return {}


def proxy_training_loss(
    output: Any,
    *,
    auxiliary_loss: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[str]]:
    if not isinstance(output, Mapping):
        raise TypeError(f"Expected model output mapping, got {type(output)!r}")
    terms: list[torch.Tensor] = []
    used_keys: list[str] = []
    for key in sorted(TRAINING_OUTPUT_KEYS):
        value = output.get(key)
        if not torch.is_tensor(value) or not value.is_floating_point() or not value.requires_grad:
            continue
        terms.append(value.float().square().mean())
        used_keys.append(key)
    if auxiliary_loss is not None:
        if not torch.is_tensor(auxiliary_loss) or not auxiliary_loss.requires_grad:
            raise ValueError("Auxiliary loss must be a differentiable tensor")
        terms.append(auxiliary_loss.float())
        used_keys.append("indexer_loss")
    if not terms:
        raise ValueError("No differentiable training outputs were found")
    return torch.stack(terms).sum(), used_keys


def summarize_timings_ms(timings_ms: list[float]) -> dict[str, float]:
    if not timings_ms:
        raise ValueError("At least one timing is required")
    return {
        "mean_ms": float(statistics.mean(timings_ms)),
        "median_ms": float(statistics.median(timings_ms)),
        "min_ms": float(min(timings_ms)),
        "max_ms": float(max(timings_ms)),
    }


def make_synthetic_input(*, frames: int, height: int, width: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(frames) * 1009 + int(height) * 17 + int(width))
    tensor = torch.rand((1, int(frames), 3, int(height), int(width)), generator=generator, dtype=torch.float32)
    return tensor.to(device=device)


def run_training_step(model: Any, imgs: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> list[str]:
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        output = model(imgs)
        auxiliary_loss = getattr(model, "indexer_loss", None)
        if isinstance(auxiliary_loss, list):
            auxiliary_loss = torch.stack([item.float() for item in auxiliary_loss]).sum()
        loss, used_keys = proxy_training_loss(output, auxiliary_loss=auxiliary_loss)
    loss.backward()
    del loss, output
    return used_keys


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
    training_state: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model_name,
        "frames": int(frames),
        "height": int(height),
        "width": int(width),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "training_state": training_state,
        "status": "ok",
    }
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    imgs = make_synthetic_input(frames=frames, height=height, width=width, device=device, seed=seed)
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        used_keys: list[str] = []
        for _ in range(max(int(warmup), 0)):
            used_keys = run_training_step(model, imgs, device=device, dtype=dtype)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        timings_ms: list[float] = []
        for _ in range(max(int(repeats), 1)):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device)
                start.record()
                used_keys = run_training_step(model, imgs, device=device, dtype=dtype)
                end.record()
                torch.cuda.synchronize(device)
                elapsed_ms = float(start.elapsed_time(end))
            else:
                started = time.perf_counter()
                used_keys = run_training_step(model, imgs, device=device, dtype=dtype)
                elapsed_ms = float((time.perf_counter() - started) * 1000.0)
            timings_ms.append(elapsed_ms)
        row["timings_ms"] = timings_ms
        row.update(summarize_timings_ms(timings_ms))
        row["mean_seconds"] = float(row["mean_ms"] / 1000.0)
        row["loss_terms"] = used_keys
        if device.type == "cuda":
            row["cuda_peak_allocated_gib"] = float(torch.cuda.max_memory_allocated(device) / (1024**3))
            row["cuda_peak_reserved_gib"] = float(torch.cuda.max_memory_reserved(device) / (1024**3))
    except torch.cuda.OutOfMemoryError as exc:
        row.update({"status": "oom", "error_type": type(exc).__name__, "error_message": str(exc)})
    except RuntimeError as exc:
        status = "oom" if "out of memory" in str(exc).lower() else "failed"
        row.update({"status": status, "error_type": type(exc).__name__, "error_message": str(exc)})
    finally:
        model.zero_grad(set_to_none=True)
        del imgs
        gc.collect()
        if device.type == "cuda":
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
        "method": "zero_grad + model.train forward + differentiable output proxy/indexer loss + backward; excludes data loading and optimizer step",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if device.type == "cuda":
        metadata["cuda_device_name"] = torch.cuda.get_device_name(device)
        metadata["cuda_device_count"] = int(torch.cuda.device_count())

    rows: list[dict[str, Any]] = []
    for model_name in args.models:
        print(f"[training-time] loading model={model_name}", flush=True)
        repo_root_loaded, model, loaded_checkpoint = load_named_model(model_name, device)
        training_state = configure_training_state(model)
        total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
        trainable_parameters = count_trainable_parameters(model)
        for frames in args.frames:
            print(f"[training-time] model={model_name} frames={frames}", flush=True)
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
                training_state=training_state,
            )
            row["repo_root"] = str(repo_root_loaded)
            row["loaded_checkpoint"] = str(loaded_checkpoint)
            row["total_parameters"] = total_parameters
            row["trainable_parameters"] = trainable_parameters
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (output_dir / "training_time_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "training_time_rows.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "training_time_rows.csv", rows)
    print(f"[training-time] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
