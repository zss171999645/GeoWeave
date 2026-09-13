#!/usr/bin/env python3
"""Compare GeoWeave outputs and runtime under operator-only environment flags."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


DEFAULT_CLEARED_FLAGS = (
    "VGGT_INDEXER_TOPK_BLOCK",
    "VGGT_INDEXER_TOPK_MERGE_BLOCKS",
    "VGGT_INDEXER_TOPK_MERGE_INDEX_BLOCK_K",
    "VGGT_INDEXER_TOPK_BLOCK_MERGE_POLICY",
    "VGGT_INDEXER_TOPK_BUCKET_POLICY",
    "VGGT_INDEXER_TOPK_EXACT_KERNEL",
    "VGGT_INDEXER_TOPK_MERGE_TWO_KERNEL",
    "VGGT_INDEXER_TOPK_MERGE_TWO_CUDA",
    "VGGT_INDEXER_SCORE_TOPK_FUSED_CUDA",
    "VGGT_SPARSE_FLASH_ATTN_BLOCK_M",
    "VGGT_SPARSE_FLASH_ATTN_BLOCK_N",
    "VGGT_SPARSE_FLASH_ATTN_NUM_WARPS",
    "VGGT_SPARSE_FLASH_ATTN_NUM_STAGES",
    "VGGT_SPARSE_FLASH_DISABLE_ATTN_SUM",
    "VGGT_SPARSE_FLASH_DIRECT_BHTD",
    "VGGT_SPARSE_FLASH_NO_QKV_CONTIG",
    "VGGT_SPARSE_FLASH_NO_KV_CONTIG",
    "VGGT_DSA_FULLCHAIN_FASTPATH",
    "VGGT_DSA_FULLCHAIN_CUDAGRAPH",
    "VGGT_DSA_FULLCHAIN_GRAPH_WARMUP",
    "VGGT_DSA_FUSE_QKV_INDEXER_PROJ",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--height", type=int, default=392)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--optimized-env", nargs="*", default=[])
    parser.add_argument("--output-json", required=True)
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


def parse_env_items(items: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE env item, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"Empty env key in {item!r}")
        env[key] = value
    return env


@contextmanager
def temporary_env(updates: dict[str, str], clear: tuple[str, ...] = DEFAULT_CLEARED_FLAGS):
    touched = set(clear) | set(updates)
    old = {key: os.environ.get(key) for key in touched}
    try:
        for key in clear:
            os.environ.pop(key, None)
        os.environ.update(updates)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def make_input(frames: int, height: int, width: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(frames) * 1009 + int(height) * 17 + int(width))
    imgs = torch.rand((1, frames, 3, height, width), generator=generator, dtype=torch.float32)
    return imgs.to(device=device)


def load_geoweave_model(device: torch.device):
    from reprodata_pi3_protocol_infer import DEFAULT_GEOWEAVE_CONFIG, default_checkpoint, load_model

    _, model, checkpoint = load_model("geoweave", default_checkpoint("geoweave"), DEFAULT_GEOWEAVE_CONFIG, device)
    model.eval()
    return model, checkpoint


def detach_output(output: Any) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    if not isinstance(output, dict):
        return result
    for key, value in output.items():
        if torch.is_tensor(value):
            result[str(key)] = value.detach().float().cpu()
    return result


def run_case(
    *,
    model: Any,
    imgs: torch.Tensor,
    device: torch.device,
    env: dict[str, str],
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    timings_ms: list[float] = []
    output_cpu: dict[str, torch.Tensor] = {}

    with temporary_env(env):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        with torch.no_grad():
            for _ in range(max(int(warmup), 0)):
                with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                    warm = model(imgs)
                del warm
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

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
                    timings_ms.append(float(start.elapsed_time(end)))
                else:
                    started = time.perf_counter()
                    output = model(imgs)
                    timings_ms.append(float((time.perf_counter() - started) * 1000.0))
                output_cpu = detach_output(output)
                del output

        row: dict[str, Any] = {
            "env": dict(env),
            "timings_ms": timings_ms,
            "mean_ms": float(statistics.mean(timings_ms)),
            "median_ms": float(statistics.median(timings_ms)),
            "min_ms": float(min(timings_ms)),
            "max_ms": float(max(timings_ms)),
            "output_shapes": {key: list(value.shape) for key, value in output_cpu.items()},
        }
        if device.type == "cuda":
            row["cuda_peak_allocated_gib"] = float(torch.cuda.max_memory_allocated(device) / (1024**3))
            row["cuda_peak_reserved_gib"] = float(torch.cuda.max_memory_reserved(device) / (1024**3))
    return {"stats": row, "output": output_cpu}


def compare_outputs(base: dict[str, torch.Tensor], other: dict[str, torch.Tensor]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in sorted(set(base) | set(other)):
        if key not in base or key not in other:
            metrics[key] = {"status": "missing"}
            continue
        lhs = base[key]
        rhs = other[key]
        if tuple(lhs.shape) != tuple(rhs.shape):
            metrics[key] = {"status": "shape_mismatch", "base_shape": list(lhs.shape), "other_shape": list(rhs.shape)}
            continue
        diff = (lhs - rhs).abs()
        metrics[key] = {
            "status": "ok",
            "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
            "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        }
    return metrics


def main() -> None:
    args = parse_args()
    repo_root = add_repo_imports()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    optimized_env = parse_env_items(args.optimized_env)
    imgs = make_input(args.frames, args.height, args.width, device, args.seed)
    model, checkpoint = load_geoweave_model(device)

    baseline = run_case(
        model=model,
        imgs=imgs,
        device=device,
        env={},
        warmup=args.warmup,
        repeats=args.repeats,
    )
    optimized = run_case(
        model=model,
        imgs=imgs,
        device=device,
        env=optimized_env,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    result = {
        "repo_root": str(repo_root),
        "checkpoint": str(checkpoint),
        "frames": int(args.frames),
        "height": int(args.height),
        "width": int(args.width),
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "baseline": baseline["stats"],
        "optimized": optimized["stats"],
        "comparison": compare_outputs(baseline["output"], optimized["output"]),
    }
    path = Path(args.output_json).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
