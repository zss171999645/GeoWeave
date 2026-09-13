#!/usr/bin/env python3
"""Probe inference runtime and CUDA memory without saving point outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Sequence

from PIL import Image

from reprodata_pi3_protocol_infer import (
    DEFAULT_GEOWEAVE_CONFIG,
    default_checkpoint,
    load_model,
    load_protocol,
)


DEFAULT_MODELS = ("pi3_base", "geoweave_pi3", "dense_teacher_topk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", nargs="*", required=True)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--load-img-size", type=int, default=518)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--point-source", choices=["native", "depth_pose"], default="native")
    parser.add_argument("--teacher-topk", type=int, default=0)
    return parser.parse_args()


def finite_values(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def mean_or_none(values: Iterable[float]) -> float | None:
    vals = finite_values(values)
    return float(statistics.mean(vals)) if vals else None


def median_or_none(values: Iterable[float]) -> float | None:
    vals = finite_values(values)
    return float(statistics.median(vals)) if vals else None


def max_or_none(values: Iterable[float]) -> float | None:
    vals = finite_values(values)
    return float(max(vals)) if vals else None


def iter_protocol_jobs(
    protocol: dict[str, Any],
    *,
    variants: Sequence[str],
    sample_limit: int = 0,
) -> Iterator[tuple[str, str, Path]]:
    samples = list(protocol.get("samples", []))
    if sample_limit > 0:
        samples = samples[:sample_limit]
    for sample in samples:
        sample_id = str(sample["sample_id"])
        sample_variants = sample.get("variants", {})
        for variant in variants:
            payload = sample_variants.get(variant)
            if payload is None:
                continue
            yield sample_id, str(variant), Path(payload["input_dir"])


def collect_images(input_dir: Path, max_images: int) -> list[Path]:
    image_paths: list[Path] = []
    for pattern in ("input_v*.png", "input_v*.jpg", "input_v*.jpeg", "*.png", "*.jpg", "*.jpeg"):
        for path in sorted(input_dir.glob(pattern)):
            if path.is_file() and path not in image_paths:
                image_paths.append(path)
    image_paths = [path for path in image_paths if "column" not in path.stem.lower()]
    if max_images > 0:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise FileNotFoundError(f"No input images found under {input_dir}")
    return image_paths


def load_probe_model(model_name: str, args: argparse.Namespace):
    import torch

    if model_name == "pi3_base":
        return load_model("base", default_checkpoint("base"), None, torch.device(str(args.device)))
    if model_name == "geoweave_pi3":
        return load_model(
            "geoweave",
            default_checkpoint("geoweave"),
            DEFAULT_GEOWEAVE_CONFIG,
            torch.device(str(args.device)),
        )
    if model_name == "dense_teacher_topk":
        from blendedmvg_dense_teacher_topk_baseline import setup_model

        dense_args = SimpleNamespace(
            checkpoint=str(default_checkpoint("geoweave")),
            config=str(DEFAULT_GEOWEAVE_CONFIG),
            device=str(args.device),
            teacher_topk=int(args.teacher_topk),
        )
        model, loaded_checkpoint = setup_model(dense_args)
        return Path(__file__).resolve().parents[1], model, loaded_checkpoint
    raise ValueError(f"Unknown model: {model_name}")


def run_forward_discard(model: Any, input_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    from aidi.scripts.baselines.eval_pi3_mv_recon_core import infer_pi3_mv_pointclouds

    image_paths = collect_images(input_dir.expanduser().resolve(), int(args.max_images))
    first_image = Image.open(image_paths[0]).convert("RGB")
    started = time.time()
    points = infer_pi3_mv_pointclouds(
        filelist=[str(path) for path in image_paths],
        model=model,
        load_img_size=int(args.load_img_size),
        device=str(args.device),
        verbose=False,
        data_size=(first_image.height, first_image.width),
        point_source=str(args.point_source),
    )
    elapsed = time.time() - started
    shape = list(points.shape)
    del points
    return {"elapsed_seconds": float(elapsed), "points_shape": shape, "num_images": len(image_paths)}


def probe_model_jobs(
    *,
    model_name: str,
    jobs: Sequence[tuple[str, str, Path]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    import gc
    import torch

    repo_root, model, loaded_checkpoint = load_probe_model(model_name, args)
    rows: list[dict[str, Any]] = []
    for index, (sample_id, variant, input_dir) in enumerate(jobs, start=1):
        row: dict[str, Any] = {
            "setting": str(args.setting),
            "model": str(model_name),
            "sample_id": str(sample_id),
            "variant": str(variant),
            "input_dir": str(input_dir),
            "job_index": int(index),
            "job_count": int(len(jobs)),
            "repo_root": str(repo_root),
            "loaded_checkpoint": str(loaded_checkpoint),
        }
        try:
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            row.update(run_forward_discard(model, input_dir, args))
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                row["cuda_peak_allocated_gib"] = float(torch.cuda.max_memory_allocated() / (1024**3))
                row["cuda_peak_reserved_gib"] = float(torch.cuda.max_memory_reserved() / (1024**3))
                row["cuda_device"] = torch.cuda.get_device_name(torch.device(str(args.device)))
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = f"failed:{type(exc).__name__}:{exc}"
        rows.append(row)
        gc.collect()
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    del model
    gc.collect()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["setting"]), str(row["model"])), []).append(row)

    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group_rows in groups.items():
        ok = [row for row in group_rows if row.get("status") == "ok"]
        aggregate[key] = {
            "setting": key[0],
            "model": key[1],
            "jobs": len(group_rows),
            "ok_jobs": len(ok),
            "failed_jobs": len(group_rows) - len(ok),
            "mean_elapsed_seconds": mean_or_none(row.get("elapsed_seconds", float("nan")) for row in ok),
            "median_elapsed_seconds": median_or_none(row.get("elapsed_seconds", float("nan")) for row in ok),
            "max_elapsed_seconds": max_or_none(row.get("elapsed_seconds", float("nan")) for row in ok),
            "mean_cuda_peak_allocated_gib": mean_or_none(row.get("cuda_peak_allocated_gib", float("nan")) for row in ok),
            "max_cuda_peak_allocated_gib": max_or_none(row.get("cuda_peak_allocated_gib", float("nan")) for row in ok),
            "mean_cuda_peak_reserved_gib": mean_or_none(row.get("cuda_peak_reserved_gib", float("nan")) for row in ok),
            "max_cuda_peak_reserved_gib": max_or_none(row.get("cuda_peak_reserved_gib", float("nan")) for row in ok),
        }
    return aggregate


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    protocol = load_protocol(args.protocol)
    jobs = list(iter_protocol_jobs(protocol, variants=args.variants, sample_limit=int(args.sample_limit)))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for model_name in args.models:
        print(f"[runtime-probe] setting={args.setting} model={model_name} jobs={len(jobs)}", flush=True)
        all_rows.extend(probe_model_jobs(model_name=str(model_name), jobs=jobs, args=args))

    aggregate = aggregate_rows(all_rows)
    aggregate_rows_list = [aggregate[key] for key in sorted(aggregate)]
    write_csv(output_dir / "runtime_probe_rows.csv", all_rows)
    write_csv(output_dir / "runtime_probe_aggregate.csv", aggregate_rows_list)
    (output_dir / "runtime_probe_rows.json").write_text(json.dumps(all_rows, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "runtime_probe_aggregate.json").write_text(
        json.dumps(aggregate_rows_list, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[runtime-probe] wrote {output_dir / 'runtime_probe_aggregate.json'}")


if __name__ == "__main__":
    main()
