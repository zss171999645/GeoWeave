#!/usr/bin/env python3
"""Compute clean-vs-distractor prefix point-map consistency on reproduced data.

For each model/sample, the first `prefix_size` views are identical images in the
clean and distractor variants. The metric aligns the distractor prefix point map
to the clean prefix point map with a single row-vector Sim(3), then reports the
paired residual at corresponding pixels. Lower normalized residual means the
prefix reconstruction changed less when only the context views changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_MODELS = (
    "pi3_base",
    "geoweave_pi3",
    "dense_teacher_topk",
    "fastvggt_m0_r090",
    "vggt_nomerging_conf1",
)


@dataclass(frozen=True)
class RowSim3:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    target_rms: float
    n_align: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-name", default="prefix_point_consistency")
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--clean-variant", default="clean_tail")
    parser.add_argument("--noise-variant", default="plausible_noise_tail")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--max-points", type=int, default=120000)
    return parser.parse_args()


def load_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean_or_nan(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.mean(vals)) if vals else float("nan")


def median_or_nan(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(vals)) if vals else float("nan")


def load_prefix_points(path: Path, prefix_size: int) -> np.ndarray:
    with np.load(path) as data:
        points = np.asarray(data["points"], dtype=np.float64)
    if points.ndim not in {3, 4} or points.shape[-1] != 3:
        raise ValueError(f"Unsupported points shape in {path}: {points.shape}")
    if points.shape[0] < prefix_size:
        raise ValueError(f"Need at least {prefix_size} prefix views in {path}, got {points.shape[0]}")
    return points[:prefix_size]


def flatten_corresponding_points(
    clean: np.ndarray,
    noise: np.ndarray,
    *,
    stride: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    clean_arr = np.asarray(clean, dtype=np.float64)
    noise_arr = np.asarray(noise, dtype=np.float64)
    if clean_arr.shape != noise_arr.shape:
        raise ValueError(f"Clean/noise point shapes differ: {clean_arr.shape} vs {noise_arr.shape}")
    if clean_arr.ndim == 4:
        step = max(int(stride), 1)
        clean_arr = clean_arr[:, ::step, ::step, :]
        noise_arr = noise_arr[:, ::step, ::step, :]
    elif clean_arr.ndim != 3:
        raise ValueError(f"Unsupported point array shape: {clean_arr.shape}")

    valid = np.isfinite(clean_arr).all(axis=-1) & np.isfinite(noise_arr).all(axis=-1)
    valid &= np.abs(clean_arr).max(axis=-1) < 1.0e6
    valid &= np.abs(noise_arr).max(axis=-1) < 1.0e6
    clean_flat = clean_arr[valid]
    noise_flat = noise_arr[valid]
    if max_points > 0 and len(clean_flat) > max_points:
        indices = np.linspace(0, len(clean_flat) - 1, int(max_points)).round().astype(np.int64)
        clean_flat = clean_flat[indices]
        noise_flat = noise_flat[indices]
    return clean_flat.astype(np.float64, copy=False), noise_flat.astype(np.float64, copy=False)


def estimate_row_sim3(source: np.ndarray, target: np.ndarray) -> RowSim3:
    source_arr = np.asarray(source, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    if source_arr.shape != target_arr.shape or source_arr.ndim != 2 or source_arr.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 arrays, got {source_arr.shape} and {target_arr.shape}")
    if source_arr.shape[0] < 8:
        raise ValueError(f"Need at least 8 points for Sim(3), got {source_arr.shape[0]}")

    src_mu = source_arr.mean(axis=0)
    tgt_mu = target_arr.mean(axis=0)
    src_c = source_arr - src_mu
    tgt_c = target_arr - tgt_mu
    covariance = src_c.T @ tgt_c / float(source_arr.shape[0])
    u, singular, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        singular[-1] *= -1.0
        rotation = u @ vt
    var_src = float(np.mean(np.sum(src_c * src_c, axis=1)))
    scale = float(np.sum(singular) / max(var_src, 1.0e-12))
    translation = tgt_mu - scale * (src_mu @ rotation)
    target_rms = float(np.sqrt(np.mean(np.sum(tgt_c * tgt_c, axis=1))))
    return RowSim3(
        scale=scale,
        rotation=rotation,
        translation=translation,
        target_rms=max(target_rms, 1.0e-12),
        n_align=int(source_arr.shape[0]),
    )


def apply_row_sim3(points: np.ndarray, transform: RowSim3) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    return (pts @ transform.rotation) * float(transform.scale) + transform.translation


def compute_prefix_change(
    clean_points: np.ndarray,
    noise_points: np.ndarray,
    *,
    stride: int,
    max_points: int,
) -> dict[str, Any]:
    clean_flat, noise_flat = flatten_corresponding_points(
        clean_points,
        noise_points,
        stride=int(stride),
        max_points=int(max_points),
    )
    if len(clean_flat) < 8:
        return {"status": f"failed:not_enough_points:{len(clean_flat)}"}
    try:
        transform = estimate_row_sim3(noise_flat, clean_flat)
        aligned_noise = apply_row_sim3(noise_flat, transform)
        paired_dist = np.linalg.norm(aligned_noise - clean_flat, axis=1)
    except Exception as exc:
        return {"status": f"failed:{type(exc).__name__}:{exc}"}

    norm = float(transform.target_rms)
    return {
        "status": "ok",
        "n_points": int(len(clean_flat)),
        "n_align": int(transform.n_align),
        "align_scale_noise_to_clean": float(transform.scale),
        "clean_target_rms": norm,
        "paired_mean": float(np.mean(paired_dist)),
        "paired_median": float(np.median(paired_dist)),
        "paired_p90": float(np.percentile(paired_dist, 90.0)),
        "paired_mean_norm": float(np.mean(paired_dist / norm)),
        "paired_median_norm": float(np.median(paired_dist / norm)),
        "paired_p90_norm": float(np.percentile(paired_dist / norm, 90.0)),
    }


def model_output_file(output_root: Path, model: str, sample_id: str, variant: str) -> Path:
    return output_root / model / sample_id / variant / "points.npz"


def compute_rows(
    protocol: dict[str, Any],
    output_root: str | Path,
    *,
    models: Sequence[str],
    clean_variant: str,
    noise_variant: str,
    stride: int,
    max_points: int,
    sample_limit: int = 0,
) -> list[dict[str, Any]]:
    root = Path(output_root).expanduser().resolve()
    prefix_size = int(protocol.get("prefix_size", 6))
    samples = list(protocol.get("samples", []))
    if sample_limit > 0:
        samples = samples[:sample_limit]

    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        for model in models:
            clean_file = model_output_file(root, str(model), sample_id, clean_variant)
            noise_file = model_output_file(root, str(model), sample_id, noise_variant)
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "model": str(model),
                "clean_variant": clean_variant,
                "noise_variant": noise_variant,
                "clean_output": str(clean_file),
                "noise_output": str(noise_file),
            }
            try:
                clean_points = load_prefix_points(clean_file, prefix_size)
                noise_points = load_prefix_points(noise_file, prefix_size)
                row.update(
                    compute_prefix_change(
                        clean_points,
                        noise_points,
                        stride=int(stride),
                        max_points=int(max_points),
                    )
                )
            except Exception as exc:
                row["status"] = f"failed:{type(exc).__name__}:{exc}"
            rows.append(row)
    return rows


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_model: list[dict[str, Any]] = []
    models = []
    for row in rows:
        model = str(row.get("model"))
        if model not in models:
            models.append(model)
    for model in models:
        subset = [row for row in rows if row.get("model") == model and row.get("status") == "ok"]
        by_model.append(
            {
                "model": model,
                "n": len(subset),
                "paired_mean_norm": mean_or_nan(row.get("paired_mean_norm", float("nan")) for row in subset),
                "paired_median_norm": median_or_nan(row.get("paired_median_norm", float("nan")) for row in subset),
                "paired_p90_norm": mean_or_nan(row.get("paired_p90_norm", float("nan")) for row in subset),
                "paired_mean": mean_or_nan(row.get("paired_mean", float("nan")) for row in subset),
                "clean_target_rms": mean_or_nan(row.get("clean_target_rms", float("nan")) for row in subset),
            }
        )

    base = next((row for row in by_model if row["model"] == "pi3_base"), None)
    relative_vs_pi3: list[dict[str, Any]] = []
    for row in by_model:
        if base is None or row["model"] == "pi3_base":
            continue
        base_value = float(base.get("paired_mean_norm", float("nan")))
        value = float(row.get("paired_mean_norm", float("nan")))
        relative_vs_pi3.append(
            {
                "model": row["model"],
                "paired_mean_norm_relative_reduction": (
                    float((base_value - value) / base_value)
                    if math.isfinite(base_value) and base_value != 0.0 and math.isfinite(value)
                    else float("nan")
                ),
            }
        )
    return {"by_model": by_model, "relative_vs_pi3": relative_vs_pi3}


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_markdown(summary: dict[str, Any], csv_path: Path) -> str:
    lines = [
        "# Reproduced-data Prefix Point Consistency",
        "",
        "Metric: clean/noise prefix point maps are compared at corresponding pixels after one Sim(3) alignment from noise to clean. Lower normalized residual means the fixed prefix changed less when the four context views changed.",
        "",
        f"Protocol: `{summary['protocol']}`",
        f"Output root: `{summary['output_root']}`",
        f"Stride: {summary['stride']}",
        "",
        "| Model | OK pairs | Paired mean norm ↓ | Paired median norm ↓ | Paired p90 norm ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["aggregate"]["by_model"]:
        lines.append(
            f"| {row['model']} | {row['n']} | {row['paired_mean_norm']:.6f} | "
            f"{row['paired_median_norm']:.6f} | {row['paired_p90_norm']:.6f} |"
        )
    if summary["aggregate"]["relative_vs_pi3"]:
        lines.extend(["", "Relative reduction vs Pi3 base, higher is better:", ""])
        lines.append("| Model | Paired mean norm relative reduction |")
        lines.append("|---|---:|")
        for row in summary["aggregate"]["relative_vs_pi3"]:
            value = row["paired_mean_norm_relative_reduction"]
            text = "nan" if not math.isfinite(float(value)) else f"{float(value):.2%}"
            lines.append(f"| {row['model']} | {text} |")
    failed = [row for row in summary["rows"] if row.get("status") != "ok"]
    if failed:
        lines.extend(["", "## Failed Rows", ""])
        for row in failed[:50]:
            lines.append(f"- {row.get('model')} {row.get('sample_id')}: {row.get('status')}")
    lines.extend(["", f"Detailed CSV: `{csv_path}`", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    protocol = load_protocol(protocol_path)
    rows = compute_rows(
        protocol,
        output_root,
        models=list(args.models),
        clean_variant=str(args.clean_variant),
        noise_variant=str(args.noise_variant),
        stride=int(args.stride),
        max_points=int(args.max_points),
        sample_limit=int(args.sample_limit),
    )
    summary_dir = output_root / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / f"{args.summary_name}.csv"
    json_path = summary_dir / f"{args.summary_name}.json"
    md_path = summary_dir / f"{args.summary_name}.md"
    write_csv(csv_path, rows)
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": str(protocol_path),
        "output_root": str(output_root),
        "clean_variant": str(args.clean_variant),
        "noise_variant": str(args.noise_variant),
        "stride": int(args.stride),
        "max_points": int(args.max_points),
        "rows": rows,
        "aggregate": aggregate_rows(rows),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(summary, csv_path), encoding="utf-8")
    print(f"[prefix-consistency] wrote {csv_path}")
    print(f"[prefix-consistency] wrote {json_path}")
    print(f"[prefix-consistency] wrote {md_path}")


if __name__ == "__main__":
    main()
