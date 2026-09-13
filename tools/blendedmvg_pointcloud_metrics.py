#!/usr/bin/env python3
"""Compute Acc/Comp/NC point-cloud metrics for BlendedMVG rebuttal outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from blendedmvg_point_degradation_protocol import (
    MODELS,
    PREFIX_SIZE,
    VARIANTS,
    finite_points,
    flatten_metric_points,
    load_prefix_gt,
    load_prefix_pred,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-name", default="pointcloud_metrics")
    parser.add_argument("--metric-stride", type=int, default=16)
    parser.add_argument("--max-metric-points", type=int, default=120000)
    parser.add_argument("--normal-metrics", action="store_true")
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    return parser.parse_args()


def mean_or_nan(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.mean(vals)) if vals else float("nan")


def median_or_nan(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(vals)) if vals else float("nan")


def estimate_sim3_transform(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Estimate row-vector Sim(3) transform mapping source to target."""

    n = min(int(source.shape[0]), int(target.shape[0]))
    if n < 8:
        raise ValueError(f"Need at least 8 points for Sim3 alignment, got {n}")
    src = source[:n].astype(np.float64, copy=False)
    tgt = target[:n].astype(np.float64, copy=False)
    src_mu = src.mean(axis=0)
    tgt_mu = tgt.mean(axis=0)
    src_c = src - src_mu
    tgt_c = tgt - tgt_mu
    cov = src_c.T @ tgt_c / float(n)
    u, s, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        s[-1] *= -1.0
        r = u @ vt
    var_src = float(np.mean(np.sum(src_c * src_c, axis=1)))
    scale = float(np.sum(s) / max(var_src, 1.0e-12))
    t = tgt_mu - scale * (src_mu @ r)
    target_rms = float(np.sqrt(np.mean(np.sum(tgt_c * tgt_c, axis=1))))
    return {
        "scale": scale,
        "rotation": r,
        "translation": t,
        "n_align_points": int(n),
        "target_rms": max(target_rms, 1.0e-12),
    }


def apply_sim3(points: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    return (points.astype(np.float64, copy=False) @ transform["rotation"]) * float(transform["scale"]) + transform[
        "translation"
    ]


def nearest_neighbors(reference: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree as KDTree  # type: ignore

        tree = KDTree(reference)
        distances, indices = tree.query(query, workers=-1)
        return distances.astype(np.float64, copy=False), indices.astype(np.int64, copy=False)
    except Exception:
        pass

    reference = np.asarray(reference, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    distances = np.empty((len(query),), dtype=np.float64)
    indices = np.empty((len(query),), dtype=np.int64)
    chunk = 4096
    for start in range(0, len(query), chunk):
        end = min(start + chunk, len(query))
        diff = query[start:end, None, :] - reference[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        idx = np.argmin(dist2, axis=1)
        indices[start:end] = idx
        distances[start:end] = np.sqrt(dist2[np.arange(end - start), idx])
    return distances, indices


def estimate_normals(points: np.ndarray) -> np.ndarray | None:
    try:
        import open3d as o3d  # type: ignore
    except Exception:
        return None
    if len(points) < 8:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=min(30, len(points))))
    normals = np.asarray(pcd.normals, dtype=np.float64)
    if normals.shape != points.shape or not np.isfinite(normals).all():
        return None
    return normals


def compute_aligned_cloud_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    *,
    normal_metrics: bool = True,
) -> dict[str, Any]:
    pred = np.asarray(pred_points, dtype=np.float64)
    gt = np.asarray(gt_points, dtype=np.float64)
    finite = np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1)
    pred = pred[finite]
    gt = gt[finite]
    if len(pred) < 8 or len(gt) < 8:
        return {"status": f"failed:not_enough_points:{len(pred)}"}

    try:
        transform = estimate_sim3_transform(pred, gt)
        aligned_pred = apply_sim3(pred, transform)
        acc_dist, acc_idx = nearest_neighbors(gt, aligned_pred)
        comp_dist, comp_idx = nearest_neighbors(aligned_pred, gt)
    except Exception as exc:
        return {"status": f"failed:{type(exc).__name__}:{exc}"}

    norm = float(transform["target_rms"])
    metrics: dict[str, Any] = {
        "status": "ok",
        "n_points": int(len(pred)),
        "n_align_points": int(transform["n_align_points"]),
        "align_scale": float(transform["scale"]),
        "target_rms": norm,
        "acc_mean": float(np.mean(acc_dist)),
        "acc_median": float(np.median(acc_dist)),
        "acc_p90": float(np.percentile(acc_dist, 90.0)),
        "comp_mean": float(np.mean(comp_dist)),
        "comp_median": float(np.median(comp_dist)),
        "comp_p90": float(np.percentile(comp_dist, 90.0)),
        "acc_mean_norm": float(np.mean(acc_dist / norm)),
        "acc_median_norm": float(np.median(acc_dist / norm)),
        "acc_p90_norm": float(np.percentile(acc_dist / norm, 90.0)),
        "comp_mean_norm": float(np.mean(comp_dist / norm)),
        "comp_median_norm": float(np.median(comp_dist / norm)),
        "comp_p90_norm": float(np.percentile(comp_dist / norm, 90.0)),
    }

    if normal_metrics:
        pred_normals = estimate_normals(aligned_pred)
        gt_normals = estimate_normals(gt)
        if pred_normals is not None and gt_normals is not None:
            acc_nc = np.abs(np.sum(gt_normals[acc_idx] * pred_normals, axis=-1))
            comp_nc = np.abs(np.sum(gt_normals * pred_normals[comp_idx], axis=-1))
            metrics.update(
                {
                    "nc_acc_mean": float(np.mean(acc_nc)),
                    "nc_acc_median": float(np.median(acc_nc)),
                    "nc_comp_mean": float(np.mean(comp_nc)),
                    "nc_comp_median": float(np.median(comp_nc)),
                    "nc_mean": float(0.5 * (np.mean(acc_nc) + np.mean(comp_nc))),
                    "nc_median": float(0.5 * (np.median(acc_nc) + np.median(comp_nc))),
                }
            )
        else:
            metrics["normal_status"] = "skipped:open3d_unavailable_or_failed"

    return metrics


def compute_rows(args: argparse.Namespace, protocol: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    prefix_size = int(protocol.get("prefix_size", PREFIX_SIZE))
    rows: list[dict[str, Any]] = []
    for sample in protocol["samples"]:
        sample_id = sample["sample_id"]
        for model in args.models:
            for variant in args.variants:
                out_dir = output_root / model / sample_id / variant
                row: dict[str, Any] = {
                    "sample_id": sample_id,
                    "model": model,
                    "variant": variant,
                    "output_dir": str(out_dir),
                }
                try:
                    pred = load_prefix_pred(out_dir, prefix_size=prefix_size)
                    gt, gt_valid = load_prefix_gt(sample, variant, prefix_size=prefix_size)
                    valid = finite_points(pred) & gt_valid
                    pred_flat = flatten_metric_points(
                        pred,
                        valid=valid,
                        stride=int(args.metric_stride),
                        max_points=int(args.max_metric_points),
                    )
                    gt_flat = flatten_metric_points(
                        gt,
                        valid=valid,
                        stride=int(args.metric_stride),
                        max_points=int(args.max_metric_points),
                    )
                    row.update(compute_aligned_cloud_metrics(pred_flat, gt_flat, normal_metrics=bool(args.normal_metrics)))
                except Exception as exc:
                    row["status"] = f"failed:{type(exc).__name__}:{exc}"
                rows.append(row)
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model_variant: list[dict[str, Any]] = []
    models = []
    variants = []
    for row in rows:
        if row.get("model") not in models:
            models.append(row.get("model"))
        if row.get("variant") not in variants:
            variants.append(row.get("variant"))

    metric_keys = [
        "acc_mean_norm",
        "acc_median_norm",
        "comp_mean_norm",
        "comp_median_norm",
        "nc_mean",
        "nc_median",
        "acc_mean",
        "comp_mean",
    ]
    for model in models:
        for variant in variants:
            subset = [r for r in rows if r.get("model") == model and r.get("variant") == variant and r.get("status") == "ok"]
            payload: dict[str, Any] = {"model": model, "variant": variant, "n": len(subset)}
            for key in metric_keys:
                payload[key] = mean_or_nan(r.get(key, float("nan")) for r in subset)
            by_model_variant.append(payload)

    relative_vs_pi3: list[dict[str, Any]] = []
    for variant in variants:
        base = next((r for r in by_model_variant if r["model"] == "pi3_base" and r["variant"] == variant), None)
        for row in by_model_variant:
            if row["variant"] != variant or row["model"] == "pi3_base" or base is None:
                continue
            payload = {"model": row["model"], "variant": variant}
            for key in ("acc_mean_norm", "comp_mean_norm"):
                b = float(base.get(key, float("nan")))
                v = float(row.get(key, float("nan")))
                payload[f"{key}_relative_reduction"] = float((b - v) / b) if math.isfinite(b) and b != 0 else float("nan")
            relative_vs_pi3.append(payload)

    return {
        "by_model_variant": by_model_variant,
        "relative_vs_pi3": relative_vs_pi3,
    }


def render_markdown(summary: dict[str, Any], csv_path: Path) -> str:
    lines = [
        "# BlendedMVG Point-Cloud Metrics",
        "",
        f"Protocol: `{summary['protocol']}`",
        f"Output root: `{summary['output_root']}`",
        f"Metric stride: {summary['metric_stride']}",
        f"Normal metrics: {summary['normal_metrics']}",
        "",
        "Prefix point-cloud metrics after Sim(3), lower is better for Acc/Comp and higher is better for NC.",
        "",
        "| model | variant | n | Acc mean norm | Comp mean norm | NC mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregate"]["by_model_variant"]:
        nc = row.get("nc_mean", float("nan"))
        nc_text = "nan" if not math.isfinite(float(nc)) else f"{float(nc):.6f}"
        lines.append(
            f"| {row['model']} | {row['variant']} | {row['n']} | "
            f"{row['acc_mean_norm']:.6f} | {row['comp_mean_norm']:.6f} | {nc_text} |"
        )
    lines.extend(["", "Relative reduction vs Pi3 base, higher is better:", ""])
    lines.append("| model | variant | Acc rel. reduction | Comp rel. reduction |")
    lines.append("|---|---:|---:|---:|")
    for row in summary["aggregate"]["relative_vs_pi3"]:
        lines.append(
            f"| {row['model']} | {row['variant']} | "
            f"{row['acc_mean_norm_relative_reduction']:.2%} | "
            f"{row['comp_mean_norm_relative_reduction']:.2%} |"
        )
    lines.extend(["", f"Detailed CSV: `{csv_path}`", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rows = compute_rows(args, protocol, output_root)
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
        "metric_stride": int(args.metric_stride),
        "max_metric_points": int(args.max_metric_points),
        "normal_metrics": bool(args.normal_metrics),
        "num_rows": len(rows),
        "num_ok": sum(1 for row in rows if row.get("status") == "ok"),
        "aggregate": aggregate_rows(rows),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(summary, csv_path), encoding="utf-8")
    print(f"[pointcloud-metrics] wrote {csv_path}")
    print(f"[pointcloud-metrics] wrote {json_path}")
    print(f"[pointcloud-metrics] wrote {md_path}")


if __name__ == "__main__":
    main()
