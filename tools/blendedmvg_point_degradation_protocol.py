#!/usr/bin/env python3
"""Build and summarize a BlendedMVG point-degradation rebuttal protocol.

The protocol keeps the first six input images fixed and changes only the last
four context images:

- clean: same-scene high-overlap context
- tail: same-scene low-overlap context
- xscene: cross-scene distractor context

The summary compares predicted point maps on the fixed prefix after Sim(3)
alignment, and also measures prefix point-map error against BlendedMVG GT depth
after Sim(3) alignment.
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
from typing import Any, Iterable

import numpy as np
from PIL import Image


PREFIX_SIZE = 6
CONTEXT_SIZE = 4
VARIANTS = ("clean", "tail", "xscene")
MODELS = ("geoweave_final", "pi3_base")


@dataclass(frozen=True)
class FrameRecord:
    scene: str
    cache_path: Path
    frame_id: int
    depth: np.ndarray
    c2w: np.ndarray
    intrinsic: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="Create input folders and protocol.json.")
    prep.add_argument("--dataset-root", required=True)
    prep.add_argument("--input-root", required=True)
    prep.add_argument("--num-samples", type=int, default=5)
    prep.add_argument("--max-frames-per-scene", type=int, default=60)
    prep.add_argument("--overlap-stride", type=int, default=24)
    prep.add_argument("--depth-rel-tol", type=float, default=0.02)
    prep.add_argument("--min-top5-overlap", type=float, default=0.05)
    prep.add_argument("--limit-scenes-scan", type=int, default=40)
    prep.add_argument("--overwrite", action="store_true")

    jobs = sub.add_parser("emit-jobs", help="Emit TSV jobs for the launcher.")
    jobs.add_argument("--protocol", required=True)
    jobs.add_argument("--output-root", required=True)

    summ = sub.add_parser("summarize", help="Summarize completed model outputs.")
    summ.add_argument("--protocol", required=True)
    summ.add_argument("--output-root", required=True)
    summ.add_argument("--metric-stride", type=int, default=16)
    summ.add_argument("--max-metric-points", type=int, default=200000)
    return parser.parse_args()


def scene_cache_dirs(dataset_root: Path, limit: int) -> list[Path]:
    dirs = []
    for path in sorted(dataset_root.iterdir()):
        cache_dir = path / "cache"
        if cache_dir.is_dir() and len(list(cache_dir.glob("*.npz"))) >= PREFIX_SIZE + CONTEXT_SIZE:
            dirs.append(path)
            if limit > 0 and len(dirs) >= limit:
                break
    return dirs


def load_frame(scene_dir: Path, cache_path: Path) -> FrameRecord:
    data = np.load(cache_path)
    return FrameRecord(
        scene=scene_dir.name,
        cache_path=cache_path,
        frame_id=int(cache_path.stem),
        depth=np.asarray(data["depth"], dtype=np.float32),
        c2w=np.asarray(data["pose"], dtype=np.float32),
        intrinsic=np.asarray(data["intrinsic"], dtype=np.float32),
    )


def load_scene_records(scene_dir: Path, max_frames: int) -> list[FrameRecord]:
    paths = sorted((scene_dir / "cache").glob("*.npz"))
    if max_frames > 0:
        paths = paths[:max_frames]
    return [load_frame(scene_dir, path) for path in paths]


def build_world_points(frame: FrameRecord, stride: int) -> np.ndarray:
    depth = frame.depth
    valid = depth > 1.0e-4
    ys, xs = np.nonzero(valid[::stride, ::stride])
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    ys = np.clip(ys * stride, 0, depth.shape[0] - 1)
    xs = np.clip(xs * stride, 0, depth.shape[1] - 1)
    z = depth[ys, xs]
    pixels = np.stack([xs.astype(np.float32), ys.astype(np.float32), np.ones_like(z)], axis=0)
    cam = (np.linalg.inv(frame.intrinsic).astype(np.float32) @ pixels) * z[None, :]
    pts = (frame.c2w[:3, :3].astype(np.float32) @ cam) + frame.c2w[:3, 3:4].astype(np.float32)
    return pts.T.astype(np.float32)


def directional_overlap(ref_pts: np.ndarray, cand: FrameRecord, rel_tol: float) -> float:
    if ref_pts.shape[0] == 0:
        return 0.0
    w2c = np.linalg.inv(cand.c2w).astype(np.float32)
    cam = (w2c[:3, :3] @ ref_pts.T) + w2c[:3, 3:4]
    z = cam[2]
    in_front = z > 1.0e-4
    if not np.any(in_front):
        return 0.0
    proj = cand.intrinsic.astype(np.float32) @ cam
    u = proj[0] / np.maximum(proj[2], 1.0e-8)
    v = proj[1] / np.maximum(proj[2], 1.0e-8)
    h, w = cand.depth.shape
    inside = in_front & (u >= 0.0) & (u <= float(w - 1)) & (v >= 0.0) & (v <= float(h - 1))
    if not np.any(inside):
        return 0.0
    ui = np.rint(u[inside]).astype(np.int32)
    vi = np.rint(v[inside]).astype(np.int32)
    depth_src = cand.depth[vi, ui]
    projected_depth = z[inside]
    visible = (depth_src > 1.0e-4) & (
        np.abs(depth_src - projected_depth) <= (float(rel_tol) * np.maximum(depth_src, 1.0e-4))
    )
    return float(visible.sum() / max(ref_pts.shape[0], 1))


def overlap_table(records: list[FrameRecord], stride: int, rel_tol: float) -> dict[int, dict[int, float]]:
    table: dict[int, dict[int, float]] = {}
    ref_points = [build_world_points(frame, stride=stride) for frame in records]
    for ref_idx, ref in enumerate(records):
        row: dict[int, float] = {}
        for cand_idx, cand in enumerate(records):
            if cand_idx == ref_idx:
                continue
            row[cand_idx] = directional_overlap(ref_points[ref_idx], cand, rel_tol=rel_tol)
        table[ref_idx] = row
    return table


def select_scene_tuple(
    scene_dir: Path,
    records: list[FrameRecord],
    table: dict[int, dict[int, float]],
    min_top5_overlap: float,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for ref_idx, scores in table.items():
        if len(scores) < PREFIX_SIZE + CONTEXT_SIZE - 1:
            continue
        high = sorted(scores.items(), key=lambda item: (-item[1], abs(item[0] - ref_idx), item[0]))
        prefix_support = [idx for idx, _ in high[: PREFIX_SIZE - 1]]
        clean_extra = [idx for idx, _ in high[PREFIX_SIZE - 1 : PREFIX_SIZE - 1 + CONTEXT_SIZE]]
        if len(prefix_support) < PREFIX_SIZE - 1 or len(clean_extra) < CONTEXT_SIZE:
            continue
        excluded = {ref_idx, *prefix_support}
        low = sorted(
            [(idx, ov) for idx, ov in scores.items() if idx not in excluded],
            key=lambda item: (item[1], abs(item[0] - ref_idx), item[0]),
        )
        if len(low) < CONTEXT_SIZE:
            continue
        top5_mean = float(np.mean([scores[idx] for idx in prefix_support]))
        if top5_mean < min_top5_overlap:
            continue
        low4 = [idx for idx, _ in low[:CONTEXT_SIZE]]
        low4_mean = float(np.mean([scores[idx] for idx in low4]))
        clean_extra_mean = float(np.mean([scores[idx] for idx in clean_extra]))
        score = top5_mean + clean_extra_mean - low4_mean
        payload = {
            "scene": scene_dir.name,
            "ref_index": int(ref_idx),
            "prefix_indices": [int(ref_idx), *[int(i) for i in prefix_support]],
            "clean_extra_indices": [int(i) for i in clean_extra],
            "tail_extra_indices": [int(i) for i in low4],
            "prefix_overlap_mean": top5_mean,
            "clean_context_overlap_mean": clean_extra_mean,
            "tail_context_overlap_mean": low4_mean,
            "score": float(score),
        }
        if best is None or payload["score"] > best["score"]:
            best = payload
    return best


def write_input_dir(input_dir: Path, frames: list[FrameRecord], overwrite: bool) -> None:
    if input_dir.exists() and not overwrite:
        raise FileExistsError(f"Input directory exists: {input_dir}")
    input_dir.mkdir(parents=True, exist_ok=True)
    for old in input_dir.glob("input_v*.png"):
        old.unlink()
    for index, frame in enumerate(frames):
        color = np.asarray(np.load(frame.cache_path)["color"], dtype=np.uint8)
        Image.fromarray(color).save(input_dir / f"input_v{index:02d}.png")


def frame_meta(frame: FrameRecord, role: str, overlap_to_ref: float | None) -> dict[str, Any]:
    return {
        "scene": frame.scene,
        "cache_path": str(frame.cache_path),
        "frame_id": int(frame.frame_id),
        "role": role,
        "overlap_to_ref": None if overlap_to_ref is None else float(overlap_to_ref),
    }


def prepare(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    input_root = Path(args.input_root).expanduser().resolve()
    input_root.mkdir(parents=True, exist_ok=True)

    scene_dirs = scene_cache_dirs(dataset_root, limit=int(args.limit_scenes_scan))
    if len(scene_dirs) < 2:
        raise RuntimeError(f"Need at least two BlendedMVG cache scenes under {dataset_root}")

    samples: list[dict[str, Any]] = []
    prepared_scene_records: dict[str, list[FrameRecord]] = {}
    prepared_tables: dict[str, dict[int, dict[int, float]]] = {}

    for scene_pos, scene_dir in enumerate(scene_dirs):
        if len(samples) >= int(args.num_samples):
            break
        records = load_scene_records(scene_dir, max_frames=int(args.max_frames_per_scene))
        if len(records) < PREFIX_SIZE + CONTEXT_SIZE:
            continue
        table = overlap_table(records, stride=int(args.overlap_stride), rel_tol=float(args.depth_rel_tol))
        selected = select_scene_tuple(
            scene_dir=scene_dir,
            records=records,
            table=table,
            min_top5_overlap=float(args.min_top5_overlap),
        )
        if selected is None:
            continue

        distractor_scene_dir = scene_dirs[(scene_pos + 1) % len(scene_dirs)]
        distractor_records = prepared_scene_records.get(distractor_scene_dir.name)
        if distractor_records is None:
            distractor_records = load_scene_records(distractor_scene_dir, max_frames=int(args.max_frames_per_scene))
            prepared_scene_records[distractor_scene_dir.name] = distractor_records
        if len(distractor_records) < CONTEXT_SIZE:
            continue

        prepared_scene_records[scene_dir.name] = records
        prepared_tables[scene_dir.name] = table
        sample_id = f"sample_{len(samples):02d}_{scene_dir.name}"
        prefix = [records[idx] for idx in selected["prefix_indices"]]
        clean_extra = [records[idx] for idx in selected["clean_extra_indices"]]
        tail_extra = [records[idx] for idx in selected["tail_extra_indices"]]
        xscene_extra = [
            distractor_records[min(i * max(len(distractor_records) // CONTEXT_SIZE, 1), len(distractor_records) - 1)]
            for i in range(CONTEXT_SIZE)
        ]

        variants: dict[str, Any] = {}
        for variant_name, ordered_frames in {
            "clean": [*prefix, *clean_extra],
            "tail": [*prefix, *tail_extra],
            "xscene": [*prefix, *xscene_extra],
        }.items():
            variant_dir = input_root / sample_id / variant_name
            write_input_dir(variant_dir, ordered_frames, overwrite=bool(args.overwrite))
            frame_rows: list[dict[str, Any]] = []
            for out_idx, frame in enumerate(ordered_frames):
                if out_idx == 0:
                    role = "ref"
                    overlap = 1.0
                elif out_idx < PREFIX_SIZE:
                    role = "prefix_support"
                    overlap = table[selected["ref_index"]].get(selected["prefix_indices"][out_idx])
                elif variant_name == "clean":
                    role = "clean_context"
                    overlap = table[selected["ref_index"]].get(selected["clean_extra_indices"][out_idx - PREFIX_SIZE])
                elif variant_name == "tail":
                    role = "tail_context"
                    overlap = table[selected["ref_index"]].get(selected["tail_extra_indices"][out_idx - PREFIX_SIZE])
                else:
                    role = "cross_scene_context"
                    overlap = None
                frame_rows.append(frame_meta(frame, role=role, overlap_to_ref=overlap))
            variants[variant_name] = {
                "input_dir": str(variant_dir),
                "frames": frame_rows,
            }

        samples.append(
            {
                "sample_id": sample_id,
                "source_scene": scene_dir.name,
                "distractor_scene": distractor_scene_dir.name,
                "prefix_size": PREFIX_SIZE,
                "context_size": CONTEXT_SIZE,
                "selection": selected,
                "variants": variants,
            }
        )

    if len(samples) < int(args.num_samples):
        raise RuntimeError(f"Prepared only {len(samples)} samples; requested {args.num_samples}")

    protocol = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": "blendedmvg_prefix_point_degradation_v1",
        "dataset_root": str(dataset_root),
        "input_root": str(input_root),
        "prefix_size": PREFIX_SIZE,
        "context_size": CONTEXT_SIZE,
        "variants": list(VARIANTS),
        "selection_config": {
            "num_samples": int(args.num_samples),
            "max_frames_per_scene": int(args.max_frames_per_scene),
            "overlap_stride": int(args.overlap_stride),
            "depth_rel_tol": float(args.depth_rel_tol),
            "min_top5_overlap": float(args.min_top5_overlap),
            "limit_scenes_scan": int(args.limit_scenes_scan),
        },
        "samples": samples,
    }
    protocol_path = input_root / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[prepare] wrote {protocol_path}")
    print(f"[prepare] samples={len(samples)}")


def emit_jobs(args: argparse.Namespace) -> None:
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    output_root = Path(args.output_root).expanduser().resolve()
    writer = csv.writer(__import__("sys").stdout, delimiter="\t", lineterminator="\n")
    for sample in protocol["samples"]:
        sample_id = sample["sample_id"]
        for variant in VARIANTS:
            input_dir = sample["variants"][variant]["input_dir"]
            for model in MODELS:
                out_dir = output_root / model / sample_id / variant
                writer.writerow([model, sample_id, variant, input_dir, str(out_dir)])


def subsample_mask(shape: tuple[int, int, int, int], stride: int) -> np.ndarray:
    mask = np.zeros(shape[:3], dtype=bool)
    mask[:, :: max(stride, 1), :: max(stride, 1)] = True
    return mask


def gt_points_from_cache(cache_path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(cache_path)
    depth = np.asarray(data["depth"], dtype=np.float32)
    c2w = np.asarray(data["pose"], dtype=np.float32)
    intrinsic = np.asarray(data["intrinsic"], dtype=np.float32)
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    z = depth
    pixels = np.stack([xs.reshape(-1), ys.reshape(-1), np.ones(h * w, dtype=np.float32)], axis=0)
    cam = (np.linalg.inv(intrinsic).astype(np.float32) @ pixels) * z.reshape(1, -1)
    pts = (c2w[:3, :3].astype(np.float32) @ cam) + c2w[:3, 3:4].astype(np.float32)
    pts = pts.T.reshape(h, w, 3).astype(np.float32)
    valid = (depth > 1.0e-4) & np.isfinite(pts).all(axis=-1)
    return pts, valid


def finite_points(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1) & (np.abs(points).max(axis=-1) < 1.0e6)


def flatten_metric_points(
    points: np.ndarray,
    valid: np.ndarray,
    stride: int,
    max_points: int,
) -> np.ndarray:
    sampled = subsample_mask(points.shape, stride=stride) & valid
    flat = points[sampled].reshape(-1, 3)
    if max_points > 0 and flat.shape[0] > max_points:
        indices = np.linspace(0, flat.shape[0] - 1, max_points).astype(np.int64)
        flat = flat[indices]
    return flat.astype(np.float64, copy=False)


def sim3_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
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
    aligned = (src @ r) * scale + (tgt_mu - scale * (src_mu @ r))
    target_rms = float(np.sqrt(np.mean(np.sum(tgt_c * tgt_c, axis=1))))
    return aligned, {"scale": scale, "target_rms": max(target_rms, 1.0e-12), "n_points": float(n)}


def sim3_error(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    aligned, meta = sim3_align(source, target)
    tgt = target[: aligned.shape[0]]
    dist = np.linalg.norm(aligned - tgt, axis=1)
    norm = meta["target_rms"]
    return {
        "mean": float(np.mean(dist)),
        "median": float(np.median(dist)),
        "p90": float(np.percentile(dist, 90.0)),
        "mean_norm": float(np.mean(dist / norm)),
        "median_norm": float(np.median(dist / norm)),
        "p90_norm": float(np.percentile(dist / norm, 90.0)),
        "n_points": int(meta["n_points"]),
        "align_scale": float(meta["scale"]),
        "target_rms": float(norm),
    }


def load_prefix_pred(output_dir: Path, prefix_size: int) -> np.ndarray:
    npz_path = output_dir / "points.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    points = np.asarray(np.load(npz_path)["points"], dtype=np.float32)
    if points.shape[0] < prefix_size:
        raise RuntimeError(f"Not enough predicted views in {npz_path}: {points.shape}")
    return points[:prefix_size]


def load_prefix_gt(sample: dict[str, Any], variant: str, prefix_size: int) -> tuple[np.ndarray, np.ndarray]:
    pts_list = []
    valid_list = []
    for frame in sample["variants"][variant]["frames"][:prefix_size]:
        pts, valid = gt_points_from_cache(frame["cache_path"])
        pts_list.append(pts)
        valid_list.append(valid)
    return np.stack(pts_list, axis=0), np.stack(valid_list, axis=0)


def mean_or_nan(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.mean(vals)) if vals else float("nan")


def summarize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    prefix_size = int(protocol.get("prefix_size", PREFIX_SIZE))

    rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    pred_cache: dict[tuple[str, str, str], np.ndarray] = {}

    for sample in protocol["samples"]:
        sample_id = sample["sample_id"]
        for model in MODELS:
            for variant in VARIANTS:
                out_dir = output_root / model / sample_id / variant
                row: dict[str, Any] = {
                    "sample_id": sample_id,
                    "model": model,
                    "variant": variant,
                    "output_dir": str(out_dir),
                    "status": "ok",
                }
                try:
                    pred = load_prefix_pred(out_dir, prefix_size=prefix_size)
                    pred_cache[(model, sample_id, variant)] = pred
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
                    err = sim3_error(pred_flat, gt_flat)
                    for key, value in err.items():
                        row[f"gt_prefix_{key}"] = value
                except Exception as exc:  # keep partial summaries useful
                    row["status"] = f"failed:{type(exc).__name__}:{exc}"
                rows.append(row)

            clean_pred = pred_cache.get((model, sample_id, "clean"))
            if clean_pred is None:
                continue
            for variant in ("tail", "xscene"):
                var_pred = pred_cache.get((model, sample_id, variant))
                if var_pred is None:
                    continue
                valid = finite_points(clean_pred) & finite_points(var_pred)
                clean_flat = flatten_metric_points(
                    clean_pred,
                    valid=valid,
                    stride=int(args.metric_stride),
                    max_points=int(args.max_metric_points),
                )
                var_flat = flatten_metric_points(
                    var_pred,
                    valid=valid,
                    stride=int(args.metric_stride),
                    max_points=int(args.max_metric_points),
                )
                err = sim3_error(var_flat, clean_flat)
                paired = {
                    "sample_id": sample_id,
                    "model": model,
                    "perturbation": variant,
                    "status": "ok",
                }
                for key, value in err.items():
                    paired[f"paired_prefix_{key}"] = value
                paired_rows.append(paired)

    summary_dir = output_root / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    gt_csv = summary_dir / "gt_prefix_metrics.csv"
    paired_csv = summary_dir / "paired_prefix_degradation.csv"
    write_csv(gt_csv, rows)
    write_csv(paired_csv, paired_rows)

    aggregate: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": str(protocol_path),
        "output_root": str(output_root),
        "metric_stride": int(args.metric_stride),
        "num_gt_rows": len(rows),
        "num_paired_rows": len(paired_rows),
        "gt_prefix_by_model_variant": [],
        "paired_prefix_by_model_perturbation": [],
        "gt_delta_vs_clean": [],
        "overlap_stats": overlap_stats(protocol),
    }

    for model in MODELS:
        for variant in VARIANTS:
            subset = [r for r in rows if r.get("model") == model and r.get("variant") == variant and r.get("status") == "ok"]
            aggregate["gt_prefix_by_model_variant"].append(
                {
                    "model": model,
                    "variant": variant,
                    "n": len(subset),
                    "mean_norm": mean_or_nan(r.get("gt_prefix_mean_norm", float("nan")) for r in subset),
                    "median_norm": mean_or_nan(r.get("gt_prefix_median_norm", float("nan")) for r in subset),
                    "p90_norm": mean_or_nan(r.get("gt_prefix_p90_norm", float("nan")) for r in subset),
                }
            )
        for perturbation in ("tail", "xscene"):
            subset = [
                r
                for r in paired_rows
                if r.get("model") == model and r.get("perturbation") == perturbation and r.get("status") == "ok"
            ]
            aggregate["paired_prefix_by_model_perturbation"].append(
                {
                    "model": model,
                    "perturbation": perturbation,
                    "n": len(subset),
                    "mean_norm": mean_or_nan(r.get("paired_prefix_mean_norm", float("nan")) for r in subset),
                    "median_norm": mean_or_nan(r.get("paired_prefix_median_norm", float("nan")) for r in subset),
                    "p90_norm": mean_or_nan(r.get("paired_prefix_p90_norm", float("nan")) for r in subset),
                }
            )
            deltas = []
            for sample in protocol["samples"]:
                sid = sample["sample_id"]
                clean = next(
                    (
                        r
                        for r in rows
                        if r.get("model") == model and r.get("sample_id") == sid and r.get("variant") == "clean"
                    ),
                    None,
                )
                pert = next(
                    (
                        r
                        for r in rows
                        if r.get("model") == model and r.get("sample_id") == sid and r.get("variant") == perturbation
                    ),
                    None,
                )
                if clean and pert and clean.get("status") == "ok" and pert.get("status") == "ok":
                    deltas.append(float(pert["gt_prefix_mean_norm"]) - float(clean["gt_prefix_mean_norm"]))
            aggregate["gt_delta_vs_clean"].append(
                {
                    "model": model,
                    "perturbation": perturbation,
                    "n": len(deltas),
                    "mean_delta_norm": mean_or_nan(deltas),
                    "median_delta_norm": float(np.median(deltas)) if deltas else float("nan"),
                }
            )

    json_path = summary_dir / "summary.json"
    md_path = summary_dir / "summary.md"
    json_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(aggregate, gt_csv, paired_csv), encoding="utf-8")
    print(f"[summarize] wrote {json_path}")
    print(f"[summarize] wrote {md_path}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def overlap_stats(protocol: dict[str, Any]) -> dict[str, float]:
    prefix = []
    clean = []
    tail = []
    for sample in protocol["samples"]:
        sel = sample["selection"]
        prefix.append(float(sel["prefix_overlap_mean"]))
        clean.append(float(sel["clean_context_overlap_mean"]))
        tail.append(float(sel["tail_context_overlap_mean"]))
    return {
        "prefix_overlap_mean": mean_or_nan(prefix),
        "clean_context_overlap_mean": mean_or_nan(clean),
        "tail_context_overlap_mean": mean_or_nan(tail),
    }


def render_markdown(aggregate: dict[str, Any], gt_csv: Path, paired_csv: Path) -> str:
    lines = [
        "# BlendedMVG Prefix Point-Degradation Summary",
        "",
        f"Protocol: `{aggregate['protocol']}`",
        f"Output root: `{aggregate['output_root']}`",
        f"Metric stride: {aggregate['metric_stride']}",
        "",
        "Overlap means used for protocol selection:",
    ]
    for key, value in aggregate["overlap_stats"].items():
        lines.append(f"- {key}: {value:.6f}")
    lines.extend(["", "Paired prefix prediction change after Sim(3) alignment, lower is better:", ""])
    lines.append("| model | perturbation | n | mean_norm | median_norm | p90_norm |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in aggregate["paired_prefix_by_model_perturbation"]:
        lines.append(
            f"| {row['model']} | {row['perturbation']} | {row['n']} | "
            f"{row['mean_norm']:.6f} | {row['median_norm']:.6f} | {row['p90_norm']:.6f} |"
        )
    lines.extend(["", "GT prefix point error after Sim(3) alignment, lower is better:", ""])
    lines.append("| model | variant | n | mean_norm | median_norm | p90_norm |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in aggregate["gt_prefix_by_model_variant"]:
        lines.append(
            f"| {row['model']} | {row['variant']} | {row['n']} | "
            f"{row['mean_norm']:.6f} | {row['median_norm']:.6f} | {row['p90_norm']:.6f} |"
        )
    lines.extend(["", "GT prefix error delta vs clean context, lower is better:", ""])
    lines.append("| model | perturbation | n | mean_delta_norm | median_delta_norm |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in aggregate["gt_delta_vs_clean"]:
        lines.append(
            f"| {row['model']} | {row['perturbation']} | {row['n']} | "
            f"{row['mean_delta_norm']:.6f} | {row['median_delta_norm']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Detailed GT CSV: `{gt_csv}`",
            f"Detailed paired CSV: `{paired_csv}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "emit-jobs":
        emit_jobs(args)
    elif args.cmd == "summarize":
        summarize(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
