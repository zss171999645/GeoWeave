#!/usr/bin/env python3
"""Compute GT-backed point-map accuracy for reproduced rebuttal datasets."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from blendedmvg_pointcloud_metrics import apply_sim3, compute_aligned_cloud_metrics, estimate_sim3_transform


WAYMO_DEPTH_SCALE = 256.0


@dataclass(frozen=True)
class WaymoSourceIndex:
    gt_root: Path
    mapping: dict[tuple[str, int], Path]

    def resolve(self, *, short_scene: str | int, cam_zero_based: str | int) -> Path:
        scene = str(short_scene)
        cam_n = int(cam_zero_based) + 1
        key = (scene, cam_n)
        if key not in self.mapping:
            known = ", ".join(f"{s}:cam{c}" for s, c in sorted(self.mapping)[:12])
            raise KeyError(f"Waymo source not found for {scene} cam{cam_zero_based}; known starts with {known}")
        return self.mapping[key]


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_waymo_source_index(gt_root: str | Path) -> WaymoSourceIndex:
    root = Path(gt_root).expanduser().resolve()
    manifest = load_json(root / "manifest.json")
    mapping: dict[tuple[str, int], Path] = {}
    for item in manifest.get("waymo_required_pairs", []):
        source_dir = Path(str(item["source_dir"]))
        if source_dir.parts[:3] != ("/", "mnt", "bos"):
            raise ValueError(f"Unexpected Waymo source_dir outside /mnt/bos: {source_dir}")
        relative = Path(*source_dir.parts[3:])
        mapping[(str(item["short_scene"]), int(item["camN"]))] = root / relative
    return WaymoSourceIndex(gt_root=root, mapping=mapping)


def scale_intrinsic(k: np.ndarray, *, source_hw: tuple[int, int], target_hw: tuple[int, int]) -> np.ndarray:
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    scaled = np.asarray(k, dtype=np.float64).copy()
    scaled[0, 0] *= float(target_w) / float(source_w)
    scaled[0, 2] *= float(target_w) / float(source_w)
    scaled[1, 1] *= float(target_h) / float(source_h)
    scaled[1, 2] *= float(target_h) / float(source_h)
    return scaled


def normalized_intrinsic_to_pixels(normalized: np.ndarray, *, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    k = np.asarray(normalized, dtype=np.float64).copy()
    k[0, 0] *= float(target_w)
    k[0, 2] *= float(target_w)
    k[1, 1] *= float(target_h)
    k[1, 2] *= float(target_h)
    return k


def depth_to_world_points(depth: np.ndarray, intrinsic: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    depth_arr = np.asarray(depth, dtype=np.float64)
    k = np.asarray(intrinsic, dtype=np.float64)
    pose = np.asarray(c2w, dtype=np.float64)
    h, w = depth_arr.shape
    ys, xs = np.indices((h, w), dtype=np.float64)
    z = depth_arr
    x = (xs - k[0, 2]) / k[0, 0] * z
    y = (ys - k[1, 2]) / k[1, 1] * z
    cam = np.stack([x, y, z], axis=-1)
    world = cam @ pose[:3, :3].T + pose[:3, 3]
    return world.astype(np.float32, copy=False)


def world_points_to_camera_depth(points_world: np.ndarray, c2w_stack: np.ndarray) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64)
    poses = np.asarray(c2w_stack, dtype=np.float64)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"Expected world point maps shaped (V,H,W,3), got {points.shape}")
    if poses.shape != (points.shape[0], 4, 4):
        raise ValueError(f"Expected c2w stack shaped ({points.shape[0]},4,4), got {poses.shape}")
    depth = np.empty(points.shape[:-1], dtype=np.float64)
    for view_index, pose in enumerate(poses):
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        camera_points = (points[view_index] - translation) @ rotation
        depth[view_index] = camera_points[..., 2]
    return depth


def compute_aligned_depth_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    *,
    gt_c2w: np.ndarray,
    valid: np.ndarray,
    metric_stride: int,
    max_metric_points: int,
) -> dict[str, Any]:
    pred = np.asarray(pred_points, dtype=np.float64)
    gt = np.asarray(gt_points, dtype=np.float64)
    valid_mask = np.asarray(valid, dtype=bool)
    pred_flat = flatten_metric_points(
        pred,
        valid=valid_mask,
        stride=int(metric_stride),
        max_points=int(max_metric_points),
    )
    gt_flat = flatten_metric_points(
        gt,
        valid=valid_mask,
        stride=int(metric_stride),
        max_points=int(max_metric_points),
    )
    if len(pred_flat) < 8 or len(gt_flat) < 8:
        return {"depth_status": f"failed:not_enough_alignment_points:{len(pred_flat)}"}

    try:
        transform = estimate_sim3_transform(pred_flat, gt_flat)
        aligned_pred = apply_sim3(pred.reshape(-1, 3), transform).reshape(pred.shape)
        pred_depth = world_points_to_camera_depth(aligned_pred, gt_c2w)
        gt_depth = world_points_to_camera_depth(gt, gt_c2w)
    except Exception as exc:
        return {"depth_status": f"failed:{type(exc).__name__}:{exc}"}

    depth_valid = (
        valid_mask
        & np.isfinite(pred_depth)
        & np.isfinite(gt_depth)
        & (pred_depth > 1.0e-6)
        & (gt_depth > 1.0e-6)
    )
    pred_values = pred_depth.reshape(-1)[depth_valid.reshape(-1)]
    gt_values = gt_depth.reshape(-1)[depth_valid.reshape(-1)]
    step = max(int(metric_stride), 1)
    if step > 1:
        pred_values = pred_values[::step]
        gt_values = gt_values[::step]
    if len(pred_values) > int(max_metric_points):
        positions = np.linspace(0, len(pred_values) - 1, int(max_metric_points), dtype=np.int64)
        pred_values = pred_values[positions]
        gt_values = gt_values[positions]
    if len(pred_values) < 8:
        return {"depth_status": f"failed:not_enough_valid_depth:{len(pred_values)}"}

    residual = pred_values - gt_values
    ratio = np.maximum(pred_values / gt_values, gt_values / pred_values)
    gt_rms = float(np.sqrt(np.mean(gt_values * gt_values)))
    return {
        "depth_status": "ok",
        "depth_valid_pixels": int(len(pred_values)),
        "depth_abs_rel": float(np.mean(np.abs(residual) / gt_values)),
        "depth_rmse": float(np.sqrt(np.mean(residual * residual))),
        "depth_rmse_norm": float(np.sqrt(np.mean(residual * residual)) / max(gt_rms, 1.0e-12)),
        "depth_delta1": float(np.mean(ratio < 1.25)),
    }


def resolve_eval_indices(*, prefix_size: int, total_views: int, mode: str) -> list[int]:
    if mode == "prefix":
        return list(range(min(int(prefix_size), int(total_views))))
    if mode == "all":
        return list(range(int(total_views)))
    raise ValueError(f"Unsupported eval mode: {mode}")


def resize_nearest(array: np.ndarray, *, target_hw: tuple[int, int]) -> np.ndarray:
    if tuple(array.shape[:2]) == tuple(target_hw):
        return array
    image = Image.fromarray(array)
    resized = image.resize((target_hw[1], target_hw[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(resized)


def load_pose_stack(path: str | Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] == 16:
        return arr.reshape(-1, 4, 4)
    if arr.shape == (4, 4):
        return arr.reshape(1, 4, 4)
    raise ValueError(f"Unsupported pose matrix shape {arr.shape} at {path}")


def load_single_pose(path: str | Path) -> np.ndarray:
    pose = np.loadtxt(path, dtype=np.float64)
    if pose.shape != (4, 4):
        pose = pose.reshape(4, 4)
    return pose


def load_scannet_normalized_intrinsics(path: str | Path) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        rows = [ast.literal_eval(line) for line in text.splitlines() if line.strip()]
        arr = np.asarray(rows, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[1:] != (3, 3):
            raise ValueError(f"Unsupported ScanNet++ literal intrinsic shape {arr.shape}: {path}")
        return arr
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 9:
        raise ValueError(f"Unsupported ScanNet++ intrinsic table shape {arr.shape}: {path}")
    return arr.reshape(-1, 3, 3)


def load_waymo_intrinsic(path: str | Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if len(values) < 4:
        raise ValueError(f"Waymo intrinsic file has fewer than 4 values: {path}")
    fx, fy, cx, cy = values[:4]
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_exr_depth(path: str | Path) -> np.ndarray:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2  # type: ignore

    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read EXR depth: {path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return np.asarray(depth, dtype=np.float32)


def load_waymo_depth_png(path: str | Path) -> np.ndarray:
    raw = np.asarray(Image.open(path), dtype=np.float32)
    return raw / WAYMO_DEPTH_SCALE


def finite_valid_depth(depth: np.ndarray) -> np.ndarray:
    return np.isfinite(depth) & (depth > 1.0e-6)


def load_points_npz(path: str | Path) -> np.ndarray:
    with np.load(path) as data:
        if "points" not in data:
            raise KeyError(f"points key missing in {path}; keys={list(data.keys())}")
        points = np.asarray(data["points"], dtype=np.float32)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"Expected point map shape (V,H,W,3), got {points.shape} at {path}")
    return points


def sample_id_wrong_waymo(sample_id: str) -> tuple[str, str]:
    match = re.search(r"wrong_waymo-(\d+)-cam(\d+)", sample_id)
    if match is None:
        raise ValueError(f"Could not parse wrong Waymo scene/camera from sample_id: {sample_id}")
    return match.group(1), match.group(2)


def waymo_source_for_output_index(
    *,
    setting: str,
    sample: dict[str, Any],
    sample_id: str,
    variant: str,
    output_index: int,
) -> tuple[str, str, int]:
    meta = sample["metadata"]
    if setting == "waymo_weak":
        token = str(meta["ordered_frame_names"][output_index])
        cam, frame = token.split(":", 1)
        return str(meta["scene_name"]), cam, int(frame)

    if setting != "waymo_plausible":
        raise ValueError(f"Unsupported Waymo setting: {setting}")

    prefix_size = int(sample.get("prefix_size", 6))
    if output_index < prefix_size:
        return str(meta["prefix_scene"]), str(meta["prefix_camera"]), int(meta["prefix_frame_names"][output_index])

    tail_idx = output_index - prefix_size
    if variant == "clean_tail":
        return str(meta["prefix_scene"]), str(meta["prefix_camera"]), int(meta["context_frame_names"][tail_idx])
    if variant == "plausible_noise_tail":
        wrong_scene, wrong_cam = sample_id_wrong_waymo(sample_id)
        return wrong_scene, wrong_cam, int(meta["context_frame_names"][tail_idx])
    raise ValueError(f"Unsupported Waymo plausible variant: {variant}")


def attach_sample_metadata(protocol: dict[str, Any]) -> None:
    for sample in protocol.get("samples", []):
        variants = sample.get("variants", {})
        first_variant = next(iter(variants.values()), None)
        if first_variant is None:
            sample["metadata"] = {}
            continue
        sequence_dir = Path(first_variant.get("sequence_dir") or first_variant.get("input_dir")).expanduser().resolve()
        meta_path = sequence_dir / "tuple_meta.json"
        sample["metadata"] = load_json(meta_path) if meta_path.is_file() else {}


def infer_sequence_dir(sample: dict[str, Any], variant: str) -> Path:
    payload = sample["variants"][variant]
    return Path(payload.get("sequence_dir") or payload["input_dir"]).expanduser().resolve()


def scannet_scene_id(sample: dict[str, Any], sequence_dir: Path) -> str:
    meta = sample["metadata"]
    if "source_scene_root" in meta:
        return Path(str(meta["source_scene_root"])).name
    if "scene_name" in meta:
        return str(meta["scene_name"]).split("-")[-1]
    match = re.search(r"scannetpp-([0-9a-f]+)", sequence_dir.name)
    if match is None:
        raise ValueError(f"Could not infer ScanNet++ scene id from {sequence_dir}")
    return match.group(1)


def build_scannet_gt_stack(
    *,
    sample: dict[str, Any],
    variant: str,
    gt_root: Path,
    pred_shape: tuple[int, int, int, int],
    eval_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    sequence_dir = infer_sequence_dir(sample, variant)
    meta = sample["metadata"]
    scene = scannet_scene_id(sample, sequence_dir)
    source_scene_dir = gt_root / "scannetpp/scannetpplus/Scannetpp/data" / scene
    intrinsic_table = load_scannet_normalized_intrinsics(source_scene_dir / f"{scene}_intrinsic.txt")
    poses = load_pose_stack(sequence_dir / "pose_90.txt")
    target_hw = (int(pred_shape[1]), int(pred_shape[2]))
    gt_points: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    c2w_poses: list[np.ndarray] = []
    source_rows: list[dict[str, Any]] = []
    frames = meta.get("frames", [])
    for out_index in eval_indices:
        if out_index >= len(frames):
            raise IndexError(f"ScanNet++ frame index {out_index} outside metadata frames length {len(frames)}")
        frame_id = int(frames[out_index]["frame_id"])
        depth_path = sequence_dir / "depth_90" / f"frame_{out_index:04d}.exr"
        depth = load_exr_depth(depth_path)
        source_hw = tuple(depth.shape[:2])
        depth = resize_nearest(depth, target_hw=target_hw).astype(np.float32, copy=False)
        if frame_id >= len(intrinsic_table):
            raise IndexError(f"ScanNet++ frame_id {frame_id} outside intrinsic table length {len(intrinsic_table)}")
        k = normalized_intrinsic_to_pixels(intrinsic_table[frame_id], target_hw=target_hw)
        c2w = poses[out_index]
        points = depth_to_world_points(depth, k, c2w)
        valid = finite_valid_depth(depth)
        gt_points.append(points)
        valid_masks.append(valid)
        c2w_poses.append(c2w)
        source_rows.append(
            {
                "view_index": int(out_index),
                "scene": scene,
                "frame_id": frame_id,
                "depth_path": str(depth_path),
                "intrinsic_path": str(source_scene_dir / f"{scene}_intrinsic.txt"),
                "pose_path": str(sequence_dir / "pose_90.txt"),
                "source_hw": list(source_hw),
                "target_hw": list(target_hw),
            }
        )
    return np.stack(gt_points, axis=0), np.stack(valid_masks, axis=0), np.stack(c2w_poses, axis=0), source_rows


def build_waymo_gt_stack(
    *,
    setting: str,
    sample: dict[str, Any],
    sample_id: str,
    variant: str,
    waymo_index: WaymoSourceIndex,
    pred_shape: tuple[int, int, int, int],
    eval_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    target_hw = (int(pred_shape[1]), int(pred_shape[2]))
    gt_points: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    c2w_poses: list[np.ndarray] = []
    source_rows: list[dict[str, Any]] = []
    for out_index in eval_indices:
        scene, cam_zero_based, frame_id = waymo_source_for_output_index(
            setting=setting,
            sample=sample,
            sample_id=sample_id,
            variant=variant,
            output_index=out_index,
        )
        source_dir = waymo_index.resolve(short_scene=scene, cam_zero_based=cam_zero_based)
        depth_path = source_dir / "depth" / f"{frame_id:03d}.png"
        pose_path = source_dir / "pose" / f"{frame_id:03d}.txt"
        intrinsic_path = source_dir / "intrinsic.txt"
        depth = load_waymo_depth_png(depth_path)
        source_hw = tuple(depth.shape[:2])
        k = scale_intrinsic(load_waymo_intrinsic(intrinsic_path), source_hw=source_hw, target_hw=target_hw)
        depth = resize_nearest(depth, target_hw=target_hw).astype(np.float32, copy=False)
        c2w = load_single_pose(pose_path)
        points = depth_to_world_points(depth, k, c2w)
        valid = finite_valid_depth(depth)
        gt_points.append(points)
        valid_masks.append(valid)
        c2w_poses.append(c2w)
        source_rows.append(
            {
                "view_index": int(out_index),
                "scene": scene,
                "cam_zero_based": str(cam_zero_based),
                "camN": int(cam_zero_based) + 1,
                "frame_id": int(frame_id),
                "depth_path": str(depth_path),
                "intrinsic_path": str(intrinsic_path),
                "pose_path": str(pose_path),
                "source_hw": list(source_hw),
                "target_hw": list(target_hw),
            }
        )
    return np.stack(gt_points, axis=0), np.stack(valid_masks, axis=0), np.stack(c2w_poses, axis=0), source_rows


def flatten_metric_points(
    points: np.ndarray,
    *,
    valid: np.ndarray,
    stride: int,
    max_points: int,
) -> np.ndarray:
    step = max(int(stride), 1)
    flat = points.reshape(-1, 3)
    flat_valid = valid.reshape(-1)
    sampled = flat[flat_valid & np.isfinite(flat).all(axis=-1)]
    if step > 1:
        sampled = sampled[::step]
    if len(sampled) > int(max_points):
        positions = np.linspace(0, len(sampled) - 1, int(max_points), dtype=np.int64)
        sampled = sampled[positions]
    return sampled.astype(np.float64, copy=False)


def mean_or_nan(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.mean(vals)) if vals else float("nan")


def format_float(value: float) -> str:
    return "nan" if not math.isfinite(float(value)) else f"{float(value):.6f}"


def compute_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = load_json(args.protocol)
    attach_sample_metadata(protocol)
    gt_root = Path(args.gt_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    waymo_index = load_waymo_source_index(gt_root) if args.setting.startswith("waymo") else None
    samples = list(protocol.get("samples", []))
    if int(args.sample_limit) > 0:
        samples = samples[: int(args.sample_limit)]
    rows: list[dict[str, Any]] = []
    gt_sources: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        variants = [v for v in args.variants if v in sample.get("variants", {})]
        prefix_size = int(sample.get("prefix_size", protocol.get("prefix_size", args.default_prefix_size)))
        for model in args.models:
            for variant in variants:
                points_path = output_root / model / sample_id / variant / "points.npz"
                row: dict[str, Any] = {
                    "sample_id": sample_id,
                    "model": model,
                    "variant": variant,
                    "setting": args.setting,
                    "eval_mode": args.eval_mode,
                    "points_path": str(points_path),
                }
                if not points_path.is_file():
                    row["status"] = "missing_points"
                    rows.append(row)
                    continue
                try:
                    pred_all = load_points_npz(points_path)
                    eval_indices = resolve_eval_indices(
                        prefix_size=prefix_size,
                        total_views=int(pred_all.shape[0]),
                        mode=args.eval_mode,
                    )
                    pred = pred_all[eval_indices]
                    if args.setting == "scannetpp_weak":
                        gt, gt_valid, gt_c2w, source_rows = build_scannet_gt_stack(
                            sample=sample,
                            variant=variant,
                            gt_root=gt_root,
                            pred_shape=pred.shape,
                            eval_indices=eval_indices,
                        )
                    elif args.setting in {"waymo_weak", "waymo_plausible"}:
                        if waymo_index is None:
                            raise RuntimeError("Waymo source index was not loaded")
                        gt, gt_valid, gt_c2w, source_rows = build_waymo_gt_stack(
                            setting=args.setting,
                            sample=sample,
                            sample_id=sample_id,
                            variant=variant,
                            waymo_index=waymo_index,
                            pred_shape=pred.shape,
                            eval_indices=eval_indices,
                        )
                    else:
                        raise ValueError(f"Unsupported setting: {args.setting}")
                    valid = gt_valid & np.isfinite(pred).all(axis=-1)
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
                    metrics = compute_aligned_cloud_metrics(
                        pred_flat,
                        gt_flat,
                        normal_metrics=bool(args.normal_metrics),
                    )
                    row.update(metrics)
                    if bool(args.depth_metrics):
                        depth_metrics = compute_aligned_depth_metrics(
                            pred,
                            gt,
                            gt_c2w=gt_c2w,
                            valid=valid,
                            metric_stride=int(args.metric_stride),
                            max_metric_points=int(args.max_metric_points),
                        )
                        row.update(depth_metrics)
                        if depth_metrics.get("depth_status") != "ok":
                            row["status"] = f"failed:{depth_metrics.get('depth_status')}"
                    row.update(
                        {
                            "n_views": int(len(eval_indices)),
                            "eval_indices": ",".join(str(i) for i in eval_indices),
                            "valid_pixel_ratio": float(valid.mean()),
                            "pred_shape": "x".join(str(x) for x in pred.shape),
                            "gt_shape": "x".join(str(x) for x in gt.shape),
                        }
                    )
                    for source in source_rows:
                        gt_sources.append(
                            {
                                "sample_id": sample_id,
                                "model": model,
                                "variant": variant,
                                "setting": args.setting,
                                **source,
                            }
                        )
                except Exception as exc:
                    row["status"] = f"failed:{type(exc).__name__}:{exc}"
                rows.append(row)
    return rows, gt_sources


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("model")), str(row.get("variant"))), []).append(row)
    aggregate: list[dict[str, Any]] = []
    for (model, variant), group_rows in sorted(groups.items()):
        ok = [row for row in group_rows if row.get("status") == "ok"]
        aggregate.append(
            {
                "model": model,
                "variant": variant,
                "n_total": len(group_rows),
                "n_ok": len(ok),
                "n_missing_points": sum(1 for row in group_rows if row.get("status") == "missing_points"),
                "n_failed": sum(
                    1
                    for row in group_rows
                    if str(row.get("status", "")).startswith("failed:") and row.get("status") != "missing_points"
                ),
                "acc_mean_norm": mean_or_nan(row.get("acc_mean_norm", float("nan")) for row in ok),
                "comp_mean_norm": mean_or_nan(row.get("comp_mean_norm", float("nan")) for row in ok),
                "acc_mean": mean_or_nan(row.get("acc_mean", float("nan")) for row in ok),
                "comp_mean": mean_or_nan(row.get("comp_mean", float("nan")) for row in ok),
                "nc_acc_mean": mean_or_nan(row.get("nc_acc_mean", float("nan")) for row in ok),
                "nc_comp_mean": mean_or_nan(row.get("nc_comp_mean", float("nan")) for row in ok),
                "nc_mean": mean_or_nan(row.get("nc_mean", float("nan")) for row in ok),
                "depth_abs_rel": mean_or_nan(row.get("depth_abs_rel", float("nan")) for row in ok),
                "depth_rmse": mean_or_nan(row.get("depth_rmse", float("nan")) for row in ok),
                "depth_rmse_norm": mean_or_nan(row.get("depth_rmse_norm", float("nan")) for row in ok),
                "depth_delta1": mean_or_nan(row.get("depth_delta1", float("nan")) for row in ok),
                "depth_valid_pixels": mean_or_nan(row.get("depth_valid_pixels", float("nan")) for row in ok),
                "target_rms": mean_or_nan(row.get("target_rms", float("nan")) for row in ok),
                "valid_pixel_ratio": mean_or_nan(row.get("valid_pixel_ratio", float("nan")) for row in ok),
                "n_points": mean_or_nan(row.get("n_points", float("nan")) for row in ok),
            }
        )
    return {
        "aggregate": aggregate,
        "num_rows": len(rows),
        "num_ok": sum(1 for row in rows if row.get("status") == "ok"),
        "num_missing_points": sum(1 for row in rows if row.get("status") == "missing_points"),
        "num_failed": sum(1 for row in rows if str(row.get("status", "")).startswith("failed:")),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path: Path, *, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    lines = [
        f"# {args.summary_name}",
        "",
        f"- setting: `{args.setting}`",
        f"- eval_mode: `{args.eval_mode}`",
        f"- metric_stride: `{args.metric_stride}`",
        f"- max_metric_points: `{args.max_metric_points}`",
        f"- normal_metrics: `{bool(args.normal_metrics)}`",
        f"- depth_metrics: `{bool(args.depth_metrics)}`",
        f"- rows: {summary['num_rows']}, ok: {summary['num_ok']}, missing: {summary['num_missing_points']}, failed: {summary['num_failed']}",
        "",
        "| model | variant | n ok / total | acc_mean_norm ↓ | comp_mean_norm ↓ | NC ↑ | depth AbsRel ↓ | depth RMSE norm ↓ | depth delta1 ↑ | valid_pixel_ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregate"]:
        lines.append(
            "| {model} | {variant} | {n_ok}/{n_total} | {acc_mean_norm} | {comp_mean_norm} | {nc_mean} | {depth_abs_rel} | {depth_rmse_norm} | {depth_delta1} | {valid} |".format(
                model=row["model"],
                variant=row["variant"],
                n_ok=row["n_ok"],
                n_total=row["n_total"],
                acc_mean_norm=format_float(row["acc_mean_norm"]),
                comp_mean_norm=format_float(row["comp_mean_norm"]),
                nc_mean=format_float(row["nc_mean"]),
                depth_abs_rel=format_float(row["depth_abs_rel"]),
                depth_rmse_norm=format_float(row["depth_rmse_norm"]),
                depth_delta1=format_float(row["depth_delta1"]),
                valid=format_float(row["valid_pixel_ratio"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--setting", choices=["scannetpp_weak", "waymo_weak", "waymo_plausible"], required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--summary-name", default="gt_point_accuracy")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--eval-mode", choices=["all", "prefix"], default="all")
    parser.add_argument("--default-prefix-size", type=int, default=5)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--metric-stride", type=int, default=16)
    parser.add_argument("--max-metric-points", type=int, default=120000)
    parser.add_argument("--normal-metrics", action="store_true")
    parser.add_argument("--depth-metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    rows, gt_sources = compute_rows(args)
    summary = summarize_rows(rows)
    write_csv(result_root / f"{args.summary_name}_rows.csv", rows)
    write_csv(result_root / f"{args.summary_name}_gt_sources.csv", gt_sources)
    write_json(result_root / f"{args.summary_name}_rows.json", rows)
    write_json(result_root / f"{args.summary_name}_gt_sources.json", gt_sources)
    write_json(result_root / f"{args.summary_name}_summary.json", summary)
    write_summary_md(result_root / f"{args.summary_name}_summary.md", args=args, summary=summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["num_ok"] == 0:
        raise SystemExit("No successful rows; inspect row statuses.")


if __name__ == "__main__":
    main()
