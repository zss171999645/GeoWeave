#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import shutil
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from PIL import Image


class AttentionNoisePair:
    def __init__(self, ref_id: int, clean_ids: List[int], noise_ids: List[int]) -> None:
        self.ref_id = int(ref_id)
        self.clean_ids = [int(item) for item in clean_ids]
        self.noise_ids = [int(item) for item in noise_ids]


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_DEPTH_EXTS = {".png", ".exr", ".hdr", ".npy", ".pfm"}


def load_read_camera():
    from easyvolcap.utils.easy_utils import read_camera

    return read_camera


def parse_pose_txt(pose_path: Path) -> np.ndarray:
    rows: List[np.ndarray] = []
    for line in pose_path.read_text(encoding="utf-8").strip().splitlines():
        values = [float(item) for item in line.split()]
        if len(values) != 16:
            raise ValueError(f"Expected 16 values per pose row: {pose_path}")
        rows.append(np.asarray(values, dtype=np.float64).reshape(4, 4))
    if not rows:
        raise ValueError(f"Empty pose file: {pose_path}")
    return np.stack(rows, axis=0)


def frame_id_from_path(path: Path) -> int:
    source_path = path.resolve() if path.is_symlink() else path
    digits = "".join(ch for ch in source_path.stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse frame id from {source_path.name}")
    return int(digits)


def load_png_depth(depth_path: Path, depth_scale: float = 1000.0) -> np.ndarray:
    depth_png = np.array(Image.open(depth_path), dtype=np.int32)
    if depth_png.ndim != 2:
        raise ValueError(f"Expected single-channel depth image: {depth_path}")
    depth = depth_png.astype(np.float32) / float(depth_scale)
    depth[depth_png <= 0] = 0.0
    return depth


def load_depth_map(depth_path: Path) -> np.ndarray:
    return load_depth_map_with_shape(depth_path=depth_path, raw_shape_hw=None)


def load_raw_float32_depth(depth_path: Path, raw_shape_hw: Sequence[int]) -> np.ndarray:
    if len(raw_shape_hw) != 2:
        raise ValueError(f"Expected raw depth shape (H, W), got {raw_shape_hw}")
    h, w = int(raw_shape_hw[0]), int(raw_shape_hw[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid raw depth shape for {depth_path}: {(h, w)}")
    depth = np.fromfile(str(depth_path), dtype=np.float32)
    expected = int(h) * int(w)
    if int(depth.size) != expected:
        raise ValueError(f"Raw depth size mismatch for {depth_path}: expected {expected}, got {depth.size}")
    depth = depth.reshape(h, w).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth <= 1e-4] = 0.0
    return depth


def load_depth_map_with_shape(depth_path: Path, raw_shape_hw: Sequence[int] | None = None) -> np.ndarray:
    suffix = depth_path.suffix.lower()
    if raw_shape_hw is not None:
        return load_raw_float32_depth(depth_path=depth_path, raw_shape_hw=raw_shape_hw)
    if suffix == ".png":
        return load_png_depth(depth_path)
    from easyvolcap.utils.data_utils import load_depth

    depth = np.asarray(load_depth(str(depth_path)), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth map with shape (H, W) or (H, W, 1): {depth_path}")
    depth[~np.isfinite(depth)] = 0.0
    depth[depth <= 1e-4] = 0.0
    return depth


def load_intrinsic_txt(intrinsic_path: Path) -> np.ndarray:
    matrix = np.loadtxt(intrinsic_path, dtype=np.float64)
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected 3x3 or 4x4 intrinsic matrix: {intrinsic_path}")
    return matrix.astype(np.float32)


def resolve_depth_intrinsic_path(seq_root: Path, depth_paths: List[Path]) -> Path | None:
    local_path = seq_root / "depth_intrinsics" / "depth_intrinsic.txt"
    if local_path.is_file():
        return local_path
    if not depth_paths:
        return None
    first_depth = depth_paths[0]
    if first_depth.is_symlink():
        raw_scene_root = first_depth.resolve().parents[1]
        raw_path = raw_scene_root / "depth_intrinsics" / "depth_intrinsic.txt"
        if raw_path.is_file():
            return raw_path
    return None


def build_exact_official_scene_record(seq_root: Path) -> Dict[str, Any]:
    color_dir = seq_root / "color_90"
    depth_dir = seq_root / "depth_90"
    pose_path = seq_root / "pose_90.txt"
    if not color_dir.is_dir() or not depth_dir.is_dir() or not pose_path.is_file():
        raise FileNotFoundError(f"Missing official exact layout under {seq_root}")

    color_paths = sorted(color_dir.glob("*.jpg"))
    depth_paths = sorted(depth_dir.glob("*.png"))
    poses = parse_pose_txt(pose_path)

    if not color_paths:
        raise RuntimeError(f"No color frames found under {color_dir}")
    if len(color_paths) != len(depth_paths) or len(color_paths) != int(poses.shape[0]):
        raise RuntimeError(f"Frame count mismatch under {seq_root}")

    color_ids = [frame_id_from_path(path) for path in color_paths]
    depth_ids = [frame_id_from_path(path) for path in depth_paths]
    if color_ids != depth_ids:
        raise RuntimeError(f"Color/depth frame id mismatch under {seq_root}")
    depth_intrinsic_path = resolve_depth_intrinsic_path(seq_root=seq_root, depth_paths=depth_paths)

    return {
        "scene_name": seq_root.name,
        "seq_root": seq_root,
        "frame_names": [f"{frame_id:04d}" for frame_id in color_ids],
        "frame_ids": color_ids,
        "color_paths": color_paths,
        "depth_paths": depth_paths,
        "poses": poses,
        "depth_intrinsic_path": depth_intrinsic_path,
        "depth_intrinsic": load_intrinsic_txt(depth_intrinsic_path) if depth_intrinsic_path is not None else None,
        "intrinsics": [load_intrinsic_txt(depth_intrinsic_path) for _ in color_ids] if depth_intrinsic_path is not None else None,
    }


def camera_rt_to_c2w(rt: np.ndarray, dataset_name: str = "") -> np.ndarray:
    rt = np.asarray(rt, dtype=np.float64)
    if rt.shape != (3, 4):
        raise ValueError(f"Expected camera RT with shape (3, 4), got {rt.shape}")
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :4] = rt
    if str(dataset_name).strip().lower() == "scannetv2":
        return c2w
    return np.linalg.inv(c2w)


def build_c2w_stack_from_cameras(
    cameras: Mapping[str, Any],
    ordered_names: Sequence[str],
    dataset_name: str = "",
) -> np.ndarray:
    poses = [
        camera_rt_to_c2w(np.asarray(cameras[name].RT, dtype=np.float64), dataset_name=dataset_name)
        for name in ordered_names
    ]
    if not poses:
        raise RuntimeError("No camera poses available for ordered sequence names")
    return np.stack(poses, axis=0).astype(np.float32)


def build_frame_file_map(frame_dir: Path, allowed_exts: Sequence[str]) -> Dict[str, Path]:
    allowed = {str(item).lower() for item in allowed_exts}
    mapping = {
        path.stem: path
        for path in sorted(frame_dir.iterdir())
        if path.is_file() and path.suffix.lower() in allowed
    }
    if not mapping:
        raise FileNotFoundError(f"No supported files found under {frame_dir}")
    return mapping


def resolve_first_supported_file(frame_dir: Path, allowed_exts: Sequence[str]) -> Path | None:
    allowed = {str(item).lower() for item in allowed_exts}
    for path in sorted(frame_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in allowed and not path.name.startswith("."):
            return path
    return None


def build_nested_frame_file_map(frame_root: Path, allowed_exts: Sequence[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for frame_dir in sorted(path for path in frame_root.iterdir() if path.is_dir()):
        frame_file = resolve_first_supported_file(frame_dir, allowed_exts)
        if frame_file is not None:
            mapping[frame_dir.name] = frame_file
    if not mapping:
        raise FileNotFoundError(f"No supported nested frame files found under {frame_root}")
    return mapping


def build_evc_camera_scene_record(
    seq_root: Path,
    dataset_name: str,
    image_rel_path: str = "images/00",
    depth_rel_path: str = "depths/00",
    camera_rel_path: str = "cameras/00",
    scene_name: str = "",
) -> Dict[str, Any]:
    seq_root = Path(seq_root).resolve()
    image_dir = seq_root / image_rel_path
    depth_dir = seq_root / depth_rel_path
    camera_dir = seq_root / camera_rel_path
    intri_path = camera_dir / "intri.yml"
    extri_path = camera_dir / "extri.yml"
    if not image_dir.is_dir() or not depth_dir.is_dir() or (not intri_path.is_file()) or (not extri_path.is_file()):
        raise FileNotFoundError(
            f"Missing EVC scene layout under {seq_root}: image_dir={image_dir}, depth_dir={depth_dir}, "
            f"intri={intri_path}, extri={extri_path}"
        )

    cameras = load_read_camera()(str(intri_path), str(extri_path), use_dict=False)
    image_map = build_frame_file_map(image_dir, sorted(SUPPORTED_IMAGE_EXTS))
    depth_map = build_frame_file_map(depth_dir, sorted(SUPPORTED_DEPTH_EXTS))
    ordered_names = [
        name
        for name in sorted(image_map.keys(), key=lambda item: int(item))
        if name in cameras and name in depth_map
    ]
    if not ordered_names:
        raise RuntimeError(f"No overlapping frame names among images/depths/cameras under {seq_root}")

    poses = build_c2w_stack_from_cameras(cameras=cameras, ordered_names=ordered_names, dataset_name=dataset_name)
    intrinsics = [np.asarray(cameras[name].K, dtype=np.float32) for name in ordered_names]
    return {
        "scene_name": str(scene_name or seq_root.name),
        "seq_root": seq_root,
        "frame_names": [str(name) for name in ordered_names],
        "frame_ids": [int(name) for name in ordered_names],
        "color_paths": [image_map[name] for name in ordered_names],
        "depth_paths": [depth_map[name] for name in ordered_names],
        "poses": poses,
        "intrinsics": intrinsics,
    }


def build_nested_evc_camera_scene_record(
    seq_root: Path,
    dataset_name: str,
    image_rel_path: str = "images",
    depth_rel_path: str = "depths",
    intri_rel_path: str = "intri.yml",
    extri_rel_path: str = "extri.yml",
    scene_name: str = "",
) -> Dict[str, Any]:
    seq_root = Path(seq_root).resolve()
    image_root = seq_root / image_rel_path
    depth_root = seq_root / depth_rel_path
    intri_path = seq_root / intri_rel_path
    extri_path = seq_root / extri_rel_path
    if (not image_root.is_dir()) or (not depth_root.is_dir()) or (not intri_path.is_file()) or (not extri_path.is_file()):
        raise FileNotFoundError(
            f"Missing nested EVC scene layout under {seq_root}: image_root={image_root}, "
            f"depth_root={depth_root}, intri={intri_path}, extri={extri_path}"
        )

    cameras = load_read_camera()(str(intri_path), str(extri_path), use_dict=False)
    image_map = build_nested_frame_file_map(image_root, sorted(SUPPORTED_IMAGE_EXTS))
    depth_map = build_nested_frame_file_map(depth_root, sorted(SUPPORTED_DEPTH_EXTS))
    ordered_names = [
        name
        for name in sorted(image_map.keys(), key=lambda item: int(item))
        if name in cameras and name in depth_map
    ]
    if not ordered_names:
        raise RuntimeError(f"No overlapping nested frame names among images/depths/cameras under {seq_root}")

    poses = build_c2w_stack_from_cameras(cameras=cameras, ordered_names=ordered_names, dataset_name=dataset_name)
    intrinsics = [np.asarray(cameras[name].K, dtype=np.float32) for name in ordered_names]
    return {
        "scene_name": str(scene_name or seq_root.name),
        "seq_root": seq_root,
        "frame_names": [str(name) for name in ordered_names],
        "frame_ids": [int(name) for name in ordered_names],
        "color_paths": [image_map[name] for name in ordered_names],
        "depth_paths": [depth_map[name] for name in ordered_names],
        "poses": poses,
        "intrinsics": intrinsics,
    }


def build_eth3d_pi3_scene_record(seq_root: Path, scene_name: str = "") -> Dict[str, Any]:
    seq_root = Path(seq_root).resolve()
    image_dir = seq_root / "images" / "custom_undistorted"
    depth_dir = seq_root / "ground_truth_depth" / "custom_undistorted"
    camera_dir = seq_root / "custom_undistorted_cam"
    if not image_dir.is_dir() or not depth_dir.is_dir() or not camera_dir.is_dir():
        raise FileNotFoundError(
            f"Missing ETH3D PI3-style scene layout under {seq_root}: "
            f"image_dir={image_dir}, depth_dir={depth_dir}, camera_dir={camera_dir}"
        )

    image_map = build_frame_file_map(image_dir, sorted(SUPPORTED_IMAGE_EXTS))
    depth_map = build_frame_file_map(depth_dir, sorted(SUPPORTED_IMAGE_EXTS))
    camera_map = {
        path.stem: path
        for path in sorted(camera_dir.iterdir())
        if path.is_file() and path.suffix.lower() == ".npz" and not path.name.startswith(".")
    }
    ordered_names = [
        name
        for name in sorted(image_map.keys())
        if name in depth_map and name in camera_map
    ]
    if not ordered_names:
        raise RuntimeError(f"No overlapping ETH3D image/depth/camera names under {seq_root}")

    poses: List[np.ndarray] = []
    intrinsics: List[np.ndarray] = []
    depth_shapes: Dict[str, tuple[int, int]] = {}
    for name in ordered_names:
        with Image.open(image_map[name]) as image:
            width, height = image.size
        cam = np.load(str(camera_map[name]))
        intrinsic = np.asarray(cam["intrinsics"], dtype=np.float32)
        w2c = np.asarray(cam["extrinsics"], dtype=np.float64)
        if w2c.shape != (4, 4):
            raise ValueError(f"Expected ETH3D extrinsics shape (4, 4), got {w2c.shape}: {camera_map[name]}")
        poses.append(np.linalg.inv(w2c).astype(np.float32))
        intrinsics.append(intrinsic)
        depth_shapes[str(depth_map[name].resolve())] = (int(height), int(width))

    return {
        "scene_name": str(scene_name or seq_root.name),
        "seq_root": seq_root,
        "frame_names": [str(name) for name in ordered_names],
        "frame_ids": [frame_id_from_path(image_map[name]) for name in ordered_names],
        "color_paths": [image_map[name] for name in ordered_names],
        "depth_paths": [depth_map[name] for name in ordered_names],
        "poses": np.stack(poses, axis=0).astype(np.float32),
        "intrinsics": intrinsics,
        "depth_shapes": depth_shapes,
    }


def subsample_scene_record(
    scene_record: Mapping[str, Any],
    target_num_frames: int,
    source_prestride: int = 3,
) -> Dict[str, Any]:
    frame_ids = list(scene_record["frame_ids"])
    if target_num_frames <= 0 or len(frame_ids) <= target_num_frames:
        return dict(scene_record)
    if source_prestride <= 0:
        raise ValueError(f"source_prestride must be positive, got {source_prestride}")

    source_limit = min(len(frame_ids), int(target_num_frames) * int(source_prestride))
    selected_indices = list(range(0, source_limit, int(source_prestride)))[: int(target_num_frames)]
    if not selected_indices:
        return dict(scene_record)

    payload = dict(scene_record)
    payload["frame_ids"] = [int(frame_ids[index]) for index in selected_indices]
    if "frame_names" in scene_record:
        payload["frame_names"] = [str(scene_record["frame_names"][index]) for index in selected_indices]
    payload["color_paths"] = [scene_record["color_paths"][index] for index in selected_indices]
    payload["depth_paths"] = [scene_record["depth_paths"][index] for index in selected_indices]
    payload["poses"] = np.asarray(scene_record["poses"])[selected_indices]
    if "intrinsics" in scene_record and scene_record["intrinsics"] is not None:
        payload["intrinsics"] = [scene_record["intrinsics"][index] for index in selected_indices]
    return payload


def select_attention_noise_pair(
    ref_id: int,
    overlap_by_ref: Mapping[int, float],
    clean_overlap_threshold: float,
    noise_overlap_threshold: float,
    clean_temporal_dedup: int,
) -> AttentionNoisePair:
    clean_candidates = [
        (int(candidate_id), float(overlap))
        for candidate_id, overlap in overlap_by_ref.items()
        if int(candidate_id) != int(ref_id) and float(overlap) >= float(clean_overlap_threshold)
    ]
    clean_candidates.sort(key=lambda item: (-item[1], item[0]))

    clean_support_ids: List[int] = []
    for candidate_id, _ in clean_candidates:
        if any(abs(candidate_id - kept_id) < int(clean_temporal_dedup) for kept_id in clean_support_ids):
            continue
        clean_support_ids.append(candidate_id)
        if len(clean_support_ids) == 9:
            break

    noise_candidates = sorted(
        int(candidate_id)
        for candidate_id, overlap in overlap_by_ref.items()
        if int(candidate_id) != int(ref_id) and float(overlap) <= float(noise_overlap_threshold)
    )
    if len(clean_support_ids) < 9:
        raise ValueError(f"Not enough clean candidates for ref={ref_id}")
    if len(noise_candidates) < 4:
        raise ValueError(f"Not enough noise candidates for ref={ref_id}")

    return AttentionNoisePair(
        ref_id=int(ref_id),
        clean_ids=[int(ref_id), *clean_support_ids],
        noise_ids=[int(ref_id), *clean_support_ids[:5], *noise_candidates[:4]],
    )


def build_ref_world_points(depth: np.ndarray, intrinsic: np.ndarray, c2w: np.ndarray, sample_stride: int) -> np.ndarray:
    valid_mask = depth > 1e-4
    ys, xs = np.nonzero(valid_mask[::sample_stride, ::sample_stride])
    if len(xs) == 0:
        ys, xs = np.nonzero(valid_mask)
        if len(xs) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        step = 1
    else:
        step = int(sample_stride)
    ys = np.clip(ys * step, 0, depth.shape[0] - 1)
    xs = np.clip(xs * step, 0, depth.shape[1] - 1)
    z = depth[ys, xs]
    pixels = np.stack(
        [xs.astype(np.float32), ys.astype(np.float32), np.ones_like(z, dtype=np.float32)],
        axis=0,
    )
    cam_points = (np.linalg.inv(intrinsic).astype(np.float32) @ pixels) * z[None, :]
    world_points = (c2w[:3, :3].astype(np.float32) @ cam_points) + c2w[:3, 3:4].astype(np.float32)
    return world_points.T.astype(np.float32)


def compute_directional_overlap(
    ref_world_points: np.ndarray,
    candidate_depth: np.ndarray,
    candidate_intrinsic: np.ndarray,
    candidate_c2w: np.ndarray,
    depth_rel_tol: float,
) -> float:
    if ref_world_points.shape[0] == 0:
        return 0.0
    w2c = np.linalg.inv(candidate_c2w).astype(np.float32)
    cam_points = (w2c[:3, :3] @ ref_world_points.T) + w2c[:3, 3:4]
    z = cam_points[2]
    in_front = z > 1e-4
    if not np.any(in_front):
        return 0.0

    proj = candidate_intrinsic.astype(np.float32) @ cam_points
    u = proj[0] / np.maximum(proj[2], 1e-8)
    v = proj[1] / np.maximum(proj[2], 1e-8)
    h, w = candidate_depth.shape
    inside = in_front & (u >= 0.0) & (u <= float(w - 1)) & (v >= 0.0) & (v <= float(h - 1))
    if not np.any(inside):
        return 0.0

    u_idx = np.rint(u[inside]).astype(np.int32)
    v_idx = np.rint(v[inside]).astype(np.int32)
    depth_src = candidate_depth[v_idx, u_idx]
    projected_depth = z[inside]
    depth_ok = np.abs(depth_src - projected_depth) <= (float(depth_rel_tol) * np.maximum(depth_src, 1e-4))
    visible = (depth_src > 1e-4) & depth_ok
    return float(visible.sum() / max(ref_world_points.shape[0], 1))


def build_scene_overlap_table(
    scene_record: Mapping[str, Any],
    sample_stride: int,
    depth_rel_tol: float,
) -> Dict[int, Dict[int, float]]:
    frame_ids = [int(item) for item in scene_record["frame_ids"]]
    poses = np.asarray(scene_record["poses"], dtype=np.float32)
    intrinsics = scene_record.get("intrinsics", None)
    if intrinsics is None:
        intrinsic = scene_record.get("depth_intrinsic", None)
        if intrinsic is None:
            raise FileNotFoundError(f"Missing intrinsic information for scene {scene_record.get('scene_name', '<unknown>')}")
        intrinsics = [np.asarray(intrinsic, dtype=np.float32) for _ in frame_ids]
    else:
        intrinsics = [np.asarray(item, dtype=np.float32) for item in intrinsics]
    depth_shapes = {
        str(Path(path).resolve()): tuple(shape)
        for path, shape in dict(scene_record.get("depth_shapes", {})).items()
    }
    depths = [
        load_depth_map_with_shape(
            Path(path),
            raw_shape_hw=depth_shapes.get(str(Path(path).resolve())),
        )
        for path in scene_record["depth_paths"]
    ]

    overlap_table: Dict[int, Dict[int, float]] = {}
    for ref_idx, ref_id in enumerate(frame_ids):
        ref_world_points = build_ref_world_points(
            depth=depths[ref_idx],
            intrinsic=intrinsics[ref_idx],
            c2w=poses[ref_idx],
            sample_stride=sample_stride,
        )
        candidate_scores: Dict[int, float] = {}
        for candidate_idx, candidate_id in enumerate(frame_ids):
            if candidate_id == ref_id:
                continue
            candidate_scores[candidate_id] = compute_directional_overlap(
                ref_world_points=ref_world_points,
                candidate_depth=depths[candidate_idx],
                candidate_intrinsic=intrinsics[candidate_idx],
                candidate_c2w=poses[candidate_idx],
                depth_rel_tol=depth_rel_tol,
            )
        overlap_table[ref_id] = candidate_scores
    return overlap_table


def materialize_tuple_sequence(
    scene_record: Mapping[str, Any],
    tuple_name: str,
    ordered_ids: List[int],
    output_root: Path,
) -> Path:
    tuple_root = output_root / tuple_name
    color_dir = tuple_root / "color_90"
    depth_dir = tuple_root / "depth_90"
    color_dir.mkdir(parents=True, exist_ok=False)
    depth_dir.mkdir(parents=True, exist_ok=False)

    frame_ids = [int(item) for item in scene_record["frame_ids"]]
    id_to_index = {frame_id: idx for idx, frame_id in enumerate(frame_ids)}
    poses = np.asarray(scene_record["poses"], dtype=np.float64)
    pose_lines: List[str] = []

    for out_idx, frame_id in enumerate(ordered_ids):
        src_idx = id_to_index[int(frame_id)]
        color_suffix = Path(scene_record["color_paths"][src_idx]).suffix.lower() or ".jpg"
        src_depth_path = Path(scene_record["depth_paths"][src_idx])
        depth_shapes = {
            str(Path(path).resolve()): tuple(shape)
            for path, shape in dict(scene_record.get("depth_shapes", {})).items()
        }
        raw_shape = depth_shapes.get(str(src_depth_path.resolve()))
        depth_suffix = ".npy" if raw_shape is not None else (src_depth_path.suffix.lower() or ".png")
        shutil.copy2(scene_record["color_paths"][src_idx], color_dir / f"frame_{out_idx:04d}{color_suffix}")
        if raw_shape is not None:
            depth = load_raw_float32_depth(src_depth_path, raw_shape)
            np.save(depth_dir / f"frame_{out_idx:04d}{depth_suffix}", depth)
        else:
            shutil.copy2(src_depth_path, depth_dir / f"frame_{out_idx:04d}{depth_suffix}")
        pose = poses[src_idx].reshape(-1)
        pose_lines.append(" ".join(f"{float(value):.6f}" for value in pose))

    (tuple_root / "pose_90.txt").write_text("\n".join(pose_lines) + "\n", encoding="utf-8")
    return tuple_root


def select_anchor_ref_ids(
    eligible_ref_ids: List[int],
    anchor_quantiles: List[float],
    max_anchors: int,
) -> List[int]:
    if not eligible_ref_ids or max_anchors <= 0:
        return []
    ordered_ids = [int(item) for item in sorted(eligible_ref_ids)]
    selected: List[int] = []
    used_indices: set[int] = set()
    target_quantiles = list(anchor_quantiles[:max_anchors])
    for quantile in target_quantiles:
        base_index = int(np.floor(float(quantile) * len(ordered_ids)))
        base_index = min(max(base_index, 0), len(ordered_ids) - 1)
        candidate_indices = [base_index]
        for radius in range(1, len(ordered_ids)):
            right = base_index + radius
            left = base_index - radius
            if right < len(ordered_ids):
                candidate_indices.append(right)
            if left >= 0:
                candidate_indices.append(left)
        pick_index = next((index for index in candidate_indices if index not in used_indices), None)
        if pick_index is None:
            break
        used_indices.add(int(pick_index))
        selected.append(int(ordered_ids[pick_index]))
    return selected


def write_attention_noise_summary(
    output_root: Path,
    config: Mapping[str, Any],
    per_scene_rows: List[Mapping[str, Any]],
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": dict(config),
        "num_scenes_scanned": int(len(per_scene_rows)),
        "num_scenes_kept": int(sum(1 for row in per_scene_rows if int(row.get("num_clean_tuples", 0)) > 0)),
        "num_clean_tuples": int(sum(int(row.get("num_clean_tuples", 0)) for row in per_scene_rows)),
        "num_noise_tuples": int(sum(int(row.get("num_noise_tuples", 0)) for row in per_scene_rows)),
        "scenes": [dict(row) for row in per_scene_rows],
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_path
