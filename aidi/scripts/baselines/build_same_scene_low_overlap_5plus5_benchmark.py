#!/usr/bin/env python3
"""Build same-scene low-overlap 5+5 tuples for Pi3 pose evaluation.

The intended protocol is:
  - 10 input views total.
  - Views 0..4 form one local, mutually-overlapping cluster.
  - Views 5..9 form another local cluster from the same scene.
  - The two clusters are selected to have low GT-depth visibility overlap.
  - All 10 views have GT poses and should be evaluated together.

The output layout is Pi3 replica-style:
  <output>/<dataset>/<tuple>/color_90/frame_*.jpg
  <output>/<dataset>/<tuple>/pose_90.txt
  <output>/<dataset>/<tuple>/tuple_meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_repo_root_on_syspath() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


ensure_repo_root_on_syspath()

from aidi.scripts.baselines.build_evc_same_scene_candidate_pool_benchmark import (  # noqa: E402
    generic_scene_slug,
    parse_scene_specs,
    resolve_scene_roots,
)
from aidi.scripts.baselines.build_fixed10_overlap_band_benchmark import (  # noqa: E402
    load_scene_record,
    write_tuple_meta,
)
from aidi.scripts.baselines.overlap_noise_seq_map_utils import (  # noqa: E402
    SUPPORTED_DEPTH_EXTS,
    SUPPORTED_IMAGE_EXTS,
    build_c2w_stack_from_cameras,
    build_frame_file_map,
    build_nested_frame_file_map,
    build_scene_overlap_table,
    camera_rt_to_c2w,
    frame_id_from_path,
    load_read_camera,
    materialize_tuple_sequence,
)


DEFAULT_ETH3D_ROOT = (
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/"
    "eval_datasets/eth3d_pi3_style_root"
)
SUPPORTED_LAYOUTS = ("eth3d_pi3", "evc", "nested_evc", "exact_official")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DEFAULT_ETH3D_ROOT)
    parser.add_argument("--dataset-name", default="eth3d")
    parser.add_argument("--source-layout", default="eth3d_pi3", choices=SUPPORTED_LAYOUTS)
    parser.add_argument("--scene-specs", default="", help="Comma-separated scene paths/names relative to dataset root.")
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--samples-per-scene", type=int, default=3)
    parser.add_argument("--anchor-quantiles", default="0.10,0.30,0.50,0.70,0.90")
    parser.add_argument("--views-per-cluster", type=int, default=5)
    parser.add_argument("--overlap-sample-stride", type=int, default=48)
    parser.add_argument("--depth-rel-tol", type=float, default=0.05)
    parser.add_argument("--local-overlap-threshold", type=float, default=0.08)
    parser.add_argument("--anchor-low-overlap-min", type=float, default=0.0)
    parser.add_argument("--anchor-low-overlap-threshold", type=float, default=0.08)
    parser.add_argument("--cross-mean-overlap-min", type=float, default=0.0)
    parser.add_argument("--cross-mean-overlap-threshold", type=float, default=0.10)
    parser.add_argument("--cross-max-overlap-min", type=float, default=0.0)
    parser.add_argument("--cross-max-overlap-threshold", type=float, default=0.35)
    parser.add_argument("--min-center-distance-ratio", type=float, default=0.20)
    parser.add_argument("--image-subdir", default="images")
    parser.add_argument("--depth-subdir", default="depths")
    parser.add_argument("--camera-subdir", default="cameras")
    parser.add_argument("--write-preview", action="store_true")
    return parser.parse_args()


def default_output_root(dataset_name: str) -> Path:
    return repo_root() / "tmp" / f"{dataset_name}_same_scene_low_overlap_5plus5_{time.strftime('%Y%m%d_%H%M%S')}"


def parse_float_list(raw: str) -> List[float]:
    values = [float(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError(f"Expected non-empty comma-separated float list, got {raw!r}")
    return values


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def build_flexible_evc_scene_record(
    scene_root: Path,
    dataset_name: str,
    image_subdir: str,
    depth_subdir: str,
    scene_name: str,
) -> Dict[str, Any]:
    scene_root = Path(scene_root).expanduser().resolve()
    image_root = scene_root / image_subdir
    depth_root = scene_root / depth_subdir
    intri_path = scene_root / "intri.yml"
    extri_path = scene_root / "extri.yml"
    if not image_root.is_dir() or not depth_root.is_dir() or not intri_path.is_file() or not extri_path.is_file():
        raise FileNotFoundError(
            f"Missing flexible EVC scene layout: images={image_root}, depths={depth_root}, "
            f"intri={intri_path}, extri={extri_path}"
        )

    try:
        cameras = load_read_camera()(str(intri_path), str(extri_path), use_dict=False)
        camera_poses = None
        camera_intrinsics = None
    except ValueError as exc:
        if "invalid literal for int()" not in str(exc):
            raise
        camera_intrinsics, camera_poses = load_opencv_camera_yaml_flexible(
            intri_path=intri_path,
            extri_path=extri_path,
            dataset_name=dataset_name,
        )
        cameras = camera_intrinsics
    image_map = (
        build_frame_file_map(image_root, sorted(SUPPORTED_IMAGE_EXTS))
        if any(path.is_file() for path in image_root.iterdir())
        else build_nested_frame_file_map(image_root, sorted(SUPPORTED_IMAGE_EXTS))
    )
    depth_map = (
        build_frame_file_map(depth_root, sorted(SUPPORTED_DEPTH_EXTS))
        if any(path.is_file() for path in depth_root.iterdir())
        else build_nested_frame_file_map(depth_root, sorted(SUPPORTED_DEPTH_EXTS))
    )
    ordered_names = [
        name
        for name in sorted(image_map.keys(), key=lambda item: (frame_id_from_path(image_map[item]), item))
        if name in cameras and name in depth_map
    ]
    if not ordered_names:
        raise RuntimeError(f"No overlapping flexible EVC frame names under {scene_root}")
    frame_ids_out = [frame_id_from_path(image_map[name]) for name in ordered_names]
    if len(set(frame_ids_out)) != len(frame_ids_out):
        raise ValueError(f"Duplicate numeric frame ids after parsing names under {scene_root}: {frame_ids_out}")

    return {
        "scene_name": str(scene_name or scene_root.name),
        "seq_root": scene_root,
        "frame_names": [str(name) for name in ordered_names],
        "frame_ids": [int(item) for item in frame_ids_out],
        "color_paths": [image_map[name] for name in ordered_names],
        "depth_paths": [depth_map[name] for name in ordered_names],
        "poses": (
            np.stack([camera_poses[name] for name in ordered_names], axis=0).astype(np.float32)
            if camera_poses is not None
            else build_c2w_stack_from_cameras(cameras=cameras, ordered_names=ordered_names, dataset_name=dataset_name)
        ),
        "intrinsics": (
            [np.asarray(camera_intrinsics[name], dtype=np.float32) for name in ordered_names]
            if camera_intrinsics is not None
            else [np.asarray(cameras[name].K, dtype=np.float32) for name in ordered_names]
        ),
    }


def load_opencv_string_list(fs: Any, name: str) -> List[str]:
    node = fs.getNode(name)
    if node.empty():
        raise KeyError(f"Missing OpenCV YAML node: {name}")
    return [str(node.at(i).string()) for i in range(int(node.size()))]


def load_opencv_camera_yaml_flexible(
    intri_path: Path,
    extri_path: Path,
    dataset_name: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    import cv2

    fs_i = cv2.FileStorage(str(intri_path), cv2.FILE_STORAGE_READ)
    fs_e = cv2.FileStorage(str(extri_path), cv2.FILE_STORAGE_READ)
    if not fs_i.isOpened() or not fs_e.isOpened():
        raise FileNotFoundError(f"Failed opening OpenCV camera YAML: {intri_path}, {extri_path}")
    try:
        names_i = load_opencv_string_list(fs_i, "names")
        names_e = load_opencv_string_list(fs_e, "names")
        if names_i != names_e:
            raise ValueError("intri/extri camera names differ")
        intrinsics: Dict[str, np.ndarray] = {}
        poses: Dict[str, np.ndarray] = {}
        for name in names_i:
            k = fs_i.getNode(f"K_{name}").mat()
            rot = fs_e.getNode(f"Rot_{name}").mat()
            trans = fs_e.getNode(f"T_{name}").mat()
            if k is None or rot is None or trans is None:
                raise KeyError(f"Missing K/Rot/T nodes for camera {name}")
            rt = np.concatenate([np.asarray(rot, dtype=np.float64), np.asarray(trans, dtype=np.float64).reshape(3, 1)], axis=1)
            intrinsics[str(name)] = np.asarray(k, dtype=np.float32)
            poses[str(name)] = camera_rt_to_c2w(rt, dataset_name=dataset_name).astype(np.float32)
        return intrinsics, poses
    finally:
        fs_i.release()
        fs_e.release()


def load_scene_record_for_protocol(
    scene_root: Path,
    dataset_root: Path,
    args: argparse.Namespace,
    scene_name: str,
) -> Dict[str, Any]:
    try:
        return load_scene_record(
            scene_root=scene_root,
            dataset_root=dataset_root,
            dataset_name=str(args.dataset_name),
            source_layout=str(args.source_layout),
            image_subdir=str(args.image_subdir),
            depth_subdir=str(args.depth_subdir),
            camera_subdir=str(args.camera_subdir),
        )
    except ValueError as exc:
        if str(args.source_layout) != "nested_evc" or "invalid literal for int()" not in str(exc):
            raise
        return build_flexible_evc_scene_record(
            scene_root=scene_root,
            dataset_name=str(args.dataset_name),
            image_subdir=str(args.image_subdir),
            depth_subdir=str(args.depth_subdir),
            scene_name=scene_name,
        )


def frame_ids(scene_record: Mapping[str, Any]) -> List[int]:
    return [int(item) for item in scene_record["frame_ids"]]


def center_array(scene_record: Mapping[str, Any]) -> np.ndarray:
    poses = np.asarray(scene_record["poses"], dtype=np.float64)
    return poses[:, :3, 3]


def scene_diameter(centers: np.ndarray) -> float:
    if centers.shape[0] < 2:
        return 1.0
    span = centers.max(axis=0) - centers.min(axis=0)
    return float(np.linalg.norm(span)) or 1.0


def bidirectional_overlap(overlap_table: Mapping[int, Mapping[int, float]], lhs: int, rhs: int) -> float:
    lhs = int(lhs)
    rhs = int(rhs)
    return float(
        0.5
        * (
            float(overlap_table.get(lhs, {}).get(rhs, 0.0))
            + float(overlap_table.get(rhs, {}).get(lhs, 0.0))
        )
    )


def choose_local_cluster(
    anchor_id: int,
    overlap_table: Mapping[int, Mapping[int, float]],
    all_ids: Sequence[int],
    count: int,
    local_overlap_threshold: float,
    excluded_ids: set[int] | None = None,
) -> List[int] | None:
    excluded = set(excluded_ids or set())
    anchor_id = int(anchor_id)
    if anchor_id in excluded:
        return None
    rows: List[Tuple[int, float, int]] = []
    for candidate_id in all_ids:
        candidate_id = int(candidate_id)
        if candidate_id == anchor_id or candidate_id in excluded:
            continue
        overlap = bidirectional_overlap(overlap_table, anchor_id, candidate_id)
        if overlap < float(local_overlap_threshold):
            continue
        rows.append((candidate_id, overlap, abs(candidate_id - anchor_id)))
    rows.sort(key=lambda item: (-item[1], item[2], item[0]))
    need = int(count) - 1
    if len(rows) < need:
        return None
    return [anchor_id, *[candidate_id for candidate_id, _, _ in rows[:need]]]


def cross_group_stats(
    group_a: Sequence[int],
    group_b: Sequence[int],
    overlap_table: Mapping[int, Mapping[int, float]],
) -> Dict[str, float]:
    values = [
        bidirectional_overlap(overlap_table, int(a), int(b))
        for a in group_a
        for b in group_b
    ]
    arr = np.asarray(values, dtype=np.float64)
    return {
        "cross_overlap_mean": float(arr.mean()) if arr.size else 0.0,
        "cross_overlap_max": float(arr.max()) if arr.size else 0.0,
        "cross_overlap_min": float(arr.min()) if arr.size else 0.0,
    }


def internal_overlap_mean(group: Sequence[int], overlap_table: Mapping[int, Mapping[int, float]]) -> float:
    values: List[float] = []
    for i, lhs in enumerate(group):
        for rhs in group[i + 1 :]:
            values.append(bidirectional_overlap(overlap_table, int(lhs), int(rhs)))
    return float(np.mean(values)) if values else 0.0


def anchor_ids_by_quantile(ids: Sequence[int], quantiles: Sequence[float], max_count: int) -> List[int]:
    ordered = list(sorted(int(item) for item in ids))
    if not ordered:
        return []
    picked: List[int] = []
    used: set[int] = set()
    for quantile in list(quantiles)[: int(max_count)]:
        center = min(max(int(round(float(quantile) * (len(ordered) - 1))), 0), len(ordered) - 1)
        candidates = [center]
        for delta in range(1, len(ordered)):
            if center + delta < len(ordered):
                candidates.append(center + delta)
            if center - delta >= 0:
                candidates.append(center - delta)
        index = next((item for item in candidates if item not in used), None)
        if index is None:
            break
        used.add(index)
        picked.append(ordered[index])
    return picked


def find_low_overlap_pairs(
    scene_record: Mapping[str, Any],
    overlap_table: Mapping[int, Mapping[int, float]],
    args: argparse.Namespace,
    quantiles: Sequence[float],
) -> List[Dict[str, Any]]:
    ids = frame_ids(scene_record)
    centers = center_array(scene_record)
    id_to_idx = {frame_id: idx for idx, frame_id in enumerate(ids)}
    diameter = scene_diameter(centers)
    anchors_a = anchor_ids_by_quantile(ids, quantiles, max_count=max(int(args.samples_per_scene) * 2, int(args.samples_per_scene)))
    rows: List[Dict[str, Any]] = []
    used_ordered_sets: set[Tuple[int, ...]] = set()

    for anchor_a in anchors_a:
        cluster_a = choose_local_cluster(
            anchor_id=anchor_a,
            overlap_table=overlap_table,
            all_ids=ids,
            count=int(args.views_per_cluster),
            local_overlap_threshold=float(args.local_overlap_threshold),
        )
        if cluster_a is None:
            continue
        candidates: List[Dict[str, Any]] = []
        for anchor_b in ids:
            if int(anchor_b) in set(cluster_a):
                continue
            anchor_overlap = bidirectional_overlap(overlap_table, anchor_a, anchor_b)
            if anchor_overlap < float(args.anchor_low_overlap_min):
                continue
            if anchor_overlap > float(args.anchor_low_overlap_threshold):
                continue
            cluster_b = choose_local_cluster(
                anchor_id=anchor_b,
                overlap_table=overlap_table,
                all_ids=ids,
                count=int(args.views_per_cluster),
                local_overlap_threshold=float(args.local_overlap_threshold),
                excluded_ids=set(cluster_a),
            )
            if cluster_b is None:
                continue
            stats = cross_group_stats(cluster_a, cluster_b, overlap_table)
            if stats["cross_overlap_mean"] < float(args.cross_mean_overlap_min):
                continue
            if stats["cross_overlap_mean"] > float(args.cross_mean_overlap_threshold):
                continue
            if stats["cross_overlap_max"] < float(args.cross_max_overlap_min):
                continue
            if stats["cross_overlap_max"] > float(args.cross_max_overlap_threshold):
                continue
            dist = float(np.linalg.norm(centers[id_to_idx[int(anchor_a)]] - centers[id_to_idx[int(anchor_b)]]))
            dist_ratio = float(dist / diameter)
            if dist_ratio < float(args.min_center_distance_ratio):
                continue
            ordered = [*cluster_a, *cluster_b]
            if tuple(ordered) in used_ordered_sets:
                continue
            candidates.append(
                {
                    "anchor_a": int(anchor_a),
                    "anchor_b": int(anchor_b),
                    "group_a_ids": [int(item) for item in cluster_a],
                    "group_b_ids": [int(item) for item in cluster_b],
                    "ordered_ids": [int(item) for item in ordered],
                    "anchor_pair_overlap": float(anchor_overlap),
                    "center_distance": float(dist),
                    "center_distance_ratio": float(dist_ratio),
                    "group_a_internal_overlap_mean": internal_overlap_mean(cluster_a, overlap_table),
                    "group_b_internal_overlap_mean": internal_overlap_mean(cluster_b, overlap_table),
                    **stats,
                }
            )
        candidates.sort(
            key=lambda item: (
                float(item["cross_overlap_mean"]),
                float(item["anchor_pair_overlap"]),
                -float(item["center_distance_ratio"]),
                -float(item["group_a_internal_overlap_mean"] + item["group_b_internal_overlap_mean"]),
            )
        )
        if candidates:
            picked = candidates[0]
            used_ordered_sets.add(tuple(int(item) for item in picked["ordered_ids"]))
            rows.append(picked)
        if len(rows) >= int(args.samples_per_scene):
            break
    return rows


def make_frame_rows(selection: Mapping[str, Any], overlap_table: Mapping[int, Mapping[int, float]]) -> List[Dict[str, Any]]:
    group_a = [int(item) for item in selection["group_a_ids"]]
    group_b = [int(item) for item in selection["group_b_ids"]]
    rows: List[Dict[str, Any]] = []
    for out_index, frame_id in enumerate([*group_a, *group_b]):
        group = "A" if out_index < len(group_a) else "B"
        anchor = int(selection["anchor_a"] if group == "A" else selection["anchor_b"])
        other_anchor = int(selection["anchor_b"] if group == "A" else selection["anchor_a"])
        rows.append(
            {
                "out_index": int(out_index),
                "frame_id": int(frame_id),
                "group": group,
                "role": "anchor" if frame_id == anchor else "support",
                "overlap_to_group_anchor": float(bidirectional_overlap(overlap_table, frame_id, anchor)) if frame_id != anchor else 1.0,
                "overlap_to_other_group_anchor": float(bidirectional_overlap(overlap_table, frame_id, other_anchor)),
            }
        )
    return rows


def write_preview_grid(tuple_root: Path, tuple_meta: Mapping[str, Any], preview_root: Path, thumb_width: int = 260) -> Path:
    images = sorted(
        path
        for path in (tuple_root / "color_90").iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    frame_rows = list(tuple_meta.get("frames", []))
    label_h = 42
    cells: List[Image.Image] = []
    font = ImageFont.load_default()
    for index, path in enumerate(images):
        image = Image.open(path).convert("RGB")
        scale = float(thumb_width) / max(float(image.width), 1.0)
        thumb_h = max(1, int(round(float(image.height) * scale)))
        image = image.resize((thumb_width, thumb_h), Image.Resampling.LANCZOS)
        row = frame_rows[index] if index < len(frame_rows) else {}
        group = str(row.get("group", "?"))
        color = (0, 160, 180) if group == "A" else (211, 127, 24)
        cell = Image.new("RGB", (thumb_width, thumb_h + label_h), "white")
        cell.paste(image, (0, label_h))
        draw = ImageDraw.Draw(cell)
        draw.rectangle((0, 0, thumb_width - 1, label_h - 1), outline=color, width=4)
        label = (
            f"v{index} group {group} id={row.get('frame_id', '')} "
            f"crossO={float(row.get('overlap_to_other_group_anchor', 0.0)):.3f}"
        )
        draw.text((8, 12), label[:72], fill=(20, 20, 20), font=font)
        cells.append(cell)
    if not cells:
        raise FileNotFoundError(f"No images under {tuple_root / 'color_90'}")
    columns = 5
    rows = int(math.ceil(len(cells) / columns))
    cell_w = max(cell.width for cell in cells)
    cell_h = max(cell.height for cell in cells)
    grid = Image.new("RGB", (columns * cell_w, rows * cell_h), (245, 241, 231))
    for index, cell in enumerate(cells):
        grid.paste(cell, ((index % columns) * cell_w, (index // columns) * cell_h))
    preview_root.mkdir(parents=True, exist_ok=True)
    preview_path = preview_root / f"{tuple_root.name}.jpg"
    grid.save(preview_path, quality=92)
    return preview_path


def build_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else default_output_root(args.dataset_name)
    dataset_output_root = output_root / str(args.dataset_name)
    dataset_output_root.mkdir(parents=True, exist_ok=True)
    scene_specs = parse_scene_specs(args.scene_specs)
    quantiles = parse_float_list(args.anchor_quantiles)
    scene_roots = resolve_scene_roots(
        dataset_root=dataset_root,
        scene_specs=scene_specs,
        limit_scenes=int(args.limit_scenes),
        dataset_name=str(args.dataset_name),
        source_layout=str(args.source_layout),
        image_subdir=str(args.image_subdir),
        depth_subdir=str(args.depth_subdir),
        camera_subdir=str(args.camera_subdir),
    )

    tuple_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for scene_root in scene_roots:
        scene_name = generic_scene_slug(scene_root, dataset_root)
        try:
            scene_record = load_scene_record_for_protocol(
                scene_root=scene_root,
                dataset_root=dataset_root,
                args=args,
                scene_name=scene_name,
            )
            if len(frame_ids(scene_record)) < int(args.views_per_cluster) * 2:
                raise ValueError(f"not enough frames: {len(frame_ids(scene_record))}")
            overlap_table = build_scene_overlap_table(
                scene_record=scene_record,
                sample_stride=int(args.overlap_sample_stride),
                depth_rel_tol=float(args.depth_rel_tol),
            )
            selections = find_low_overlap_pairs(
                scene_record=scene_record,
                overlap_table=overlap_table,
                args=args,
                quantiles=quantiles,
            )
            if not selections:
                skipped.append({"scene": scene_name, "reason": "no selection matched thresholds"})
                continue
            for sample_index, selection in enumerate(selections):
                tuple_name = (
                    f"{scene_name}__lo5x2_a{int(selection['anchor_a']):06d}"
                    f"_b{int(selection['anchor_b']):06d}_{sample_index:02d}"
                ).replace("/", "-")
                tuple_root = materialize_tuple_sequence(
                    scene_record=scene_record,
                    tuple_name=tuple_name,
                    ordered_ids=[int(item) for item in selection["ordered_ids"]],
                    output_root=dataset_output_root,
                )
                tuple_meta: Dict[str, Any] = {
                    "protocol": "same_scene_low_overlap_5plus5_v1",
                    "dataset_name": str(args.dataset_name),
                    "scene_name": scene_name,
                    "source_scene_root": str(scene_root),
                    "tuple_name": tuple_name,
                    "views_per_cluster": int(args.views_per_cluster),
                    "context_start": int(args.views_per_cluster),
                    "eval_frame_indices": list(range(int(args.views_per_cluster) * 2)),
                    "group_a_ids": [int(item) for item in selection["group_a_ids"]],
                    "group_b_ids": [int(item) for item in selection["group_b_ids"]],
                    "ordered_frame_ids": [int(item) for item in selection["ordered_ids"]],
                    "frames": make_frame_rows(selection, overlap_table),
                    "selection_metrics": {
                        key: float(value)
                        for key, value in selection.items()
                        if isinstance(value, (int, float, np.floating)) and key not in {"anchor_a", "anchor_b"}
                    },
                    "anchor_a": int(selection["anchor_a"]),
                    "anchor_b": int(selection["anchor_b"]),
                }
                write_tuple_meta(tuple_root, tuple_meta)
                row = {
                    "dataset": str(args.dataset_name),
                    "scene": scene_name,
                    "tuple_name": tuple_name,
                    "tuple_root": str(tuple_root),
                    "anchor_a": int(selection["anchor_a"]),
                    "anchor_b": int(selection["anchor_b"]),
                    "cross_overlap_mean": float(selection["cross_overlap_mean"]),
                    "cross_overlap_max": float(selection["cross_overlap_max"]),
                    "anchor_pair_overlap": float(selection["anchor_pair_overlap"]),
                    "center_distance_ratio": float(selection["center_distance_ratio"]),
                    "group_a_internal_overlap_mean": float(selection["group_a_internal_overlap_mean"]),
                    "group_b_internal_overlap_mean": float(selection["group_b_internal_overlap_mean"]),
                }
                if args.write_preview:
                    preview_path = write_preview_grid(tuple_root, tuple_meta, output_root / "previews")
                    row["preview_path"] = str(preview_path)
                tuple_rows.append(row)
        except Exception as exc:
            skipped.append({"scene": scene_name, "reason": repr(exc)})

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "same_scene_low_overlap_5plus5_v1",
        "dataset_name": str(args.dataset_name),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "config": vars(args),
        "num_scenes_seen": int(len(scene_roots)),
        "num_tuples": int(len(tuple_rows)),
        "num_skipped": int(len(skipped)),
        "tuples": tuple_rows,
        "skipped": skipped,
    }
    write_json(output_root / "summary.json", summary)
    write_csv(output_root / "tuples.csv", tuple_rows)
    write_csv(output_root / "skipped.csv", skipped)
    return summary


def main() -> None:
    args = parse_args()
    summary = build_benchmark(args)
    print(f"[same-scene-low-overlap] output_root={summary['output_root']}", flush=True)
    print(
        f"[same-scene-low-overlap] tuples={summary['num_tuples']} skipped={summary['num_skipped']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
