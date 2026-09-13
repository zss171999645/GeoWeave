#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from PIL import Image, ImageDraw


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_repo_root_on_syspath() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


ensure_repo_root_on_syspath()

from aidi.scripts.baselines.overlap_noise_seq_map_utils import (  # noqa: E402
    build_exact_official_scene_record,
    build_eth3d_pi3_scene_record,
    build_evc_camera_scene_record,
    build_nested_evc_camera_scene_record,
    build_scene_overlap_table,
    materialize_tuple_sequence,
    select_anchor_ref_ids,
    subsample_scene_record,
)


DEFAULT_VKITTI_ROOT = "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/datasets/vkitti2_resolved"
DEFAULT_SCENE_LIST = "Scene01/clone,Scene06/clone,Scene18/clone,Scene20/clone"
SOURCE_LAYOUTS = ("evc", "nested_evc", "eth3d_pi3", "exact_official")
DEFAULT_POOL_SIZES = (6, 10, 14, 20)
DEFAULT_CANDIDATE_BANDS = (
    ("near", 0.10, 0.20),
    ("mid", 0.03, 0.10),
    ("far", 0.005, 0.03),
    ("tail", 0.0, 0.005),
)
ORDER_MODES = ("core_first", "candidate_first", "interleave")


class CandidateRow:
    def __init__(self, frame_id: int, overlap: float, band: str, temporal_delta: int) -> None:
        self.frame_id = int(frame_id)
        self.overlap = float(overlap)
        self.band = str(band)
        self.temporal_delta = int(temporal_delta)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": int(self.frame_id),
            "overlap": float(self.overlap),
            "band": str(self.band),
            "temporal_delta": int(self.temporal_delta),
        }


class SameScenePoolSelection:
    def __init__(
        self,
        ref_id: int,
        core_ids: List[int],
        candidate_rows: List[CandidateRow],
        pool_ids_by_size: Dict[int, List[int]],
    ) -> None:
        self.ref_id = int(ref_id)
        self.core_ids = [int(item) for item in core_ids]
        self.candidate_rows = list(candidate_rows)
        self.pool_ids_by_size = {
            int(pool_size): [int(frame_id) for frame_id in ordered_ids]
            for pool_size, ordered_ids in pool_ids_by_size.items()
        }


def classify_candidate_band(overlap: float, bands: Sequence[tuple[str, float, float]] = DEFAULT_CANDIDATE_BANDS) -> str:
    score = float(overlap)
    for name, low, high in bands:
        if float(low) <= score < float(high):
            return str(name)
    return ""


def parse_int_list(raw_value: str) -> List[int]:
    values = [int(item.strip()) for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated integer list")
    return values


def parse_float_list(raw_value: str) -> List[float]:
    values = [float(item.strip()) for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated float list")
    return values


def parse_scene_specs(raw_value: str) -> List[str]:
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def normalize_pool_sizes(pool_sizes: Sequence[int], core_size: int) -> List[int]:
    normalized = sorted({int(item) for item in pool_sizes})
    if not normalized:
        raise ValueError("pool_sizes cannot be empty")
    too_small = [item for item in normalized if item < int(core_size)]
    if too_small:
        raise ValueError(f"pool sizes must be >= core_size={core_size}: {too_small}")
    return normalized


def order_candidate_rows_round_robin(
    candidate_rows: Sequence[CandidateRow],
    candidate_bands: Sequence[tuple[str, float, float]],
) -> List[CandidateRow]:
    buckets: Dict[str, List[CandidateRow]] = {str(name): [] for name, _, _ in candidate_bands}
    for row in candidate_rows:
        if row.band in buckets:
            buckets[row.band].append(row)
    for rows in buckets.values():
        rows.sort(key=lambda row: (-float(row.overlap), int(row.temporal_delta), int(row.frame_id)))

    ordered: List[CandidateRow] = []
    band_names = [str(name) for name, _, _ in candidate_bands]
    cursor = 0
    while any(cursor < len(buckets[band_name]) for band_name in band_names):
        for band_name in band_names:
            rows = buckets[band_name]
            if cursor < len(rows):
                ordered.append(rows[cursor])
        cursor += 1
    return ordered


def select_same_scene_candidate_pool(
    ref_id: int,
    overlap_by_ref: Mapping[int, float],
    core_size: int = 6,
    pool_sizes: Sequence[int] = DEFAULT_POOL_SIZES,
    clean_overlap_threshold: float = 0.20,
    clean_temporal_dedup: int = 5,
    candidate_bands: Sequence[tuple[str, float, float]] = DEFAULT_CANDIDATE_BANDS,
) -> SameScenePoolSelection:
    ref_id = int(ref_id)
    core_size = int(core_size)
    if core_size < 2:
        raise ValueError(f"core_size must be at least 2, got {core_size}")
    pool_sizes = normalize_pool_sizes(pool_sizes=pool_sizes, core_size=core_size)

    clean_candidates = [
        (int(candidate_id), float(overlap))
        for candidate_id, overlap in overlap_by_ref.items()
        if int(candidate_id) != ref_id and float(overlap) >= float(clean_overlap_threshold)
    ]
    clean_candidates.sort(key=lambda item: (-item[1], abs(item[0] - ref_id), item[0]))

    clean_support_ids: List[int] = []
    for candidate_id, _ in clean_candidates:
        if any(abs(candidate_id - kept_id) < int(clean_temporal_dedup) for kept_id in clean_support_ids):
            continue
        clean_support_ids.append(int(candidate_id))
        if len(clean_support_ids) == core_size - 1:
            break

    if len(clean_support_ids) < core_size - 1:
        raise ValueError(f"Not enough clean core candidates for ref={ref_id}")

    core_ids = [ref_id, *clean_support_ids]
    excluded_ids = set(core_ids)
    band_order = {name: index for index, (name, _, _) in enumerate(candidate_bands)}
    candidate_rows: List[CandidateRow] = []
    for candidate_id, overlap in overlap_by_ref.items():
        candidate_id = int(candidate_id)
        if candidate_id in excluded_ids or candidate_id == ref_id:
            continue
        band = classify_candidate_band(float(overlap), bands=candidate_bands)
        if not band:
            continue
        candidate_rows.append(
            CandidateRow(
                frame_id=candidate_id,
                overlap=float(overlap),
                band=band,
                temporal_delta=abs(candidate_id - ref_id),
            )
        )
    candidate_rows.sort(
        key=lambda row: (
            band_order.get(row.band, 10_000),
            -float(row.overlap),
            int(row.temporal_delta),
            int(row.frame_id),
        )
    )
    candidate_rows = order_candidate_rows_round_robin(candidate_rows, candidate_bands=candidate_bands)

    max_suffix = max(pool_sizes) - core_size
    if len(candidate_rows) < max_suffix:
        raise ValueError(
            f"Not enough same-scene candidate frames for ref={ref_id}: "
            f"need {max_suffix}, got {len(candidate_rows)}"
        )

    pool_ids_by_size = {
        int(pool_size): [*core_ids, *[row.frame_id for row in candidate_rows[: int(pool_size) - core_size]]]
        for pool_size in pool_sizes
    }
    return SameScenePoolSelection(
        ref_id=ref_id,
        core_ids=core_ids,
        candidate_rows=candidate_rows,
        pool_ids_by_size=pool_ids_by_size,
    )


def discover_vkitti_scene_roots(dataset_root: Path, variant_filter: str = "clone") -> List[Path]:
    dataset_root = Path(dataset_root)
    if (dataset_root / "images" / "00").is_dir() and (dataset_root / "depths" / "00").is_dir():
        if variant_filter and dataset_root.name != variant_filter:
            return []
        return [dataset_root]
    roots: List[Path] = []
    for scene_root in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        for variant_root in sorted(path for path in scene_root.iterdir() if path.is_dir()):
            if variant_filter and variant_root.name != variant_filter:
                continue
            if (variant_root / "images" / "00").is_dir() and (variant_root / "depths" / "00").is_dir():
                roots.append(variant_root)
    return roots


def has_generic_evc_scene_layout(
    scene_root: Path,
    image_subdir: str = "images/00",
    depth_subdir: str = "depths/00",
    camera_subdir: str = "cameras/00",
) -> bool:
    return (
        (scene_root / image_subdir).is_dir()
        and (scene_root / depth_subdir).is_dir()
        and (scene_root / camera_subdir).is_dir()
    )


def discover_generic_evc_scene_roots(
    dataset_root: Path,
    image_subdir: str = "images/00",
    depth_subdir: str = "depths/00",
    camera_subdir: str = "cameras/00",
) -> List[Path]:
    dataset_root = Path(dataset_root)
    if has_generic_evc_scene_layout(dataset_root, image_subdir, depth_subdir, camera_subdir):
        return [dataset_root]

    roots: List[Path] = []
    for scene_root in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        if has_generic_evc_scene_layout(scene_root, image_subdir, depth_subdir, camera_subdir):
            roots.append(scene_root)
            continue
        for nested_root in sorted(path for path in scene_root.iterdir() if path.is_dir()):
            if has_generic_evc_scene_layout(nested_root, image_subdir, depth_subdir, camera_subdir):
                roots.append(nested_root)
    return roots


def has_nested_evc_scene_layout(
    scene_root: Path,
    image_subdir: str = "images",
    depth_subdir: str = "depths",
    intri_name: str = "intri.yml",
    extri_name: str = "extri.yml",
) -> bool:
    return (
        (scene_root / image_subdir).is_dir()
        and (scene_root / depth_subdir).is_dir()
        and (scene_root / intri_name).is_file()
        and (scene_root / extri_name).is_file()
    )


def discover_nested_evc_scene_roots(
    dataset_root: Path,
    image_subdir: str = "images",
    depth_subdir: str = "depths",
) -> List[Path]:
    dataset_root = Path(dataset_root)
    if has_nested_evc_scene_layout(dataset_root, image_subdir=image_subdir, depth_subdir=depth_subdir):
        return [dataset_root]

    roots: List[Path] = []
    for scene_root in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        if has_nested_evc_scene_layout(scene_root, image_subdir=image_subdir, depth_subdir=depth_subdir):
            roots.append(scene_root)
            continue
        for nested_root in sorted(path for path in scene_root.iterdir() if path.is_dir()):
            if has_nested_evc_scene_layout(nested_root, image_subdir=image_subdir, depth_subdir=depth_subdir):
                roots.append(nested_root)
    return roots


def has_eth3d_pi3_scene_layout(scene_root: Path) -> bool:
    return (
        (scene_root / "images" / "custom_undistorted").is_dir()
        and (scene_root / "ground_truth_depth" / "custom_undistorted").is_dir()
        and (scene_root / "custom_undistorted_cam").is_dir()
    )


def discover_eth3d_pi3_scene_roots(dataset_root: Path) -> List[Path]:
    dataset_root = Path(dataset_root)
    if has_eth3d_pi3_scene_layout(dataset_root):
        return [dataset_root]
    return sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and has_eth3d_pi3_scene_layout(path)
    )


def has_exact_official_scene_layout(scene_root: Path) -> bool:
    return (
        (scene_root / "color_90").is_dir()
        and (scene_root / "depth_90").is_dir()
        and (scene_root / "pose_90.txt").is_file()
    )


def discover_exact_official_scene_roots(dataset_root: Path) -> List[Path]:
    dataset_root = Path(dataset_root)
    if has_exact_official_scene_layout(dataset_root):
        return [dataset_root]
    return sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and has_exact_official_scene_layout(path)
    )


def vkitti_scene_slug(seq_root: Path) -> str:
    return f"{seq_root.parent.name}-{seq_root.name}-cam00"


def generic_scene_slug(seq_root: Path, dataset_root: Path) -> str:
    try:
        rel = seq_root.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        rel = Path(seq_root.name)
    return rel.as_posix().strip("/").replace("/", "-")


def resolve_scene_roots(
    dataset_root: Path,
    scene_specs: Sequence[str],
    limit_scenes: int = 0,
    dataset_name: str = "vkitti2",
    source_layout: str = "evc",
    image_subdir: str = "images/00",
    depth_subdir: str = "depths/00",
    camera_subdir: str = "cameras/00",
) -> List[Path]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    if scene_specs:
        roots = [
            Path(spec).expanduser().resolve() if Path(spec).is_absolute() else (dataset_root / spec).resolve()
            for spec in scene_specs
        ]
    elif source_layout == "eth3d_pi3":
        roots = discover_eth3d_pi3_scene_roots(dataset_root=dataset_root)
    elif source_layout == "exact_official":
        roots = discover_exact_official_scene_roots(dataset_root=dataset_root)
    elif source_layout == "nested_evc":
        roots = discover_nested_evc_scene_roots(
            dataset_root=dataset_root,
            image_subdir=image_subdir,
            depth_subdir=depth_subdir,
        )
    elif str(dataset_name).lower() in ("vkitti", "vkitti2"):
        roots = discover_vkitti_scene_roots(dataset_root=dataset_root, variant_filter="clone")
    else:
        roots = discover_generic_evc_scene_roots(
            dataset_root=dataset_root,
            image_subdir=image_subdir,
            depth_subdir=depth_subdir,
            camera_subdir=camera_subdir,
        )
    if limit_scenes > 0:
        roots = roots[: int(limit_scenes)]
    return roots


def write_evc_dir_links(tuple_root: Path, camera: str = "00") -> None:
    for parent_name, target_name in (("images", "color_90"), ("depths", "depth_90")):
        parent = tuple_root / parent_name
        parent.mkdir(parents=True, exist_ok=True)
        link_path = parent / camera
        if link_path.exists() or link_path.is_symlink():
            continue
        link_path.symlink_to(Path("..") / target_name)


def build_frame_rows(
    ordered_ids: Sequence[int],
    selection: SameScenePoolSelection,
    overlap_by_ref: Mapping[int, float],
    core_size: int,
) -> List[Dict[str, Any]]:
    candidate_by_id = {row.frame_id: row for row in selection.candidate_rows}
    core_ids = {int(frame_id) for frame_id in selection.core_ids}
    rows: List[Dict[str, Any]] = []
    for out_index, frame_id in enumerate(ordered_ids):
        frame_id = int(frame_id)
        if frame_id == int(selection.ref_id):
            role = "ref"
            band = "ref"
            overlap = 1.0
        elif frame_id in core_ids:
            role = "core"
            band = "core"
            overlap = float(overlap_by_ref.get(frame_id, 0.0))
        else:
            role = "candidate"
            candidate = candidate_by_id[frame_id]
            band = candidate.band
            overlap = float(candidate.overlap)
        rows.append(
            {
                "out_index": int(out_index),
                "frame_id": int(frame_id),
                "role": role,
                "band": band,
                "overlap_to_ref": float(overlap),
                "temporal_delta_to_ref": int(abs(frame_id - selection.ref_id)),
            }
        )
    return rows


def order_pool_frame_ids(base_ordered_ids: Sequence[int], core_size: int, order_mode: str) -> tuple[List[int], List[int]]:
    order_mode = str(order_mode)
    if order_mode not in ORDER_MODES:
        raise ValueError(f"Unsupported order_mode={order_mode!r}; expected one of {ORDER_MODES}")
    core_size = int(core_size)
    base_ids = [int(item) for item in base_ordered_ids]
    core_ids = base_ids[:core_size]
    candidate_ids = base_ids[core_size:]
    if len(core_ids) != core_size:
        raise ValueError(f"Not enough core frames: expected {core_size}, got {len(core_ids)}")

    if order_mode == "core_first":
        return base_ids, list(range(core_size))
    if order_mode == "candidate_first":
        return [*candidate_ids, *core_ids], list(range(len(candidate_ids), len(candidate_ids) + core_size))

    ordered_ids: List[int] = []
    eval_indices: List[int] = []
    max_len = max(len(core_ids), len(candidate_ids))
    for index in range(max_len):
        if index < len(core_ids):
            eval_indices.append(len(ordered_ids))
            ordered_ids.append(core_ids[index])
        if index < len(candidate_ids):
            ordered_ids.append(candidate_ids[index])
    return ordered_ids, eval_indices


def try_write_tuple_meta(tuple_root: Any, tuple_meta: Mapping[str, Any]) -> None:
    if not isinstance(tuple_root, Path):
        return
    (tuple_root / "tuple_meta.json").write_text(
        json.dumps(dict(tuple_meta), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_preview_grid(tuple_root: Path, tuple_meta: Mapping[str, Any], preview_root: Path, thumb_width: int = 220) -> Path:
    color_dir = tuple_root / "color_90"
    image_paths = sorted(
        path
        for path in color_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    frame_rows = list(tuple_meta.get("frames", []))
    if not image_paths:
        raise FileNotFoundError(f"No preview images under {color_dir}")

    cells: List[Image.Image] = []
    label_height = 34
    for index, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        scale = float(thumb_width) / max(float(image.width), 1.0)
        thumb_height = max(1, int(round(float(image.height) * scale)))
        image = image.resize((int(thumb_width), int(thumb_height)), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (int(thumb_width), int(thumb_height) + label_height), "white")
        cell.paste(image, (0, label_height))
        draw = ImageDraw.Draw(cell)
        row = frame_rows[index] if index < len(frame_rows) else {}
        label = (
            f"{index:02d} {row.get('role', '')}/{row.get('band', '')} "
            f"id={row.get('frame_id', '')} O={float(row.get('overlap_to_ref', 0.0)):.3f}"
        )
        draw.text((4, 4), label[:48], fill=(0, 0, 0))
        cells.append(cell)

    columns = min(5, len(cells))
    rows = (len(cells) + columns - 1) // columns
    cell_width = max(cell.width for cell in cells)
    cell_height = max(cell.height for cell in cells)
    grid = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    for index, cell in enumerate(cells):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        grid.paste(cell, (x, y))

    preview_root.mkdir(parents=True, exist_ok=True)
    preview_path = preview_root / f"{tuple_root.name}.jpg"
    grid.save(preview_path, quality=92)
    return preview_path


def build_same_scene_candidate_pool_benchmark(
    dataset_root: Path,
    output_root: Path,
    scene_specs: Sequence[str],
    pool_sizes: Sequence[int],
    sample_stride: int,
    depth_rel_tol: float,
    clean_overlap_threshold: float,
    clean_temporal_dedup: int,
    anchor_quantiles: Sequence[float],
    max_anchors_per_scene: int,
    image_subdir: str,
    depth_subdir: str,
    camera_subdir: str,
    target_num_frames: int = 90,
    source_prestride: int = 3,
    core_size: int = 6,
    order_mode: str = "core_first",
    write_preview_grids: bool = False,
    limit_scenes: int = 0,
    dataset_name: str = "vkitti2",
    source_layout: str = "evc",
    write_evc_links: bool = False,
    evc_link_camera: str = "00",
) -> Path:
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    pool_sizes = normalize_pool_sizes(pool_sizes=pool_sizes, core_size=core_size)
    if str(order_mode) not in ORDER_MODES:
        raise ValueError(f"Unsupported order_mode={order_mode!r}; expected one of {ORDER_MODES}")
    if str(source_layout) not in SOURCE_LAYOUTS:
        raise ValueError(f"Unsupported source_layout={source_layout!r}; expected one of {SOURCE_LAYOUTS}")
    normalized_dataset_name = "vkitti2" if str(dataset_name).lower() == "vkitti" else str(dataset_name)
    scene_roots = resolve_scene_roots(
        dataset_root=dataset_root,
        scene_specs=scene_specs,
        limit_scenes=limit_scenes,
        dataset_name=normalized_dataset_name,
        source_layout=source_layout,
        image_subdir=image_subdir,
        depth_subdir=depth_subdir,
        camera_subdir=camera_subdir,
    )
    per_scene_rows: List[Dict[str, Any]] = []
    tuple_rows: List[Dict[str, Any]] = []

    for scene_root in scene_roots:
        if source_layout == "eth3d_pi3":
            scene_record = build_eth3d_pi3_scene_record(
                seq_root=scene_root,
                scene_name=generic_scene_slug(scene_root, dataset_root),
            )
        elif source_layout == "exact_official":
            scene_record = build_exact_official_scene_record(scene_root)
        elif source_layout == "nested_evc":
            scene_record = build_nested_evc_camera_scene_record(
                seq_root=scene_root,
                dataset_name=normalized_dataset_name,
                image_rel_path=image_subdir,
                depth_rel_path=depth_subdir,
                scene_name=generic_scene_slug(scene_root, dataset_root),
            )
        else:
            scene_name = (
                vkitti_scene_slug(scene_root)
                if normalized_dataset_name == "vkitti2"
                else generic_scene_slug(scene_root, dataset_root)
            )
            scene_record = build_evc_camera_scene_record(
                seq_root=scene_root,
                dataset_name=normalized_dataset_name,
                image_rel_path=image_subdir,
                depth_rel_path=depth_subdir,
                camera_rel_path=camera_subdir,
                scene_name=scene_name,
            )
        scene_record = subsample_scene_record(
            scene_record=scene_record,
            target_num_frames=target_num_frames,
            source_prestride=source_prestride,
        )
        overlap_table = build_scene_overlap_table(
            scene_record=scene_record,
            sample_stride=sample_stride,
            depth_rel_tol=depth_rel_tol,
        )

        eligible_selections: Dict[int, SameScenePoolSelection] = {}
        for ref_id, overlap_by_ref in overlap_table.items():
            try:
                eligible_selections[int(ref_id)] = select_same_scene_candidate_pool(
                    ref_id=int(ref_id),
                    overlap_by_ref=overlap_by_ref,
                    core_size=core_size,
                    pool_sizes=pool_sizes,
                    clean_overlap_threshold=clean_overlap_threshold,
                    clean_temporal_dedup=clean_temporal_dedup,
                )
            except ValueError:
                continue

        selected_anchor_ids = select_anchor_ref_ids(
            eligible_ref_ids=list(eligible_selections.keys()),
            anchor_quantiles=[float(item) for item in anchor_quantiles],
            max_anchors=int(max_anchors_per_scene),
        )

        scene_tuple_count = 0
        for anchor_id in selected_anchor_ids:
            selection = eligible_selections[int(anchor_id)]
            overlap_by_ref = overlap_table[int(anchor_id)]
            for pool_size in pool_sizes:
                base_ordered_ids = selection.pool_ids_by_size[int(pool_size)]
                ordered_ids, eval_frame_indices = order_pool_frame_ids(
                    base_ordered_ids=base_ordered_ids,
                    core_size=core_size,
                    order_mode=order_mode,
                )
                order_suffix = "" if order_mode == "core_first" else f"__order{order_mode}"
                tuple_name = f"{scene_record['scene_name']}__anchor{int(anchor_id):04d}__pool{int(pool_size):02d}{order_suffix}"
                tuple_root = materialize_tuple_sequence(
                    scene_record=scene_record,
                    tuple_name=tuple_name,
                    ordered_ids=ordered_ids,
                    output_root=output_root,
                )
                if write_evc_links and isinstance(tuple_root, Path):
                    write_evc_dir_links(tuple_root=tuple_root, camera=evc_link_camera)
                frame_rows = build_frame_rows(
                    ordered_ids=ordered_ids,
                    selection=selection,
                    overlap_by_ref=overlap_by_ref,
                    core_size=core_size,
                )
                tuple_meta = {
                    "protocol": "same_scene_candidate_pool_v1",
                    "scene_name": str(scene_record["scene_name"]),
                    "source_seq_root": str(scene_root),
                    "tuple_name": tuple_name,
                    "anchor_id": int(anchor_id),
                    "pool_size": int(pool_size),
                    "core_size": int(core_size),
                    "order_mode": str(order_mode),
                    "eval_frame_indices": [int(item) for item in eval_frame_indices],
                    "base_ordered_frame_ids": [int(item) for item in base_ordered_ids],
                    "ordered_frame_ids": [int(item) for item in ordered_ids],
                    "frames": frame_rows,
                }
                try_write_tuple_meta(tuple_root, tuple_meta)
                if write_preview_grids and isinstance(tuple_root, Path):
                    preview_path = write_preview_grid(
                        tuple_root=tuple_root,
                        tuple_meta=tuple_meta,
                        preview_root=output_root / "previews",
                    )
                    tuple_meta["preview_path"] = str(preview_path)
                    try_write_tuple_meta(tuple_root, tuple_meta)
                tuple_meta["tuple_root"] = str(tuple_root)
                tuple_rows.append(tuple_meta)
                scene_tuple_count += 1

        per_scene_rows.append(
            {
                "scene_name": str(scene_record["scene_name"]),
                "source_seq_root": str(scene_root),
                "num_frames": int(len(scene_record["frame_ids"])),
                "num_eligible_refs": int(len(eligible_selections)),
                "selected_anchor_ids": [int(item) for item in selected_anchor_ids],
                "num_anchors": int(len(selected_anchor_ids)),
                "num_pool_tuples": int(scene_tuple_count),
            }
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "same_scene_candidate_pool_v1",
        "config": {
            "dataset": normalized_dataset_name,
            "dataset_root": str(dataset_root),
            "output_root": str(output_root),
            "scene_specs": [str(item) for item in scene_specs],
            "source_layout": str(source_layout),
            "image_subdir": str(image_subdir),
            "depth_subdir": str(depth_subdir),
            "camera_subdir": str(camera_subdir),
            "target_num_frames": int(target_num_frames),
            "source_prestride": int(source_prestride),
            "sample_stride": int(sample_stride),
            "depth_rel_tol": float(depth_rel_tol),
            "clean_overlap_threshold": float(clean_overlap_threshold),
            "clean_temporal_dedup": int(clean_temporal_dedup),
            "core_size": int(core_size),
            "pool_sizes": [int(item) for item in pool_sizes],
            "candidate_bands": [
                {"name": name, "low": float(low), "high": float(high)}
                for name, low, high in DEFAULT_CANDIDATE_BANDS
            ],
            "anchor_quantiles": [float(item) for item in anchor_quantiles],
            "max_anchors_per_scene": int(max_anchors_per_scene),
            "limit_scenes": int(limit_scenes),
            "order_mode": str(order_mode),
            "write_preview_grids": bool(write_preview_grids),
            "write_evc_links": bool(write_evc_links),
            "evc_link_camera": str(evc_link_camera),
            "eval_frame_indices": "tuple_meta",
        },
        "num_scenes_scanned": int(len(per_scene_rows)),
        "num_scenes_kept": int(sum(1 for row in per_scene_rows if int(row["num_pool_tuples"]) > 0)),
        "num_anchors": int(sum(int(row["num_anchors"]) for row in per_scene_rows)),
        "num_pool_tuples": int(len(tuple_rows)),
        "scenes": per_scene_rows,
        "tuples": tuple_rows,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_path


def default_output_root(dataset_name: str) -> Path:
    safe_dataset = str(dataset_name).replace("/", "_")
    return repo_root() / "tmp" / f"{safe_dataset}_same_scene_candidate_pool_{time.strftime('%Y%m%d_%H%M%S')}"


def default_dataset_root(dataset_name: str) -> str:
    if str(dataset_name).lower() in ("vkitti", "vkitti2"):
        return DEFAULT_VKITTI_ROOT
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build same-scene candidate-pool expansion tuples.")
    parser.add_argument("--dataset", default="vkitti2", help="Dataset/camera convention passed to EVC camera loading.")
    parser.add_argument("--source-layout", choices=SOURCE_LAYOUTS, default="evc")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--scene-list", default="")
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--pool-sizes", default=",".join(str(item) for item in DEFAULT_POOL_SIZES))
    parser.add_argument("--core-size", type=int, default=6)
    parser.add_argument("--sample-stride", type=int, default=16)
    parser.add_argument("--depth-rel-tol", type=float, default=0.01)
    parser.add_argument("--clean-overlap-threshold", type=float, default=0.20)
    parser.add_argument("--clean-temporal-dedup", type=int, default=5)
    parser.add_argument("--max-anchors-per-scene", type=int, default=5)
    parser.add_argument("--anchor-quantiles", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--image-subdir", default="images/00")
    parser.add_argument("--depth-subdir", default="depths/00")
    parser.add_argument("--camera-subdir", default="cameras/00")
    parser.add_argument("--target-num-frames", type=int, default=90)
    parser.add_argument("--source-prestride", type=int, default=3)
    parser.add_argument(
        "--order-mode",
        choices=ORDER_MODES,
        default="core_first",
        help="Input frame order variant. tuple_meta.json records matching clean-core eval_frame_indices.",
    )
    parser.add_argument("--write-preview-grids", action="store_true")
    parser.add_argument("--write-evc-links", action="store_true")
    parser.add_argument("--evc-link-camera", default="00")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root or default_dataset_root(args.dataset)
    if not dataset_root:
        raise ValueError("--dataset-root is required for datasets without a built-in default")
    scene_specs = parse_scene_specs(args.scene_list)
    if not scene_specs and str(args.dataset).lower() in ("vkitti", "vkitti2"):
        scene_specs = parse_scene_specs(DEFAULT_SCENE_LIST)
    output_root = Path(args.output_root).expanduser() if args.output_root else default_output_root(args.dataset)
    summary_path = build_same_scene_candidate_pool_benchmark(
        dataset_root=Path(dataset_root),
        output_root=output_root,
        scene_specs=scene_specs,
        pool_sizes=parse_int_list(args.pool_sizes),
        sample_stride=args.sample_stride,
        depth_rel_tol=args.depth_rel_tol,
        clean_overlap_threshold=args.clean_overlap_threshold,
        clean_temporal_dedup=args.clean_temporal_dedup,
        anchor_quantiles=parse_float_list(args.anchor_quantiles),
        max_anchors_per_scene=args.max_anchors_per_scene,
        image_subdir=args.image_subdir,
        depth_subdir=args.depth_subdir,
        camera_subdir=args.camera_subdir,
        target_num_frames=args.target_num_frames,
        source_prestride=args.source_prestride,
        core_size=args.core_size,
        order_mode=args.order_mode,
        write_preview_grids=args.write_preview_grids,
        limit_scenes=args.limit_scenes,
        dataset_name=args.dataset,
        source_layout=args.source_layout,
        write_evc_links=args.write_evc_links,
        evc_link_camera=args.evc_link_camera,
    )
    print(f"[same-scene-pool] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
