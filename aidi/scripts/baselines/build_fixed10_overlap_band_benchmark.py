#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_repo_root_on_syspath() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


ensure_repo_root_on_syspath()

from aidi.scripts.baselines.build_evc_same_scene_candidate_pool_benchmark import (  # noqa: E402
    DEFAULT_CANDIDATE_BANDS,
    SOURCE_LAYOUTS,
    classify_candidate_band,
    generic_scene_slug,
    parse_float_list,
    parse_scene_specs,
    resolve_scene_roots,
    vkitti_scene_slug,
)
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


VARIANTS = ("near10", "mixed10", "tail10")


def parse_int_list(raw_value: str) -> List[int]:
    values = [int(item.strip()) for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated integer list")
    return values


def dedup_clean_support(
    ref_id: int,
    overlap_by_ref: Mapping[int, float],
    count: int,
    clean_overlap_threshold: float,
    clean_temporal_dedup: int,
) -> List[int]:
    rows = [
        (int(candidate_id), float(overlap))
        for candidate_id, overlap in overlap_by_ref.items()
        if int(candidate_id) != int(ref_id) and float(overlap) >= float(clean_overlap_threshold)
    ]
    rows.sort(key=lambda item: (-item[1], abs(item[0] - int(ref_id)), item[0]))

    kept: List[int] = []
    for candidate_id, _ in rows:
        if any(abs(candidate_id - existing) < int(clean_temporal_dedup) for existing in kept):
            continue
        kept.append(int(candidate_id))
        if len(kept) == int(count):
            break
    if len(kept) < int(count):
        raise ValueError(f"Not enough clean support frames for ref={ref_id}: need {count}, got {len(kept)}")
    return kept


def top_overlap_ids(ref_id: int, overlap_by_ref: Mapping[int, float], count: int) -> List[int]:
    rows = [
        (int(candidate_id), float(overlap))
        for candidate_id, overlap in overlap_by_ref.items()
        if int(candidate_id) != int(ref_id)
    ]
    rows.sort(key=lambda item: (-item[1], abs(item[0] - int(ref_id)), item[0]))
    if len(rows) < int(count):
        raise ValueError(f"Not enough candidate frames for ref={ref_id}: need {count}, got {len(rows)}")
    return [candidate_id for candidate_id, _ in rows[: int(count)]]


def band_round_robin_ids(
    ref_id: int,
    overlap_by_ref: Mapping[int, float],
    excluded_ids: set[int],
    count: int,
) -> List[int]:
    buckets: Dict[str, List[tuple[int, float]]] = {name: [] for name, _, _ in DEFAULT_CANDIDATE_BANDS}
    for candidate_id, overlap in overlap_by_ref.items():
        candidate_id = int(candidate_id)
        if candidate_id == int(ref_id) or candidate_id in excluded_ids:
            continue
        band = classify_candidate_band(float(overlap))
        if band in buckets:
            buckets[band].append((candidate_id, float(overlap)))
    for rows in buckets.values():
        rows.sort(key=lambda item: (-item[1], abs(item[0] - int(ref_id)), item[0]))

    ordered: List[int] = []
    cursor = 0
    band_names = [name for name, _, _ in DEFAULT_CANDIDATE_BANDS]
    while len(ordered) < int(count) and any(cursor < len(buckets[name]) for name in band_names):
        for name in band_names:
            rows = buckets[name]
            if cursor < len(rows):
                ordered.append(rows[cursor][0])
                if len(ordered) == int(count):
                    break
        cursor += 1
    if len(ordered) < int(count):
        raise ValueError(f"Not enough band candidates for ref={ref_id}: need {count}, got {len(ordered)}")
    return ordered


def lowest_overlap_ids(
    ref_id: int,
    overlap_by_ref: Mapping[int, float],
    excluded_ids: set[int],
    count: int,
) -> List[int]:
    rows = [
        (int(candidate_id), float(overlap))
        for candidate_id, overlap in overlap_by_ref.items()
        if int(candidate_id) != int(ref_id) and int(candidate_id) not in excluded_ids
    ]
    rows.sort(key=lambda item: (item[1], abs(item[0] - int(ref_id)), item[0]))
    if len(rows) < int(count):
        raise ValueError(f"Not enough low-overlap candidates for ref={ref_id}: need {count}, got {len(rows)}")
    return [candidate_id for candidate_id, _ in rows[: int(count)]]


def select_fixed10_overlap_variants(
    ref_id: int,
    overlap_by_ref: Mapping[int, float],
    total_size: int = 10,
    core_size: int = 6,
    clean_overlap_threshold: float = 0.20,
    clean_temporal_dedup: int = 5,
) -> Dict[str, List[int]]:
    if int(total_size) != 10:
        raise ValueError("This builder is intentionally fixed to total_size=10")
    if int(core_size) < 2 or int(core_size) >= int(total_size):
        raise ValueError(f"core_size must be in [2, total_size), got {core_size}")

    ref_id = int(ref_id)
    core_support = dedup_clean_support(
        ref_id=ref_id,
        overlap_by_ref=overlap_by_ref,
        count=int(core_size) - 1,
        clean_overlap_threshold=clean_overlap_threshold,
        clean_temporal_dedup=clean_temporal_dedup,
    )
    core_ids = [ref_id, *core_support]
    extra_count = int(total_size) - int(core_size)

    near_support = top_overlap_ids(ref_id=ref_id, overlap_by_ref=overlap_by_ref, count=int(total_size) - 1)
    mixed_extra = band_round_robin_ids(
        ref_id=ref_id,
        overlap_by_ref=overlap_by_ref,
        excluded_ids=set(core_ids),
        count=extra_count,
    )
    tail_extra = lowest_overlap_ids(
        ref_id=ref_id,
        overlap_by_ref=overlap_by_ref,
        excluded_ids=set(core_ids),
        count=extra_count,
    )
    return {
        "near10": [ref_id, *near_support],
        "mixed10": [*core_ids, *mixed_extra],
        "tail10": [*core_ids, *tail_extra],
    }


def frame_rows_for_variant(
    ordered_ids: Sequence[int],
    ref_id: int,
    core_ids: Sequence[int],
    overlap_by_ref: Mapping[int, float],
) -> List[Dict[str, Any]]:
    core_set = {int(item) for item in core_ids}
    rows: List[Dict[str, Any]] = []
    for out_index, frame_id in enumerate(ordered_ids):
        frame_id = int(frame_id)
        if frame_id == int(ref_id):
            role = "ref"
            band = "ref"
            overlap = 1.0
        elif frame_id in core_set:
            role = "core"
            band = "core"
            overlap = float(overlap_by_ref.get(frame_id, 0.0))
        else:
            role = "candidate"
            overlap = float(overlap_by_ref.get(frame_id, 0.0))
            band = classify_candidate_band(overlap) or "out_of_band"
        rows.append(
            {
                "out_index": int(out_index),
                "frame_id": int(frame_id),
                "role": role,
                "band": band,
                "overlap_to_ref": float(overlap),
                "temporal_delta_to_ref": int(abs(frame_id - int(ref_id))),
            }
        )
    return rows


def write_tuple_meta(tuple_root: Path, tuple_meta: Mapping[str, Any]) -> None:
    (tuple_root / "tuple_meta.json").write_text(
        json.dumps(dict(tuple_meta), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_scene_record(
    scene_root: Path,
    dataset_root: Path,
    dataset_name: str,
    source_layout: str,
    image_subdir: str,
    depth_subdir: str,
    camera_subdir: str,
) -> Dict[str, Any]:
    if source_layout == "eth3d_pi3":
        return build_eth3d_pi3_scene_record(
            seq_root=scene_root,
            scene_name=generic_scene_slug(scene_root, dataset_root),
        )
    if source_layout == "exact_official":
        return build_exact_official_scene_record(scene_root)
    if source_layout == "nested_evc":
        return build_nested_evc_camera_scene_record(
            seq_root=scene_root,
            dataset_name=dataset_name,
            image_rel_path=image_subdir,
            depth_rel_path=depth_subdir,
            scene_name=generic_scene_slug(scene_root, dataset_root),
        )
    scene_name = vkitti_scene_slug(scene_root) if dataset_name in ("vkitti", "vkitti2") else generic_scene_slug(scene_root, dataset_root)
    return build_evc_camera_scene_record(
        seq_root=scene_root,
        dataset_name=dataset_name,
        image_rel_path=image_subdir,
        depth_rel_path=depth_subdir,
        camera_rel_path=camera_subdir,
        scene_name=scene_name,
    )


def build_fixed10_overlap_band_benchmark(
    dataset_root: Path,
    output_root: Path,
    scene_specs: Sequence[str],
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
    limit_scenes: int = 0,
    dataset_name: str = "vkitti2",
    source_layout: str = "evc",
) -> Path:
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if source_layout not in SOURCE_LAYOUTS:
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

    scene_rows: List[Dict[str, Any]] = []
    tuple_rows: List[Dict[str, Any]] = []
    for scene_root in scene_roots:
        scene_record = load_scene_record(
            scene_root=scene_root,
            dataset_root=dataset_root,
            dataset_name=normalized_dataset_name,
            source_layout=source_layout,
            image_subdir=image_subdir,
            depth_subdir=depth_subdir,
            camera_subdir=camera_subdir,
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

        eligible: Dict[int, Dict[str, List[int]]] = {}
        core_by_ref: Dict[int, List[int]] = {}
        for ref_id, overlap_by_ref in overlap_table.items():
            try:
                variants = select_fixed10_overlap_variants(
                    ref_id=int(ref_id),
                    overlap_by_ref=overlap_by_ref,
                    total_size=10,
                    core_size=core_size,
                    clean_overlap_threshold=clean_overlap_threshold,
                    clean_temporal_dedup=clean_temporal_dedup,
                )
            except ValueError:
                continue
            eligible[int(ref_id)] = variants
            core_by_ref[int(ref_id)] = variants["mixed10"][: int(core_size)]

        selected_anchor_ids = select_anchor_ref_ids(
            eligible_ref_ids=list(eligible.keys()),
            anchor_quantiles=[float(item) for item in anchor_quantiles],
            max_anchors=int(max_anchors_per_scene),
        )

        scene_tuple_count = 0
        for ref_id in selected_anchor_ids:
            variants = eligible[int(ref_id)]
            core_ids = core_by_ref[int(ref_id)]
            overlap_by_ref = overlap_table[int(ref_id)]
            for variant_name in VARIANTS:
                ordered_ids = variants[variant_name]
                tuple_name = f"{scene_record['scene_name']}__anchor{int(ref_id):04d}__{variant_name}"
                tuple_root = materialize_tuple_sequence(
                    scene_record=scene_record,
                    tuple_name=tuple_name,
                    ordered_ids=[int(item) for item in ordered_ids],
                    output_root=output_root,
                )
                tuple_meta = {
                    "protocol": "fixed10_overlap_band_v1",
                    "scene_name": str(scene_record["scene_name"]),
                    "source_seq_root": str(scene_root),
                    "tuple_name": tuple_name,
                    "anchor_id": int(ref_id),
                    "variant": variant_name,
                    "total_size": 10,
                    "core_size": int(core_size),
                    "eval_frame_indices": list(range(10)),
                    "core_eval_frame_indices": list(range(int(core_size))),
                    "ordered_frame_ids": [int(item) for item in ordered_ids],
                    "core_frame_ids": [int(item) for item in core_ids],
                    "frames": frame_rows_for_variant(
                        ordered_ids=ordered_ids,
                        ref_id=int(ref_id),
                        core_ids=core_ids,
                        overlap_by_ref=overlap_by_ref,
                    ),
                }
                write_tuple_meta(tuple_root, tuple_meta)
                tuple_meta["tuple_root"] = str(tuple_root)
                tuple_rows.append(tuple_meta)
                scene_tuple_count += 1

        scene_rows.append(
            {
                "scene_name": str(scene_record["scene_name"]),
                "source_seq_root": str(scene_root),
                "num_frames": int(len(scene_record["frame_ids"])),
                "num_eligible_refs": int(len(eligible)),
                "selected_anchor_ids": [int(item) for item in selected_anchor_ids],
                "num_anchors": int(len(selected_anchor_ids)),
                "num_tuples": int(scene_tuple_count),
            }
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "fixed10_overlap_band_v1",
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
            "variants": list(VARIANTS),
            "anchor_quantiles": [float(item) for item in anchor_quantiles],
            "max_anchors_per_scene": int(max_anchors_per_scene),
            "limit_scenes": int(limit_scenes),
            "eval_frame_indices": list(range(10)),
        },
        "num_scenes_scanned": int(len(scene_rows)),
        "num_scenes_kept": int(sum(1 for row in scene_rows if int(row["num_tuples"]) > 0)),
        "num_anchors": int(sum(int(row["num_anchors"]) for row in scene_rows)),
        "num_tuples": int(len(tuple_rows)),
        "scenes": scene_rows,
        "tuples": tuple_rows,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed-10 same-scene overlap-band tuples.")
    parser.add_argument("--dataset", default="vkitti2")
    parser.add_argument("--source-layout", choices=SOURCE_LAYOUTS, default="evc")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-list", default="")
    parser.add_argument("--limit-scenes", type=int, default=0)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = build_fixed10_overlap_band_benchmark(
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        scene_specs=parse_scene_specs(args.scene_list),
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
        limit_scenes=args.limit_scenes,
        dataset_name=args.dataset,
        source_layout=args.source_layout,
    )
    print(f"[fixed10-overlap] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
