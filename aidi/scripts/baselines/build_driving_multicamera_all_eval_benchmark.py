#!/usr/bin/env python3
"""Build driving multi-camera all-eval tuples for Pi3 pose evaluation.

Protocol:
  - 10 input views total.
  - First 5 views come from camera A.
  - Last 5 views come from camera B in the same scene and same time window.
  - All 10 views use poses from the same scene/world and are intended to be
    evaluated. Do not pass --eval-frame-indices to the evaluator.

Output is PI3 official-style tuple layout and can be evaluated by reusing:

  eval_pi3_relpose_distance_protocol.py --datasets vkitti2 --vkitti-root <dataset-root>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_repo_root_on_syspath() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


ensure_repo_root_on_syspath()

SUPPORTED_DATASETS = ("waymo", "kitti", "vkitti2")
DEFAULT_WAYMO_ROOT = (
    "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/tao02.xie/datasets/"
    "scatt3r_evaluation/waymo_open_dataset_v1_4_3"
)
DEFAULT_KITTI_ROOT = (
    "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/tao02.xie/datasets/"
    "kitti/odometry"
)
DEFAULT_VKITTI_ROOT = "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/datasets/vkitti2_resolved"
_HELPERS: Any | None = None


def helpers() -> Any:
    global _HELPERS
    if _HELPERS is None:
        import aidi.scripts.baselines.build_three_dataset_stride4_distractor_benchmark as runtime

        _HELPERS = runtime
    return _HELPERS


def load_scene_payload(scene) -> Dict[str, Any]:
    return helpers().load_scene_payload(scene)


def materialize_pose_tuple(*args, **kwargs):
    return helpers().materialize_pose_tuple(*args, **kwargs)


def paths_and_poses_for_frames(*args, **kwargs):
    return helpers().paths_and_poses_for_frames(*args, **kwargs)


def quantile_starts(*args, **kwargs):
    return helpers().quantile_starts(*args, **kwargs)


def select_stride_frame_names(*args, **kwargs):
    return helpers().select_stride_frame_names(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="waymo,kitti,vkitti2")
    parser.add_argument("--waymo-root", default=DEFAULT_WAYMO_ROOT)
    parser.add_argument("--kitti-root", default=DEFAULT_KITTI_ROOT)
    parser.add_argument("--vkitti-root", default=DEFAULT_VKITTI_ROOT)
    parser.add_argument(
        "--vkitti-scene-specs",
        default="Scene01/clone,Scene06/clone,Scene18/clone,Scene20/clone",
        help="Comma-separated VKITTI scene/weather specs, or 'all'.",
    )
    parser.add_argument("--output-root", default="")
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--samples-per-scene", type=int, default=5)
    parser.add_argument("--anchor-quantiles", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--views-per-camera", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--waymo-camera-pairs", default="03:04,04:03")
    parser.add_argument("--kitti-camera-pairs", default="2:3,3:2")
    parser.add_argument("--vkitti-camera-pairs", default="00:01,01:00")
    parser.add_argument(
        "--time-offset",
        type=int,
        default=0,
        help="Offset in common-frame index space for camera B window.",
    )
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument(
        "--output-image-ext",
        default="",
        help="Output suffix override. Empty keeps source suffix.",
    )
    return parser.parse_args()


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def parse_datasets(value: str) -> List[str]:
    aliases = {"vkitti": "vkitti2", "virtual_kitti": "vkitti2"}
    datasets = [aliases.get(item.lower(), item.lower()) for item in parse_csv_list(value)]
    bad = [item for item in datasets if item not in SUPPORTED_DATASETS]
    if bad:
        raise ValueError(f"Unsupported datasets: {bad}; choose from {SUPPORTED_DATASETS}")
    return datasets


def parse_camera_pairs(value: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in parse_csv_list(value):
        if ":" not in item:
            raise ValueError(f"Camera pair must be formatted as A:B, got {item!r}")
        lhs, rhs = [part.strip() for part in item.split(":", 1)]
        if not lhs or not rhs:
            raise ValueError(f"Invalid camera pair: {item!r}")
        pairs.append((lhs, rhs))
    if not pairs:
        raise ValueError("camera pairs cannot be empty")
    return pairs


def parse_quantiles(value: str) -> List[float]:
    quantiles = [float(item) for item in parse_csv_list(value)]
    if not quantiles:
        raise ValueError("anchor quantiles cannot be empty")
    return quantiles


def default_output_root() -> Path:
    return repo_root() / "tmp" / f"driving_multicamera_all_eval_{time.strftime('%Y%m%d_%H%M%S')}"


def list_subdirs(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.name)


def available_cameras(scene_root: Path) -> List[str]:
    image_root = scene_root / "images"
    if not image_root.is_dir():
        return []
    return sorted([path.name for path in image_root.iterdir() if path.is_dir()])


def make_scene(dataset: str, name: str, root: Path, camera: str):
    if dataset == "waymo":
        return helpers().SceneSpec(dataset, name, root, str(camera), ("png",), has_depth=False)
    if dataset == "kitti":
        return helpers().SceneSpec(dataset, name, root, str(camera), ("png",), has_depth=False)
    if dataset == "vkitti2":
        return helpers().SceneSpec(dataset, name, root, str(camera), ("jpg", "png"), has_depth=False)
    raise ValueError(f"Unsupported dataset: {dataset}")


def discover_scene_specs(dataset: str, args: argparse.Namespace) -> List[Tuple[str, Path]]:
    scenes: List[Tuple[str, Path]] = []
    if dataset == "waymo":
        root = Path(args.waymo_root)
        scenes = [(path.name, path) for path in list_subdirs(root)]
    elif dataset == "kitti":
        root = Path(args.kitti_root)
        scenes = [(path.name, path) for path in list_subdirs(root)]
    elif dataset == "vkitti2":
        root = Path(args.vkitti_root)
        specs = parse_csv_list(args.vkitti_scene_specs)
        if len(specs) == 1 and specs[0].lower() == "all":
            for group in list_subdirs(root):
                for scene_root in list_subdirs(group):
                    scenes.append((f"{group.name}/{scene_root.name}", scene_root))
        else:
            scenes = [(spec, root / spec) for spec in specs]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if int(args.limit_scenes) > 0:
        scenes = scenes[: int(args.limit_scenes)]
    return scenes


def camera_pairs_for_dataset(dataset: str, args: argparse.Namespace) -> List[Tuple[str, str]]:
    if dataset == "waymo":
        return parse_camera_pairs(args.waymo_camera_pairs)
    if dataset == "kitti":
        return parse_camera_pairs(args.kitti_camera_pairs)
    if dataset == "vkitti2":
        return parse_camera_pairs(args.vkitti_camera_pairs)
    raise ValueError(f"Unsupported dataset: {dataset}")


def common_ordered_frames(payload_a: Mapping[str, Any], payload_b: Mapping[str, Any]) -> List[str]:
    common = set(payload_a["frames"]) & set(payload_b["frames"])
    return [name for name in payload_a["frames"] if name in common]


def safe_name(value: str) -> str:
    return str(value).replace("/", "-").replace(" ", "_")


def select_by_names(payload: Mapping[str, Any], frame_names: Sequence[str]):
    return paths_and_poses_for_frames(dict(payload), frame_names)


def build_dataset(dataset: str, output_root: Path, args: argparse.Namespace, quantiles: Sequence[float]) -> Dict[str, Any]:
    dataset_root = output_root / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    total_views = int(args.views_per_camera) * 2
    pair_specs = camera_pairs_for_dataset(dataset, args)
    rows: List[Dict[str, Any]] = []

    for scene_name, scene_root in discover_scene_specs(dataset, args):
        cameras = set(available_cameras(scene_root))
        for camera_a, camera_b in pair_specs:
            if camera_a not in cameras or camera_b not in cameras:
                rows.append(
                    {
                        "dataset": dataset,
                        "scene": scene_name,
                        "camera_a": camera_a,
                        "camera_b": camera_b,
                        "skipped": f"camera pair unavailable; available={sorted(cameras)}",
                    }
                )
                continue
            try:
                payload_a = load_scene_payload(make_scene(dataset, scene_name, scene_root, camera_a))
                payload_b = load_scene_payload(make_scene(dataset, scene_name, scene_root, camera_b))
                common_frames = common_ordered_frames(payload_a, payload_b)
                starts = quantile_starts(
                    num_frames=len(common_frames),
                    total_views=int(args.views_per_camera) + max(0, int(args.time_offset)),
                    stride=int(args.frame_stride),
                    max_anchors=int(args.samples_per_scene),
                    quantiles=quantiles,
                )
            except Exception as exc:
                rows.append(
                    {
                        "dataset": dataset,
                        "scene": scene_name,
                        "camera_a": camera_a,
                        "camera_b": camera_b,
                        "skipped": str(exc),
                    }
                )
                continue

            for start in starts:
                context_start = int(start) + int(args.time_offset)
                try:
                    target_names = select_stride_frame_names(
                        common_frames, int(start), int(args.views_per_camera), int(args.frame_stride)
                    )
                    context_names = select_stride_frame_names(
                        common_frames, int(context_start), int(args.views_per_camera), int(args.frame_stride)
                    )
                    target_images, target_poses = select_by_names(payload_a, target_names)
                    context_images, context_poses = select_by_names(payload_b, context_names)
                    sequence_name = (
                        f"{dataset}-{safe_name(scene_name)}-cam{camera_a}_cam{camera_b}"
                        f"__mc_t{int(start):04d}_c{int(context_start):04d}"
                    )
                    metadata = {
                        "protocol": "driving_multicamera_all_eval_v1",
                        "dataset": dataset,
                        "scene_name": scene_name,
                        "scene_root": str(scene_root),
                        "camera_a": camera_a,
                        "camera_b": camera_b,
                        "target_start": int(start),
                        "context_start": int(context_start),
                        "views_per_camera": int(args.views_per_camera),
                        "total_views": int(total_views),
                        "frame_stride": int(args.frame_stride),
                        "time_offset": int(args.time_offset),
                        "eval_frame_indices": list(range(total_views)),
                        "target_frame_names": target_names,
                        "context_frame_names": context_names,
                        "ordered_frame_names": [f"{camera_a}:{name}" for name in target_names]
                        + [f"{camera_b}:{name}" for name in context_names],
                    }
                    tuple_root = dataset_root / sequence_name
                    materialize_pose_tuple(
                        tuple_root=tuple_root,
                        image_paths=[*target_images, *context_images],
                        poses_c2w=[*target_poses, *context_poses],
                        copy_images=bool(args.copy_images),
                        output_image_ext=str(args.output_image_ext),
                        metadata=metadata,
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "seq": sequence_name,
                            "scene": scene_name,
                            "camera_a": camera_a,
                            "camera_b": camera_b,
                            "target_start": int(start),
                            "context_start": int(context_start),
                            "num_views": int(total_views),
                            "tuple_root": str(tuple_root),
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "dataset": dataset,
                            "scene": scene_name,
                            "camera_a": camera_a,
                            "camera_b": camera_b,
                            "target_start": int(start),
                            "context_start": int(context_start),
                            "skipped": str(exc),
                        }
                    )

    write_rows_csv(dataset_root / "samples.csv", rows)
    num_valid = sum(1 for row in rows if not row.get("skipped"))
    num_skipped = len(rows) - num_valid
    summary = {
        "dataset": dataset,
        "output_root": str(dataset_root),
        "protocol": "driving_multicamera_all_eval_v1",
        "num_valid": int(num_valid),
        "num_skipped": int(num_skipped),
        "num_rows": int(len(rows)),
        "config": {
            "samples_per_scene": int(args.samples_per_scene),
            "views_per_camera": int(args.views_per_camera),
            "frame_stride": int(args.frame_stride),
            "time_offset": int(args.time_offset),
            "camera_pairs": [list(pair) for pair in pair_specs],
        },
        "rows": rows,
    }
    (dataset_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict(row) for row in rows)


def main() -> None:
    args = parse_args()
    if int(args.views_per_camera) < 2:
        raise ValueError("--views-per-camera must be >= 2")
    if int(args.frame_stride) < 1:
        raise ValueError("--frame-stride must be >= 1")

    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else default_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    quantiles = parse_quantiles(args.anchor_quantiles)
    summaries = [
        build_dataset(dataset, output_root, args, quantiles)
        for dataset in parse_datasets(args.datasets)
    ]
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_root": str(output_root),
        "protocol": "driving_multicamera_all_eval_v1",
        "datasets": [summary["dataset"] for summary in summaries],
        "summaries": summaries,
    }
    (output_root / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
