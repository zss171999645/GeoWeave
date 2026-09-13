#!/usr/bin/env python3
"""Resolve CO3Dv2 official benchmark scene roots from official annotation files."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

SEEN_CATEGORIES: Sequence[str] = (
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve official CO3Dv2 benchmark scene roots for EVC-style dataset layouts."
    )
    parser.add_argument("--dataset-root", required=True, help="Root of the EVC-formatted CO3Dv2 dataset.")
    parser.add_argument(
        "--anno-dir",
        default="",
        help="Local directory containing official *_train.jgz/*_test.jgz annotations. If empty, download from Hugging Face.",
    )
    parser.add_argument(
        "--hf-repo",
        default="JianyuanWang/co3d_anno",
        help="Hugging Face dataset repo to use when --anno-dir is not provided.",
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--subset", choices=("seen41", "all51"), default="seen41")
    parser.add_argument("--min-num-images", type=int, default=50)
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames sampled per sequence by official eval.")
    parser.add_argument("--seed", type=int, default=0, help="Official eval random seed.")
    parser.add_argument(
        "--selection-source",
        choices=("auto", "hf_jgz", "co3d_setlists"),
        default="auto",
        help="How to resolve official scenes: downloaded *_test.jgz files or raw CO3D set_lists.",
    )
    parser.add_argument(
        "--co3d-v2-dir",
        default="",
        help="Raw CO3D v2 root containing category/set_lists, frame_annotations.jgz and sequence_annotations.jgz.",
    )
    parser.add_argument(
        "--set-list-tag",
        default="fewview_dev",
        help="Which raw CO3D set_lists file family to use when --selection-source=co3d_setlists.",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.5,
        help="Minimum viewpoint_quality_score when --selection-source=co3d_setlists.",
    )
    parser.add_argument(
        "--translation-sum-threshold",
        type=float,
        default=1e5,
        help="Skip sequences that trip the official test_co3d translation sanity check.",
    )
    parser.add_argument(
        "--scene-filter",
        default="",
        help="Optional comma-separated relative scene ids (category/sequence or sequence).",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow scenes that exist in annotations but are missing under --dataset-root.",
    )
    parser.add_argument("--output-json", default="", help="Optional JSON file to write the resolved scene roots.")
    parser.add_argument(
        "--output-scene-pools-json",
        default="",
        help="Optional JSON file to write official sampled frame indices for each resolved scene.",
    )
    parser.add_argument(
        "--output-anno-dir",
        default="",
        help="Optional directory to write official preprocess-compatible {category}_{split}.jgz annotations.",
    )
    parser.add_argument("--stats-json", default="", help="Optional JSON file to write summary stats.")
    parser.add_argument(
        "--print-format",
        choices=("repr", "json", "none"),
        default="repr",
        help="How to print resolved scene roots to stdout.",
    )
    return parser.parse_args()


def _load_jgz(path: str) -> dict:
    with gzip.open(path, "rb") as f:
        return json.loads(f.read())


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_annotation_files(args: argparse.Namespace) -> List[tuple[str, str]]:
    split_suffix = f"_{args.split}.jgz"
    if args.anno_dir:
        anno_dir = Path(args.anno_dir)
        if not anno_dir.is_dir():
            raise SystemExit(f"annotation dir not found: {anno_dir}")
        paths = sorted(anno_dir.glob(f"*{split_suffix}"))
        return [(p.name[: -len(split_suffix)], str(p)) for p in paths]

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required when --anno-dir is not set. "
            "Install it or pass a local annotation directory."
        ) from exc

    api = HfApi()
    files = sorted(
        f for f in api.list_repo_files(args.hf_repo, repo_type="dataset") if f.endswith(split_suffix)
    )
    out: List[tuple[str, str]] = []
    for filename in files:
        local_path = hf_hub_download(repo_id=args.hf_repo, filename=filename, repo_type="dataset")
        out.append((filename[: -len(split_suffix)], local_path))
    return out


def _iter_co3d_category_dirs(args: argparse.Namespace) -> List[tuple[str, Path]]:
    co3d_v2_dir = Path(args.co3d_v2_dir)
    if not co3d_v2_dir.is_dir():
        raise SystemExit(f"co3d_v2_dir not found: {co3d_v2_dir}")

    category_dirs = {p.name: p for p in co3d_v2_dir.iterdir() if p.is_dir()}
    selected_categories = _select_categories(category_dirs.keys(), args.subset)
    return [(name, category_dirs[name]) for name in selected_categories if name in category_dirs]


def _select_categories(categories: Iterable[str], subset: str) -> List[str]:
    category_set = set(categories)
    if subset == "seen41":
        return [c for c in SEEN_CATEGORIES if c in category_set]
    return sorted(category_set)


def _write_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_jgz(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(json.dumps(payload).encode("utf-8"))


def _sequence_translation_valid(frames: Sequence[dict], threshold: float) -> bool:
    for frame in frames:
        translation = frame.get("T", None)
        if translation is None:
            continue
        try:
            if float(sum(translation)) > threshold:
                return False
        except Exception:
            continue
    return True


def main() -> int:
    args = parse_args()
    dataset_root = os.path.realpath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        raise SystemExit(f"dataset root not found: {dataset_root}")

    requested_scene_filter = {x.strip() for x in args.scene_filter.split(",") if x.strip()}
    scene_roots: List[str] = []
    missing_scenes: List[str] = []
    scene_pool_overrides = {}
    per_category_stats = {}
    seq_total = 0
    seq_ge_min = 0
    seq_translation_valid = 0
    seq_translation_invalid = 0

    selection_source = args.selection_source
    if selection_source == "auto":
        selection_source = "co3d_setlists" if args.co3d_v2_dir else "hf_jgz"

    category_sequences = {}

    if selection_source == "co3d_setlists":
        category_dirs = _iter_co3d_category_dirs(args)
        selected_categories = [name for name, _ in category_dirs]
        for category, category_dir in category_dirs:
            subset_lists_path = category_dir / "set_lists" / f"set_lists_{args.set_list_tag}.json"
            frame_file = category_dir / "frame_annotations.jgz"
            sequence_file = category_dir / "sequence_annotations.jgz"
            subset_lists_data = _load_json(str(subset_lists_path))
            frame_data = _load_jgz(str(frame_file))
            sequence_data = _load_jgz(str(sequence_file))

            frame_data_processed = {}
            for frame_item in frame_data:
                sequence_name = frame_item["sequence_name"]
                frame_data_processed.setdefault(sequence_name, {})[frame_item["frame_number"]] = frame_item

            good_quality_sequences = {
                seq_item["sequence_name"]
                for seq_item in sequence_data
                if seq_item["viewpoint_quality_score"] > args.min_quality
            }

            selected_frames_by_scene = defaultdict(list)
            for sequence_name, frame_number, filepath in subset_lists_data[args.split]:
                if sequence_name not in good_quality_sequences:
                    continue
                if sequence_name not in frame_data_processed or frame_number not in frame_data_processed[sequence_name]:
                    continue
                frame_item = frame_data_processed[sequence_name][frame_number]
                selected_frames_by_scene[sequence_name].append(
                    {
                        "filepath": filepath,
                        "frame_number": frame_number,
                        "R": frame_item["viewpoint"]["R"],
                        "T": frame_item["viewpoint"]["T"],
                        "focal_length": frame_item["viewpoint"]["focal_length"],
                        "principal_point": frame_item["viewpoint"]["principal_point"],
                    }
                )
            category_sequences[category] = dict(selected_frames_by_scene)
    else:
        annotations = dict(_iter_annotation_files(args))
        selected_categories = _select_categories(annotations.keys(), args.subset)
        for category in selected_categories:
            data = _load_jgz(annotations[category])
            filtered = {}
            for sequence_name, frames in sorted(data.items()):
                filtered[sequence_name] = frames
            category_sequences[category] = filtered

    official_rng = np.random.RandomState(args.seed)
    if args.output_anno_dir:
        output_anno_dir = os.path.realpath(args.output_anno_dir)
        for category in selected_categories:
            _write_jgz(
                os.path.join(output_anno_dir, f"{category}_{args.split}.jgz"),
                category_sequences.get(category, {}),
            )
    for category in selected_categories:
        annotation_path = ""
        if selection_source == "co3d_setlists":
            annotation_path = str(Path(args.co3d_v2_dir) / category / "set_lists" / f"set_lists_{args.set_list_tag}.json")
        else:
            annotation_path = annotations[category]

        category_data = category_sequences.get(category, {})
        cat_total = len(category_data)
        cat_ge_min = 0
        cat_translation_valid = 0
        cat_translation_invalid = 0
        cat_found = 0
        cat_missing = 0

        for sequence_name in sorted(category_data):
            frames = category_data[sequence_name]
            rel_scene = f"{category}/{sequence_name}"
            requested_scene = (
                not requested_scene_filter
                or rel_scene in requested_scene_filter
                or sequence_name in requested_scene_filter
            )
            seq_total += 1
            if len(frames) < args.min_num_images:
                continue
            seq_ge_min += 1
            cat_ge_min += 1

            if not _sequence_translation_valid(frames, args.translation_sum_threshold):
                seq_translation_invalid += 1
                cat_translation_invalid += 1
                continue

            seq_translation_valid += 1
            cat_translation_valid += 1

            if args.output_scene_pools_json:
                sampled = official_rng.choice(len(frames), args.num_frames, replace=False).tolist()
                if requested_scene:
                    scene_pool_overrides[rel_scene] = sampled

            if not requested_scene:
                continue

            scene_root = os.path.join(dataset_root, category, sequence_name)
            if os.path.isdir(os.path.join(scene_root, "images")):
                scene_roots.append(scene_root)
                cat_found += 1
            else:
                missing_scenes.append(rel_scene)
                cat_missing += 1

        per_category_stats[category] = {
            "annotation_path": annotation_path,
            "seq_total": cat_total,
            "seq_ge_min_images": cat_ge_min,
            "seq_translation_valid": cat_translation_valid,
            "seq_translation_invalid": cat_translation_invalid,
            "roots_found": cat_found,
            "roots_missing": cat_missing,
        }

    scene_roots = sorted(dict.fromkeys(scene_roots))
    stats = {
        "dataset_root": dataset_root,
        "selection_source": selection_source,
        "annotation_dir": os.path.realpath(args.anno_dir) if args.anno_dir else "",
        "hf_repo": args.hf_repo if not args.anno_dir else "",
        "co3d_v2_dir": os.path.realpath(args.co3d_v2_dir) if args.co3d_v2_dir else "",
        "split": args.split,
        "subset": args.subset,
        "selected_categories": selected_categories,
        "selected_category_count": len(selected_categories),
        "min_num_images": args.min_num_images,
        "num_frames": args.num_frames,
        "seed": args.seed,
        "min_quality": args.min_quality,
        "set_list_tag": args.set_list_tag,
        "translation_sum_threshold": args.translation_sum_threshold,
        "requested_scene_filter_count": len(requested_scene_filter),
        "annotation_seq_total": seq_total,
        "annotation_seq_ge_min_images": seq_ge_min,
        "annotation_seq_translation_valid": seq_translation_valid,
        "annotation_seq_translation_invalid": seq_translation_invalid,
        "output_anno_dir": os.path.realpath(args.output_anno_dir) if args.output_anno_dir else "",
        "resolved_scene_root_count": len(scene_roots),
        "scene_pool_override_count": len(scene_pool_overrides),
        "missing_scene_count": len(missing_scenes),
        "missing_scenes_preview": missing_scenes[:50],
        "per_category": per_category_stats,
    }

    if args.output_json:
        _write_json(args.output_json, scene_roots)
    if args.output_scene_pools_json:
        _write_json(args.output_scene_pools_json, scene_pool_overrides)
    if args.stats_json:
        _write_json(args.stats_json, stats)

    if missing_scenes and not args.allow_missing:
        print(
            f"[co3dv2_official] missing {len(missing_scenes)} scenes under dataset_root={dataset_root}",
            file=sys.stderr,
        )
        return 2

    print(
        "[co3dv2_official] "
        f"source={selection_source} subset={args.subset} split={args.split} categories={len(selected_categories)} "
        f"seq_ge_min={seq_ge_min} seq_eval_valid={seq_translation_valid} roots={len(scene_roots)} missing={len(missing_scenes)}",
        file=sys.stderr,
    )

    if args.print_format == "repr":
        print(repr(scene_roots))
    elif args.print_format == "json":
        print(json.dumps(scene_roots, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
