#!/usr/bin/env python3
"""Build the fixed Waymo plausible-wrong trigger diagnostic protocol.

This protocol is intentionally a mechanism diagnostic, not an unbiased
benchmark.  Samples are selected by image/context statistics only, then the
same evaluated prefix is tested under multiple context controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aidi.scripts.baselines import build_three_dataset_stride4_distractor_benchmark as base
from aidi.scripts.baselines.build_waymo_plausible_context_diagnostic import (
    image_feature,
    image_paths,
    image_resolution,
    image_structure_score,
    pose_lines,
    tuple_meta,
)


DEFAULT_OLD_DATA_ROOT = Path(
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/"
    "eval_datasets/three_dataset_stride4_distractor_v1_20260508_2349/waymo"
)
DEFAULT_FIXED_PAIR_DATA_PARENT = Path(
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/"
    "eval_datasets/waymo_fixed_cam_pair_03_04_v1_20260511_1138"
)

VARIANTS = (
    "prefix6_only",
    "repeat_eval_last",
    "clean_tail",
    "interleaved_clean_tail",
    "plausible_wrong_tail",
    "plausible_wrong_reversed",
    "plausible_wrong_tail_clean_gt_tail",
    "plausible_wrong_blurred_tail",
    "gray_tail",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="")
    parser.add_argument(
        "--sample-spec-json",
        default="",
        help=(
            "Optional JSON list with sample_id/base_name/clean_dir/noise_dir. "
            "When provided, construction skips global candidate scanning and only "
            "materializes these frozen protocol samples."
        ),
    )
    parser.add_argument("--old-data-root", default=str(DEFAULT_OLD_DATA_ROOT))
    parser.add_argument("--fixed-pair-data-parent", default=str(DEFAULT_FIXED_PAIR_DATA_PARENT))
    parser.add_argument("--sources", default="old,fixed")
    parser.add_argument("--fixed-pair-tags", default="cam03_clean_cam04_noise,cam04_clean_cam03_noise")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--context-start", type=int, default=6)
    parser.add_argument("--total-views", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--feature-size", default="32x18")
    parser.add_argument("--require-resolution-match", action="store_true", default=True)
    parser.add_argument("--allow-resolution-mismatch", action="store_false", dest="require_resolution_match")
    parser.add_argument("--min-wrong-entropy", type=float, default=4.0)
    parser.add_argument("--min-wrong-edge-density", type=float, default=0.02)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--blur-radius", type=float, default=18.0)
    parser.add_argument("--output-image-ext", default=".jpg")
    return parser.parse_args()


def default_output_root() -> Path:
    return Path(
        "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/"
        f"eval_datasets/waymo_plausible_wrong_trigger_v1_{time.strftime('%Y%m%d_%H%M')}"
    )


def parse_csv_list(text: str) -> List[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def parse_feature_size(text: str) -> Tuple[int, int]:
    raw = str(text).strip().lower().replace(",", "x")
    parts = [item for item in raw.split("x") if item]
    if len(parts) != 2:
        raise ValueError(f"Invalid feature size {text!r}; expected WIDTHxHEIGHT")
    return int(parts[0]), int(parts[1])


def mean_structure(paths: Sequence[Path]) -> Dict[str, float]:
    rows = [image_structure_score(path) for path in paths]
    if not rows:
        return {"edge_density": 0.0, "grad_mean": 0.0, "variance": 0.0, "entropy": 0.0}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def mean_feature(paths: Sequence[Path], size: Tuple[int, int]) -> np.ndarray:
    feats = [image_feature(path, size=size) for path in paths]
    return np.mean(np.stack(feats, axis=0), axis=0)


def feature_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(lhs, dtype=np.float32) - np.asarray(rhs, dtype=np.float32))))


def zscore(values: Sequence[float], value: float) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    sigma = float(np.std(arr))
    if sigma < 1e-12:
        return 0.0
    return float((float(value) - float(np.mean(arr))) / sigma)


def link_or_copy(src: Path, dst: Path, *, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(str(src), str(dst))
    except OSError:
        shutil.copy2(src, dst)


def save_blur(src: Path, dst: Path, radius: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=float(radius))).save(dst, quality=95)


def save_gray(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        Image.new("RGB", image.size, (128, 128, 128)).save(dst, quality=95)


def discover_candidate_pairs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    sources = set(parse_csv_list(args.sources))
    rows: List[Dict[str, Any]] = []
    if "old" in sources:
        old_root = Path(args.old_data_root)
        for clean_dir in sorted(old_root.glob("*__clean")):
            base_name = clean_dir.name[: -len("__clean")]
            noise_dir = old_root / f"{base_name}__noise"
            if noise_dir.is_dir():
                rows.append(
                    {
                        "candidate_source": "old_cross_scene",
                        "base_name": base_name,
                        "clean_dir": str(clean_dir),
                        "noise_dir": str(noise_dir),
                    }
                )
    if "fixed" in sources:
        parent = Path(args.fixed_pair_data_parent)
        for pair_tag in parse_csv_list(args.fixed_pair_tags):
            data_root = parent / pair_tag / "waymo"
            for clean_dir in sorted(data_root.glob("*__clean")):
                base_name = clean_dir.name[: -len("__clean")]
                noise_dir = data_root / f"{base_name}__noise"
                if noise_dir.is_dir():
                    rows.append(
                        {
                            "candidate_source": f"fixed_{pair_tag}",
                            "base_name": base_name,
                            "clean_dir": str(clean_dir),
                            "noise_dir": str(noise_dir),
                        }
                    )
    return rows


def score_candidate(candidate: Mapping[str, Any], args: argparse.Namespace, feature_size: Tuple[int, int]) -> Dict[str, Any]:
    clean_dir = Path(str(candidate["clean_dir"]))
    noise_dir = Path(str(candidate["noise_dir"]))
    clean_images = image_paths(clean_dir, total_views=int(args.total_views))
    noise_images = image_paths(noise_dir, total_views=int(args.total_views))
    context_start = int(args.context_start)

    prefix = clean_images[:context_start]
    clean_tail = clean_images[context_start : int(args.total_views)]
    wrong_tail = noise_images[context_start : int(args.total_views)]
    prefix_scores = mean_structure(prefix)
    clean_scores = mean_structure(clean_tail)
    wrong_scores = mean_structure(wrong_tail)
    prefix_feat = mean_feature(prefix, feature_size)
    clean_feat = mean_feature(clean_tail, feature_size)
    wrong_feat = mean_feature(wrong_tail, feature_size)

    prefix_res = {image_resolution(path) for path in prefix}
    clean_res = {image_resolution(path) for path in clean_tail}
    wrong_res = {image_resolution(path) for path in wrong_tail}
    resolution_match = len(prefix_res) == 1 and clean_res == prefix_res and wrong_res == prefix_res

    out = dict(candidate)
    out.update(
        {
            "sample_id": f"{candidate['candidate_source']}_{candidate['base_name']}",
            "prefix_resolution": json.dumps([list(item) for item in sorted(prefix_res)]),
            "clean_tail_resolution": json.dumps([list(item) for item in sorted(clean_res)]),
            "wrong_tail_resolution": json.dumps([list(item) for item in sorted(wrong_res)]),
            "resolution_match": bool(resolution_match),
            "prefix_clean_distance": feature_distance(prefix_feat, clean_feat),
            "prefix_wrong_distance": feature_distance(prefix_feat, wrong_feat),
            "clean_wrong_distance": feature_distance(clean_feat, wrong_feat),
        }
    )
    out["wrong_more_like_prefix_margin"] = float(out["prefix_clean_distance"]) - float(out["prefix_wrong_distance"])
    for key, value in prefix_scores.items():
        out[f"prefix_{key}"] = float(value)
    for key, value in clean_scores.items():
        out[f"clean_tail_{key}"] = float(value)
    for key, value in wrong_scores.items():
        out[f"wrong_tail_{key}"] = float(value)
        out[f"wrong_minus_clean_{key}"] = float(value) - float(clean_scores[key])
        out[f"prefix_minus_clean_{key}"] = float(prefix_scores[key]) - float(clean_scores[key])
    return out


def select_candidates(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if args.sample_spec_json:
        payload = json.loads(Path(args.sample_spec_json).expanduser().read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"--sample-spec-json must contain a list: {args.sample_spec_json}")
        selected: List[Dict[str, Any]] = []
        for rank, item in enumerate(payload):
            if not isinstance(item, Mapping):
                raise ValueError(f"sample spec #{rank} is not a mapping")
            required = ("sample_id", "base_name", "clean_dir", "noise_dir")
            missing = [key for key in required if key not in item]
            if missing:
                raise ValueError(f"sample spec #{rank} missing keys: {missing}")
            row = dict(item)
            row.setdefault("candidate_source", "frozen_sample_spec")
            row.setdefault("selection_score", 0.0)
            row["selection_rank"] = int(row.get("selection_rank", rank))
            selected.append(row)
        selected.sort(key=lambda row: int(row["selection_rank"]))
        return selected[: int(args.top_k)] if int(args.top_k) > 0 else selected, []

    feature_size = parse_feature_size(args.feature_size)
    scored: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for candidate in discover_candidate_pairs(args):
        try:
            row = score_candidate(candidate, args, feature_size)
        except Exception as exc:
            skipped.append({**candidate, "skip_reason": str(exc)})
            continue
        if bool(args.require_resolution_match) and not bool(row["resolution_match"]):
            row["skip_reason"] = "resolution_mismatch"
            skipped.append(row)
            continue
        if float(row["wrong_tail_entropy"]) < float(args.min_wrong_entropy):
            row["skip_reason"] = "low_wrong_entropy"
            skipped.append(row)
            continue
        if float(row["wrong_tail_edge_density"]) < float(args.min_wrong_edge_density):
            row["skip_reason"] = "low_wrong_edge_density"
            skipped.append(row)
            continue
        scored.append(row)

    if not scored:
        return [], skipped

    fields = [
        "prefix_minus_clean_entropy",
        "prefix_minus_clean_edge_density",
        "wrong_minus_clean_entropy",
        "wrong_minus_clean_edge_density",
        "wrong_tail_entropy",
        "wrong_tail_edge_density",
        "wrong_more_like_prefix_margin",
        "prefix_wrong_distance",
    ]
    values = {field: [float(row[field]) for row in scored] for field in fields}
    for row in scored:
        row["selection_score"] = (
            1.0 * zscore(values["prefix_minus_clean_entropy"], float(row["prefix_minus_clean_entropy"]))
            + 0.6 * zscore(values["prefix_minus_clean_edge_density"], float(row["prefix_minus_clean_edge_density"]))
            + 1.0 * zscore(values["wrong_minus_clean_entropy"], float(row["wrong_minus_clean_entropy"]))
            + 0.6 * zscore(values["wrong_minus_clean_edge_density"], float(row["wrong_minus_clean_edge_density"]))
            + 0.4 * zscore(values["wrong_tail_entropy"], float(row["wrong_tail_entropy"]))
            + 0.4 * zscore(values["wrong_tail_edge_density"], float(row["wrong_tail_edge_density"]))
            + 0.4 * zscore(values["wrong_more_like_prefix_margin"], float(row["wrong_more_like_prefix_margin"]))
            - 0.2 * zscore(values["prefix_wrong_distance"], float(row["prefix_wrong_distance"]))
        )
    scored.sort(key=lambda row: float(row["selection_score"]), reverse=True)
    for rank, row in enumerate(scored):
        row["selection_rank"] = int(rank)
    return scored[: int(args.top_k)], skipped


def interleaved_clean_tail(clean_dir: Path, context_views: int) -> Tuple[List[Path], List[str], List[str]]:
    meta = tuple_meta(clean_dir)
    required = ("target_source_root", "target_scene_name", "target_camera", "anchor_start", "frame_stride")
    missing = [key for key in required if key not in meta]
    if missing:
        raise KeyError(f"cannot build interleaved context, missing meta keys: {missing}")
    start = int(meta["anchor_start"])
    stride = int(meta["frame_stride"])
    offset = max(stride // 2, 1)
    frame_names = [f"{start + offset + idx * stride:06d}" for idx in range(context_views)]
    scene = base.SceneSpec(
        dataset="waymo",
        name=str(meta["target_scene_name"]),
        root=Path(str(meta["target_source_root"])),
        camera=str(meta["target_camera"]),
        image_exts=("png",),
    )
    payload = base.load_scene_payload(scene)
    images, poses_np = base.paths_and_poses_for_frames(payload, frame_names)
    poses = [" ".join(f"{float(value):.8f}" for value in np.asarray(pose).reshape(-1)) for pose in poses_np]
    return images, poses, frame_names


def materialize_variant(
    out_dir: Path,
    images: Sequence[Path],
    poses: Sequence[str],
    *,
    metadata: Mapping[str, Any],
    derived: Mapping[int, str],
    copy_images: bool,
    blur_radius: float,
    output_image_ext: str,
) -> None:
    color_dir = out_dir / "color_90"
    color_dir.mkdir(parents=True, exist_ok=False)
    ext = output_image_ext if str(output_image_ext).startswith(".") else f".{output_image_ext}"
    for idx, src in enumerate(images):
        dst = color_dir / f"frame_{idx:04d}{ext}"
        action = derived.get(idx, "link")
        if action == "blur":
            save_blur(src, dst, blur_radius)
        elif action == "gray":
            save_gray(src, dst)
        else:
            link_or_copy(src, dst, copy_images=copy_images)
    (out_dir / "pose_90.txt").write_text("\n".join(poses) + "\n", encoding="utf-8")
    (out_dir / "tuple_meta.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_sample_variants(sample: Mapping[str, Any], dataset_root: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    clean_dir = Path(str(sample["clean_dir"]))
    noise_dir = Path(str(sample["noise_dir"]))
    total_views = int(args.total_views)
    context_start = int(args.context_start)
    context_views = total_views - context_start
    clean_images = image_paths(clean_dir, total_views=total_views)
    noise_images = image_paths(noise_dir, total_views=total_views)
    clean_poses = pose_lines(clean_dir)
    noise_poses = pose_lines(noise_dir)
    prefix_images = clean_images[:context_start]
    prefix_poses = clean_poses[:context_start]
    clean_tail_images = clean_images[context_start:total_views]
    clean_tail_poses = clean_poses[context_start:total_views]
    wrong_tail_images = noise_images[context_start:total_views]
    wrong_tail_poses = noise_poses[context_start:total_views]
    inter_images, inter_poses, inter_names = interleaved_clean_tail(clean_dir, context_views)

    sample_id = str(sample["sample_id"])
    variants = {
        "prefix6_only": (prefix_images, prefix_poses, {}),
        "repeat_eval_last": (
            prefix_images + [prefix_images[-1]] * context_views,
            prefix_poses + [prefix_poses[-1]] * context_views,
            {},
        ),
        "clean_tail": (prefix_images + clean_tail_images, prefix_poses + clean_tail_poses, {}),
        "interleaved_clean_tail": (prefix_images + inter_images, prefix_poses + inter_poses, {}),
        "plausible_wrong_tail": (prefix_images + wrong_tail_images, prefix_poses + wrong_tail_poses, {}),
        "plausible_wrong_reversed": (
            prefix_images + list(reversed(wrong_tail_images)),
            prefix_poses + list(reversed(wrong_tail_poses)),
            {},
        ),
        "plausible_wrong_tail_clean_gt_tail": (
            prefix_images + wrong_tail_images,
            prefix_poses + clean_tail_poses,
            {},
        ),
        "plausible_wrong_blurred_tail": (
            prefix_images + wrong_tail_images,
            prefix_poses + wrong_tail_poses,
            {idx: "blur" for idx in range(context_start, total_views)},
        ),
        "gray_tail": (
            prefix_images + wrong_tail_images,
            prefix_poses + wrong_tail_poses,
            {idx: "gray" for idx in range(context_start, total_views)},
        ),
    }

    rows: List[Dict[str, Any]] = []
    for variant, (images, poses, derived) in variants.items():
        seq = f"{sample_id}__{variant}"
        out_dir = dataset_root / seq
        meta = {
            "protocol": "waymo_plausible_wrong_trigger_v1",
            "sample_id": sample_id,
            "variant": variant,
            "base_name": sample["base_name"],
            "candidate_source": sample["candidate_source"],
            "selection_rank": sample["selection_rank"],
            "selection_score": sample["selection_score"],
            "context_start": context_start,
            "total_views": len(images),
            "eval_frame_indices": list(range(context_start)),
            "clean_dir": str(clean_dir),
            "noise_dir": str(noise_dir),
            "interleaved_context_frame_names": inter_names if variant == "interleaved_clean_tail" else [],
            "selected_sample_scores": {k: v for k, v in sample.items() if isinstance(v, (str, int, float, bool))},
            "ordered_source_images": [str(path) for path in images],
            "derived_context_actions": {str(key): value for key, value in derived.items()},
        }
        materialize_variant(
            out_dir,
            images,
            poses,
            metadata=meta,
            derived=derived,
            copy_images=bool(args.copy_images),
            blur_radius=float(args.blur_radius),
            output_image_ext=str(args.output_image_ext),
        )
        rows.append(
            {
                "seq": seq,
                "sample_id": sample_id,
                "variant": variant,
                "out_dir": str(out_dir),
                "selection_rank": int(sample["selection_rank"]),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else default_output_root()
    dataset_root = output_root / "waymo"
    dataset_root.mkdir(parents=True, exist_ok=True)
    selected, skipped = select_candidates(args)
    if not selected:
        raise RuntimeError("No selected samples; relax filters or inspect skipped_candidates.csv")

    variant_rows: List[Dict[str, Any]] = []
    for sample in selected:
        variant_rows.extend(build_sample_variants(sample, dataset_root, args))

    write_csv(output_root / "selected_samples.csv", selected)
    write_csv(output_root / "skipped_candidates.csv", skipped)
    write_csv(output_root / "variant_index.csv", variant_rows)
    summary = {
        "protocol": "waymo_plausible_wrong_trigger_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_root": str(output_root),
        "dataset_root": str(dataset_root),
        "variants": list(VARIANTS),
        "eval_frame_indices": list(range(int(args.context_start))),
        "selection": {
            "mode": "data_feature_score_only",
            "sources": parse_csv_list(args.sources),
            "top_k": int(args.top_k),
            "require_resolution_match": bool(args.require_resolution_match),
            "min_wrong_entropy": float(args.min_wrong_entropy),
            "min_wrong_edge_density": float(args.min_wrong_edge_density),
            "feature_size": str(args.feature_size),
        },
        "num_selected_samples": len(selected),
        "num_variants": len(variant_rows),
        "selected_samples_csv": str(output_root / "selected_samples.csv"),
        "variant_index_csv": str(output_root / "variant_index.csv"),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
