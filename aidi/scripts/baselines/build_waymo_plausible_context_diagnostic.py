#!/usr/bin/env python3
"""Build Waymo plausible-context collapse diagnostics.

The diagnostic keeps the evaluated prefix fixed and only changes context
frames 6..9.  This isolates whether a collapse is triggered by structured,
visually plausible but geometrically irrelevant context rather than by the
presence of arbitrary extra frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from PIL import Image, ImageFilter


DEFAULT_OLD_DATA_ROOT = Path(
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/"
    "eval_datasets/three_dataset_stride4_distractor_v1_20260508_2349/waymo"
)
DEFAULT_FIXED_DATA_ROOT = Path(
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/"
    "eval_datasets/waymo_fixed_cam_pair_03_04_v1_20260511_1138/"
    "cam03_clean_cam04_noise/waymo"
)
DEFAULT_FIXED_PAIR_DATA_PARENT = Path(
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/"
    "eval_datasets/waymo_fixed_cam_pair_03_04_v1_20260511_1138"
)
DEFAULT_OLD_RESULT_ROOT = Path(
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/"
    "result/pi3_noise/20260508_three_dataset_stride4_distractor_pi3_official_vs_sparse67"
)
DEFAULT_FIXED_RESULT_ROOT = Path(
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/"
    "result/pi3_noise/20260511_waymo_fixed_cam_pair_03_04_pi3_official_vs_sparse79"
)

DIAGNOSTIC_SAMPLES = (
    {
        "sample_id": "old_cross_scene_371159869_cam04_anchor0048",
        "base_name": "waymo-371159869-cam04__anchor0048",
        "clean_dir": DEFAULT_OLD_DATA_ROOT / "waymo-371159869-cam04__anchor0048__clean",
        "noise_dir": DEFAULT_OLD_DATA_ROOT / "waymo-371159869-cam04__anchor0048__noise",
        "note": "old cross-scene outlier; plausible noise from scene 460417311 cam01",
    },
    {
        "sample_id": "fixed_pair_371159869_cam03_anchor0144",
        "base_name": "waymo-371159869-cam03__anchor0144",
        "clean_dir": DEFAULT_FIXED_DATA_ROOT / "waymo-371159869-cam03__anchor0144__clean",
        "noise_dir": DEFAULT_FIXED_DATA_ROOT / "waymo-371159869-cam03__anchor0144__noise",
        "note": "fixed same-resolution 03->04 non-overlap camera outlier",
    },
)

VARIANTS = (
    "clean_tail",
    "plausible_noise_tail",
    "plausible_noise_reversed",
    "blurred_plausible_noise_tail",
    "gray_tail",
    "repeat_eval_last",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--old-data-root", default=str(DEFAULT_OLD_DATA_ROOT))
    parser.add_argument("--fixed-data-root", default=str(DEFAULT_FIXED_DATA_ROOT))
    parser.add_argument("--fixed-pair-data-parent", default=str(DEFAULT_FIXED_PAIR_DATA_PARENT))
    parser.add_argument("--old-result-root", default=str(DEFAULT_OLD_RESULT_ROOT))
    parser.add_argument("--fixed-result-root", default=str(DEFAULT_FIXED_RESULT_ROOT))
    parser.add_argument("--fixed-pair-tags", default="cam03_clean_cam04_noise,cam04_clean_cam03_noise")
    parser.add_argument(
        "--sample-spec-json",
        default="",
        help="Optional JSON list of sample specs with clean_dir/noise_dir. Overrides hardcoded diagnostic samples.",
    )
    parser.add_argument(
        "--auto-select-top-k",
        type=int,
        default=0,
        help="Automatically select top-K plausible-collapse samples from old/fixed Waymo paired metrics.",
    )
    parser.add_argument(
        "--auto-select-sources",
        default="old,fixed",
        help="Comma-separated sources for --auto-select-top-k: old,fixed.",
    )
    parser.add_argument(
        "--selection-min-official-ate-delta",
        type=float,
        default=0.0,
        help="Drop auto-selected candidates whose official noise-clean ATE delta is below this threshold.",
    )
    parser.add_argument(
        "--auto-select-by-data-score",
        action="store_true",
        help=(
            "Select top-K samples using only image/camera data from existing clean/noise candidate roots, "
            "without looking at model metrics. This is intended to test whether plausible diagnostics can be "
            "constructed without cherry-picking failures."
        ),
    )
    parser.add_argument(
        "--data-score-min-noise-entropy",
        type=float,
        default=0.0,
        help="Optional filter for data-score selection; keeps samples with noise-context entropy >= this value.",
    )
    parser.add_argument(
        "--data-score-min-noise-edge-density",
        type=float,
        default=0.0,
        help="Optional filter for data-score selection; keeps samples with noise-context edge density >= this value.",
    )
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--context-start", type=int, default=6)
    parser.add_argument("--total-views", type=int, default=10)
    parser.add_argument("--blur-radius", type=float, default=18.0)
    parser.add_argument("--score-old-waymo", action="store_true")
    return parser.parse_args()


def default_output_root() -> Path:
    return Path("tmp") / f"waymo_plausible_context_diagnostic_{time.strftime('%Y%m%d_%H%M%S')}"


def image_paths(tuple_dir: Path, total_views: int) -> List[Path]:
    color_dir = tuple_dir / "color_90"
    paths: List[Path] = []
    for idx in range(total_views):
        matches = sorted(color_dir.glob(f"frame_{idx:04d}.*"))
        if not matches:
            raise FileNotFoundError(f"Missing frame_{idx:04d} under {color_dir}")
        paths.append(matches[0])
    return paths


def pose_lines(tuple_dir: Path) -> List[str]:
    path = tuple_dir / "pose_90.txt"
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"No pose lines in {path}")
    return lines


def tuple_meta(tuple_dir: Path) -> Dict[str, object]:
    path = tuple_dir / "tuple_meta.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(str(src), str(dst))
    except FileExistsError:
        raise
    except OSError:
        shutil.copy2(src, dst)


def save_blurred(src: Path, dst: Path, radius: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=float(radius))).save(dst, quality=95)


def save_gray_like(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        gray = Image.new("RGB", image.size, (128, 128, 128))
        gray.save(dst, quality=95)


def materialize_variant(
    out_dir: Path,
    images: Sequence[Path],
    poses: Sequence[str],
    *,
    copy_images: bool,
    derived: Mapping[int, str],
    blur_radius: float,
    metadata: Mapping[str, object],
) -> None:
    color_dir = out_dir / "color_90"
    color_dir.mkdir(parents=True, exist_ok=False)
    for idx, src in enumerate(images):
        dst = color_dir / f"frame_{idx:04d}.jpg"
        action = derived.get(idx, "link")
        if action == "blur":
            save_blurred(src, dst, radius=blur_radius)
        elif action == "gray":
            save_gray_like(src, dst)
        else:
            link_or_copy(src, dst, copy_images=copy_images)
    (out_dir / "pose_90.txt").write_text("\n".join(poses) + "\n", encoding="utf-8")
    (out_dir / "tuple_meta.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_variants_for_sample(
    sample: Mapping[str, object],
    dataset_root: Path,
    *,
    context_start: int,
    total_views: int,
    copy_images: bool,
    blur_radius: float,
) -> List[Dict[str, object]]:
    clean_dir = Path(str(sample["clean_dir"]))
    noise_dir = Path(str(sample["noise_dir"]))
    clean_images = image_paths(clean_dir, total_views=total_views)
    noise_images = image_paths(noise_dir, total_views=total_views)
    clean_poses = pose_lines(clean_dir)
    noise_poses = pose_lines(noise_dir)
    if len(clean_poses) < total_views or len(noise_poses) < total_views:
        raise RuntimeError(f"Pose count mismatch for {sample['sample_id']}")

    prefix_images = clean_images[:context_start]
    prefix_poses = clean_poses[:context_start]
    sample_rows: List[Dict[str, object]] = []
    sample_id = str(sample["sample_id"])

    variant_payloads = {
        "clean_tail": (
            prefix_images + clean_images[context_start:total_views],
            prefix_poses + clean_poses[context_start:total_views],
            {},
        ),
        "plausible_noise_tail": (
            prefix_images + noise_images[context_start:total_views],
            prefix_poses + noise_poses[context_start:total_views],
            {},
        ),
        "plausible_noise_reversed": (
            prefix_images + list(reversed(noise_images[context_start:total_views])),
            prefix_poses + list(reversed(noise_poses[context_start:total_views])),
            {},
        ),
        "blurred_plausible_noise_tail": (
            prefix_images + noise_images[context_start:total_views],
            prefix_poses + noise_poses[context_start:total_views],
            {idx: "blur" for idx in range(context_start, total_views)},
        ),
        "gray_tail": (
            prefix_images + noise_images[context_start:total_views],
            prefix_poses + noise_poses[context_start:total_views],
            {idx: "gray" for idx in range(context_start, total_views)},
        ),
        "repeat_eval_last": (
            prefix_images + [clean_images[context_start - 1]] * (total_views - context_start),
            prefix_poses + [clean_poses[context_start - 1]] * (total_views - context_start),
            {},
        ),
    }

    for variant, (images, poses, derived) in variant_payloads.items():
        seq_name = f"{sample_id}__{variant}"
        out_dir = dataset_root / seq_name
        metadata = {
            "protocol": "waymo_plausible_context_diagnostic_v1",
            "sample_id": sample_id,
            "base_name": sample["base_name"],
            "source": sample.get("source", ""),
            "variant": variant,
            "note": sample.get("note", ""),
            "selection_metrics": sample.get("selection_metrics", {}),
            "data_selection_scores": sample.get("data_selection_scores", {}),
            "context_start": int(context_start),
            "total_views": int(total_views),
            "eval_frame_indices": list(range(context_start)),
            "clean_meta": tuple_meta(clean_dir),
            "noise_meta": tuple_meta(noise_dir),
            "ordered_source_images": [str(path) for path in images],
            "derived_context_actions": {str(key): value for key, value in derived.items()},
        }
        materialize_variant(
            out_dir=out_dir,
            images=images,
            poses=poses,
            copy_images=copy_images,
            derived=derived,
            blur_radius=blur_radius,
            metadata=metadata,
        )
        sample_rows.append(
            {
                "seq": seq_name,
                "sample_id": sample_id,
                "base_name": sample["base_name"],
                "variant": variant,
                "out_dir": str(out_dir),
            }
        )
    return sample_rows


def image_structure_score(path: Path) -> Dict[str, float]:
    with Image.open(path) as image:
        gray = image.convert("L")
        max_side = max(gray.size)
        if max_side > 384:
            scale = 384.0 / float(max_side)
            gray = gray.resize((max(1, int(gray.size[0] * scale)), max(1, int(gray.size[1] * scale))))
        arr = np.asarray(gray, dtype=np.float32) / 255.0
    dx = np.diff(arr, axis=1, prepend=arr[:, :1])
    dy = np.diff(arr, axis=0, prepend=arr[:1, :])
    grad = np.sqrt(dx * dx + dy * dy)
    hist, _ = np.histogram(arr, bins=64, range=(0.0, 1.0), density=False)
    prob = hist.astype(np.float64) / max(float(hist.sum()), 1.0)
    entropy = -float(np.sum([p * math.log(p + 1e-12, 2.0) for p in prob]))
    return {
        "edge_density": float(np.mean(grad > 0.06)),
        "grad_mean": float(np.mean(grad)),
        "variance": float(np.var(arr)),
        "entropy": entropy,
    }


def mean_scores(paths: Iterable[Path]) -> Dict[str, float]:
    rows = [image_structure_score(path) for path in paths]
    if not rows:
        return {"edge_density": 0.0, "grad_mean": 0.0, "variance": 0.0, "entropy": 0.0}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def image_feature(path: Path, size: tuple[int, int] = (32, 18)) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    thumbnail = arr.reshape(-1)
    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(arr[..., channel], bins=16, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float32)
        hist /= max(float(hist.sum()), 1.0)
        hist_parts.append(hist)
    feature = np.concatenate([thumbnail, *hist_parts], axis=0).astype(np.float32)
    feature -= float(feature.mean())
    norm = float(np.linalg.norm(feature))
    if norm > 1e-6:
        feature /= norm
    return feature


def sequence_feature_distance(lhs: Sequence[Path], rhs: Sequence[Path]) -> float:
    if len(lhs) != len(rhs):
        raise ValueError(f"Feature sequence length mismatch: {len(lhs)} vs {len(rhs)}")
    distances = []
    for left_path, right_path in zip(lhs, rhs):
        left = image_feature(Path(left_path))
        right = image_feature(Path(right_path))
        distances.append(float(np.mean(np.square(left - right))))
    return float(np.mean(distances)) if distances else float("inf")


def image_resolution(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.size[0]), int(image.size[1])


def candidate_from_dirs(
    *,
    source: str,
    base: str,
    clean_dir: Path,
    noise_dir: Path,
    note: str,
) -> Dict[str, object]:
    safe_source = str(source).replace("/", "_").replace(" ", "_")
    safe_base = str(base).replace("/", "_")
    return {
        "sample_id": f"{safe_source}_{safe_base}",
        "base_name": str(base),
        "clean_dir": str(clean_dir),
        "noise_dir": str(noise_dir),
        "source": str(source),
        "note": str(note),
    }


def score_candidate_by_data(
    sample: Mapping[str, object],
    *,
    context_start: int,
    total_views: int,
) -> Dict[str, object]:
    clean_dir = Path(str(sample["clean_dir"]))
    noise_dir = Path(str(sample["noise_dir"]))
    clean_images = image_paths(clean_dir, total_views=total_views)
    noise_images = image_paths(noise_dir, total_views=total_views)
    clean_context = clean_images[int(context_start) : int(total_views)]
    noise_context = noise_images[int(context_start) : int(total_views)]
    if not clean_context or not noise_context:
        raise ValueError(f"Missing context frames for {sample.get('sample_id', sample.get('base_name', 'unknown'))}")

    visual_distance = sequence_feature_distance(clean_context, noise_context)
    clean_scores = mean_scores(clean_context)
    noise_scores = mean_scores(noise_context)
    clean_resolutions = {image_resolution(path) for path in clean_images[: int(context_start)]}
    noise_resolutions = {image_resolution(path) for path in noise_context}
    resolution_match = len(clean_resolutions) == 1 and noise_resolutions == clean_resolutions

    # The score is only used for ranking; report raw fields so the rule remains inspectable.
    # Visual similarity is primary, while structured/non-empty context is a secondary preference.
    structure_bonus = 0.01 * float(noise_scores["entropy"]) + 0.1 * float(noise_scores["edge_density"])
    plausibility_score = -float(visual_distance) + float(structure_bonus)
    data_scores: Dict[str, object] = {
        "visual_distance": float(visual_distance),
        "plausibility_score": float(plausibility_score),
        "resolution_match": bool(resolution_match),
        "clean_context_resolution": [list(item) for item in sorted(clean_resolutions)],
        "noise_context_resolution": [list(item) for item in sorted(noise_resolutions)],
    }
    for key, value in clean_scores.items():
        data_scores[f"clean_context_{key}"] = float(value)
    for key, value in noise_scores.items():
        data_scores[f"noise_context_{key}"] = float(value)
        data_scores[f"noise_minus_clean_{key}"] = float(value) - float(clean_scores[key])
    return data_scores


def read_metric_pairs(csv_path: Path) -> Dict[str, Dict[str, float]]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    paired: Dict[str, Dict[str, float]] = {}
    for row in rows:
        seq = row["seq"]
        if seq.endswith("__clean"):
            base = seq[: -len("__clean")]
            suffix = "clean"
        elif seq.endswith("__noise"):
            base = seq[: -len("__noise")]
            suffix = "noise"
        else:
            continue
        entry = paired.setdefault(base, {})
        for key in ("ATE", "RPE trans", "RPE rot"):
            entry[f"{suffix}_{key}"] = float(row[key])
    return {key: value for key, value in paired.items() if "clean_ATE" in value and "noise_ATE" in value}


def parse_csv_list(text: str) -> List[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def load_sample_spec_json(path: Path) -> List[Dict[str, object]]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    samples: List[Dict[str, object]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"Sample spec #{idx} is not a mapping")
        if "sample_id" not in item or "clean_dir" not in item or "noise_dir" not in item:
            raise ValueError(f"Sample spec #{idx} must include sample_id, clean_dir, and noise_dir")
        samples.append(dict(item))
    return samples


def candidate_from_metric_pair(
    *,
    source: str,
    base: str,
    clean_dir: Path,
    noise_dir: Path,
    official: Mapping[str, float],
    sparse: Mapping[str, float],
    note: str,
) -> Dict[str, object]:
    official_ate_delta = float(official["noise_ATE"]) - float(official["clean_ATE"])
    sparse_ate_delta = float(sparse["noise_ATE"]) - float(sparse["clean_ATE"])
    official_rpet_delta = float(official["noise_RPE trans"]) - float(official["clean_RPE trans"])
    sparse_rpet_delta = float(sparse["noise_RPE trans"]) - float(sparse["clean_RPE trans"])
    safe_source = str(source).replace("/", "_").replace(" ", "_")
    safe_base = str(base).replace("/", "_")
    return {
        "sample_id": f"{safe_source}_{safe_base}",
        "base_name": str(base),
        "clean_dir": str(clean_dir),
        "noise_dir": str(noise_dir),
        "source": str(source),
        "note": str(note),
        "selection_metrics": {
            "official_clean_ATE": float(official["clean_ATE"]),
            "official_noise_ATE": float(official["noise_ATE"]),
            "official_ATE_delta": official_ate_delta,
            "official_RPEt_delta": official_rpet_delta,
            "sparse_clean_ATE": float(sparse["clean_ATE"]),
            "sparse_noise_ATE": float(sparse["noise_ATE"]),
            "sparse_ATE_delta": sparse_ate_delta,
            "sparse_RPEt_delta": sparse_rpet_delta,
            "ATE_delta_gap_official_minus_sparse": official_ate_delta - sparse_ate_delta,
            "RPEt_delta_gap_official_minus_sparse": official_rpet_delta - sparse_rpet_delta,
        },
    }


def collect_old_waymo_candidates(args: argparse.Namespace) -> List[Dict[str, object]]:
    old_root = Path(args.old_data_root)
    result_root = Path(args.old_result_root)
    official = read_metric_pairs(result_root / "official_waymo" / "vkitti2" / "seq_metrics.csv")
    sparse = read_metric_pairs(result_root / "sparse67_waymo" / "vkitti2" / "seq_metrics.csv")
    candidates: List[Dict[str, object]] = []
    for base in sorted(set(official) & set(sparse)):
        clean_dir = old_root / f"{base}__clean"
        noise_dir = old_root / f"{base}__noise"
        if not clean_dir.is_dir() or not noise_dir.is_dir():
            continue
        candidates.append(
            candidate_from_metric_pair(
                source="old_cross_scene",
                base=base,
                clean_dir=clean_dir,
                noise_dir=noise_dir,
                official=official[base],
                sparse=sparse[base],
                note="old stride4 cross-scene plausible distractor candidate",
            )
        )
    return candidates


def collect_fixed_pair_candidates(args: argparse.Namespace) -> List[Dict[str, object]]:
    data_parent = Path(args.fixed_pair_data_parent)
    result_root = Path(args.fixed_result_root)
    candidates: List[Dict[str, object]] = []
    for pair_tag in parse_csv_list(args.fixed_pair_tags):
        data_root = data_parent / pair_tag / "waymo"
        official_csv = result_root / pair_tag / "official" / "vkitti2" / "seq_metrics.csv"
        sparse_csv = result_root / pair_tag / "sparse79" / "vkitti2" / "seq_metrics.csv"
        if not official_csv.is_file() or not sparse_csv.is_file():
            continue
        official = read_metric_pairs(official_csv)
        sparse = read_metric_pairs(sparse_csv)
        for base in sorted(set(official) & set(sparse)):
            clean_dir = data_root / f"{base}__clean"
            noise_dir = data_root / f"{base}__noise"
            if not clean_dir.is_dir() or not noise_dir.is_dir():
                continue
            candidates.append(
                candidate_from_metric_pair(
                    source=f"fixed_{pair_tag}",
                    base=base,
                    clean_dir=clean_dir,
                    noise_dir=noise_dir,
                    official=official[base],
                    sparse=sparse[base],
                    note=f"fixed same-scene camera-pair plausible distractor candidate: {pair_tag}",
                )
            )
    return candidates


def collect_old_waymo_data_candidates(args: argparse.Namespace) -> List[Dict[str, object]]:
    old_root = Path(args.old_data_root)
    candidates: List[Dict[str, object]] = []
    for clean_dir in sorted(old_root.glob("*__clean")):
        base = clean_dir.name[: -len("__clean")]
        noise_dir = old_root / f"{base}__noise"
        if not noise_dir.is_dir():
            continue
        candidates.append(
            candidate_from_dirs(
                source="old_cross_scene_data_score",
                base=base,
                clean_dir=clean_dir,
                noise_dir=noise_dir,
                note="old stride4 cross-scene data-score candidate",
            )
        )
    return candidates


def collect_fixed_pair_data_candidates(args: argparse.Namespace) -> List[Dict[str, object]]:
    data_parent = Path(args.fixed_pair_data_parent)
    candidates: List[Dict[str, object]] = []
    for pair_tag in parse_csv_list(args.fixed_pair_tags):
        data_root = data_parent / pair_tag / "waymo"
        for clean_dir in sorted(data_root.glob("*__clean")):
            base = clean_dir.name[: -len("__clean")]
            noise_dir = data_root / f"{base}__noise"
            if not noise_dir.is_dir():
                continue
            candidates.append(
                candidate_from_dirs(
                    source=f"fixed_{pair_tag}_data_score",
                    base=base,
                    clean_dir=clean_dir,
                    noise_dir=noise_dir,
                    note=f"fixed same-scene camera-pair data-score candidate: {pair_tag}",
                )
            )
    return candidates


def auto_select_samples_by_data_score(args: argparse.Namespace) -> List[Dict[str, object]]:
    sources = set(parse_csv_list(args.auto_select_sources))
    candidates: List[Dict[str, object]] = []
    if "old" in sources:
        candidates.extend(collect_old_waymo_data_candidates(args))
    if "fixed" in sources:
        candidates.extend(collect_fixed_pair_data_candidates(args))

    scored: List[Dict[str, object]] = []
    for candidate in candidates:
        try:
            data_scores = score_candidate_by_data(
                candidate,
                context_start=int(args.context_start),
                total_views=int(args.total_views),
            )
        except Exception as exc:
            skipped = dict(candidate)
            skipped["data_score_error"] = str(exc)
            continue
        if float(data_scores["noise_context_entropy"]) < float(args.data_score_min_noise_entropy):
            continue
        if float(data_scores["noise_context_edge_density"]) < float(args.data_score_min_noise_edge_density):
            continue
        item = dict(candidate)
        item["data_selection_scores"] = data_scores
        scored.append(item)

    scored.sort(
        key=lambda item: (
            bool((item.get("data_selection_scores", {}) or {}).get("resolution_match", False)),
            -float((item.get("data_selection_scores", {}) or {}).get("visual_distance", float("inf"))),
            float((item.get("data_selection_scores", {}) or {}).get("noise_context_entropy", 0.0)),
            float((item.get("data_selection_scores", {}) or {}).get("noise_context_edge_density", 0.0)),
        ),
        reverse=True,
    )
    setattr(args, "_data_score_ranked_samples", scored)
    top_k = int(args.auto_select_top_k)
    return scored[:top_k] if top_k > 0 else scored


def write_data_score_candidates_csv(path: Path, samples: Sequence[Mapping[str, object]]) -> None:
    rows: List[Dict[str, object]] = []
    for rank, sample in enumerate(samples):
        row: Dict[str, object] = {
            "rank": int(rank),
            "sample_id": sample.get("sample_id", ""),
            "base_name": sample.get("base_name", ""),
            "source": sample.get("source", ""),
            "clean_dir": sample.get("clean_dir", ""),
            "noise_dir": sample.get("noise_dir", ""),
        }
        scores = sample.get("data_selection_scores", {})
        if isinstance(scores, Mapping):
            for key, value in scores.items():
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, ensure_ascii=False)
                else:
                    row[key] = value
        rows.append(row)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rank_plausible_candidates(
    candidates: Sequence[Mapping[str, object]],
    *,
    min_official_ate_delta: float,
) -> List[Dict[str, object]]:
    filtered: List[Dict[str, object]] = []
    for item in candidates:
        metrics = item.get("selection_metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        if float(metrics.get("official_ATE_delta", 0.0)) < float(min_official_ate_delta):
            continue
        filtered.append(dict(item))
    filtered.sort(
        key=lambda item: (
            float((item.get("selection_metrics", {}) or {}).get("ATE_delta_gap_official_minus_sparse", 0.0)),
            float((item.get("selection_metrics", {}) or {}).get("official_ATE_delta", 0.0)),
        ),
        reverse=True,
    )
    return filtered


def auto_select_samples(args: argparse.Namespace) -> List[Dict[str, object]]:
    sources = set(parse_csv_list(args.auto_select_sources))
    candidates: List[Dict[str, object]] = []
    if "old" in sources:
        candidates.extend(collect_old_waymo_candidates(args))
    if "fixed" in sources:
        candidates.extend(collect_fixed_pair_candidates(args))
    ranked = rank_plausible_candidates(
        candidates,
        min_official_ate_delta=float(args.selection_min_official_ate_delta),
    )
    top_k = int(args.auto_select_top_k)
    return ranked[:top_k] if top_k > 0 else ranked


def score_old_waymo(args: argparse.Namespace, output_root: Path) -> Path:
    old_root = Path(args.old_data_root)
    result_root = Path(args.old_result_root)
    official = read_metric_pairs(result_root / "official_waymo" / "vkitti2" / "seq_metrics.csv")
    sparse = read_metric_pairs(result_root / "sparse67_waymo" / "vkitti2" / "seq_metrics.csv")
    rows: List[Dict[str, object]] = []
    for base in sorted(set(official) & set(sparse)):
        clean_dir = old_root / f"{base}__clean"
        noise_dir = old_root / f"{base}__noise"
        try:
            clean_images = image_paths(clean_dir, total_views=int(args.total_views))
            noise_images = image_paths(noise_dir, total_views=int(args.total_views))
        except FileNotFoundError:
            continue
        clean_context = mean_scores(clean_images[int(args.context_start) : int(args.total_views)])
        noise_context = mean_scores(noise_images[int(args.context_start) : int(args.total_views)])
        row: Dict[str, object] = {"base": base}
        for prefix, payload in (("official", official[base]), ("sparse67", sparse[base])):
            for key, value in payload.items():
                row[f"{prefix}_{key}"] = value
            row[f"{prefix}_ATE_delta"] = payload["noise_ATE"] - payload["clean_ATE"]
            row[f"{prefix}_RPEt_delta"] = payload["noise_RPE trans"] - payload["clean_RPE trans"]
            row[f"{prefix}_RPEr_delta"] = payload["noise_RPE rot"] - payload["clean_RPE rot"]
        row["ATE_delta_gap_official_minus_sparse"] = float(row["official_ATE_delta"]) - float(row["sparse67_ATE_delta"])
        for key, value in clean_context.items():
            row[f"clean_context_{key}"] = value
        for key, value in noise_context.items():
            row[f"noise_context_{key}"] = value
            row[f"noise_minus_clean_{key}"] = value - clean_context[key]
        rows.append(row)

    rows.sort(key=lambda item: float(item["ATE_delta_gap_official_minus_sparse"]), reverse=True)
    out_path = output_root / "old_waymo_context_structure_scores.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else default_output_root().resolve()
    dataset_root = output_root / "waymo"
    dataset_root.mkdir(parents=True, exist_ok=True)

    old_root = Path(args.old_data_root)
    fixed_root = Path(args.fixed_data_root)
    if args.sample_spec_json:
        samples = load_sample_spec_json(Path(args.sample_spec_json))
        selection_mode = "sample_spec_json"
    elif bool(args.auto_select_by_data_score):
        samples = auto_select_samples_by_data_score(args)
        selection_mode = "auto_select_by_data_score"
    elif int(args.auto_select_top_k) > 0:
        samples = auto_select_samples(args)
        selection_mode = "auto_select_top_k"
    else:
        samples = []
        for sample in DIAGNOSTIC_SAMPLES:
            item = dict(sample)
            if str(item["sample_id"]).startswith("old_"):
                base = str(item["base_name"])
                item["clean_dir"] = old_root / f"{base}__clean"
                item["noise_dir"] = old_root / f"{base}__noise"
            elif str(item["sample_id"]).startswith("fixed_"):
                base = str(item["base_name"])
                item["clean_dir"] = fixed_root / f"{base}__clean"
                item["noise_dir"] = fixed_root / f"{base}__noise"
            samples.append(item)
        selection_mode = "hardcoded_diagnostic_samples"

    if not samples:
        raise RuntimeError("No diagnostic samples selected.")

    rows: List[Dict[str, object]] = []
    for sample in samples:
        rows.extend(
            build_variants_for_sample(
                sample,
                dataset_root=dataset_root,
                context_start=int(args.context_start),
                total_views=int(args.total_views),
                copy_images=bool(args.copy_images),
                blur_radius=float(args.blur_radius),
            )
        )

    index_path = output_root / "variant_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seq", "sample_id", "base_name", "variant", "out_dir"])
        writer.writeheader()
        writer.writerows(rows)

    data_score_candidates_path = None
    ranked_data_score_samples = getattr(args, "_data_score_ranked_samples", None)
    if ranked_data_score_samples is not None:
        data_score_candidates_path = output_root / "data_score_candidates.csv"
        write_data_score_candidates_csv(data_score_candidates_path, ranked_data_score_samples)

    score_path = None
    if args.score_old_waymo:
        score_path = score_old_waymo(args, output_root=output_root)

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "waymo_plausible_context_diagnostic_v1",
        "output_root": str(output_root),
        "dataset_root": str(dataset_root),
        "num_variants": len(rows),
        "num_base_samples": len(samples),
        "selection_mode": selection_mode,
        "selected_samples": samples,
        "variant_index": str(index_path),
        "data_score_candidates_csv": str(data_score_candidates_path) if data_score_candidates_path else "",
        "old_waymo_context_score_csv": str(score_path) if score_path else "",
        "variants": list(VARIANTS),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
