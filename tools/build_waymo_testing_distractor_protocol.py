#!/usr/bin/env python3
"""Build five-level distractor-count tuples from Waymo2 testing sequences."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from functools import lru_cache
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


TOTAL_VIEWS = 10
PREFIX_VIEWS = 6
CONTEXT_VIEWS = 4
VARIANTS = tuple(f"noise{count}" for count in range(CONTEXT_VIEWS + 1))


@dataclass(frozen=True)
class SequencePayload:
    path: Path
    segment: str
    camera: str
    frame_indices: tuple[int, ...]
    image_paths: tuple[Path, ...]
    pose_rows: tuple[str, ...]
    prefix_feature: np.ndarray
    tail_feature: np.ndarray
    prefix_entropy: float
    prefix_edge_density: float
    tail_entropy: float
    tail_edge_density: float
    resolution: tuple[int, int]


@dataclass(frozen=True)
class CandidatePair:
    clean: SequencePayload
    wrong: SequencePayload
    feature_distance: float
    clean_distance: float
    wrong_more_like_prefix_margin: float
    wrong_entropy: float
    wrong_edge_density: float
    plausibility_score: float
    selection_score: float = 0.0

    @property
    def sample_id(self) -> str:
        return (
            f"{self.clean.segment}_cam{self.clean.camera}_a{self.clean.frame_indices[0]:03d}"
            f"__wrong_{self.wrong.segment}_cam{self.wrong.camera}_t{self.wrong.frame_indices[PREFIX_VIEWS]:03d}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waymo-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--selected-count", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--windows-per-sequence", type=int, default=1)
    parser.add_argument("--min-entropy", type=float, default=4.0)
    parser.add_argument("--min-edge-density", type=float, default=0.02)
    parser.add_argument("--min-similarity-margin", type=float, default=0.0)
    return parser.parse_args()


def parse_sequence_name(path: str | Path) -> tuple[str, str]:
    name = Path(path).name
    if "_cam" not in name:
        raise ValueError(f"Waymo sequence name lacks camera suffix: {name}")
    segment, camera = name.rsplit("_cam", 1)
    if not segment or not camera:
        raise ValueError(f"Invalid Waymo sequence name: {name}")
    return segment, camera


def indexed_files(directory: Path) -> dict[int, Path]:
    rows: dict[int, Path] = {}
    for path in sorted(item for item in directory.iterdir() if item.is_file()):
        if path.stem.isdigit():
            rows[int(path.stem)] = path
    return rows


def common_frame_indices(sequence_dir: Path) -> list[int]:
    colors = indexed_files(sequence_dir / "color")
    poses = indexed_files(sequence_dir / "pose")
    return sorted(set(colors) & set(poses))


def valid_stride_windows(
    indices: Sequence[int], *, total_views: int = TOTAL_VIEWS, frame_stride: int = 4
) -> list[tuple[int, ...]]:
    available = set(int(index) for index in indices)
    starts = [
        start
        for start in sorted(available)
        if all(start + offset * frame_stride in available for offset in range(total_views))
    ]
    if not starts:
        raise ValueError(
            f"No {total_views}-view stride-{frame_stride} window in {len(available)} common frames"
        )
    return [tuple(start + offset * frame_stride for offset in range(total_views)) for start in starts]


def centered_stride_window(
    indices: Sequence[int], *, total_views: int = TOTAL_VIEWS, frame_stride: int = 4
) -> tuple[int, ...]:
    windows = valid_stride_windows(indices, total_views=total_views, frame_stride=frame_stride)
    return windows[len(windows) // 2]


def evenly_spaced_stride_windows(
    indices: Sequence[int],
    *,
    total_views: int = TOTAL_VIEWS,
    frame_stride: int = 4,
    windows_per_sequence: int = 1,
) -> list[tuple[int, ...]]:
    windows = valid_stride_windows(indices, total_views=total_views, frame_stride=frame_stride)
    if windows_per_sequence <= 0 or windows_per_sequence >= len(windows):
        return windows
    positions = np.linspace(0, len(windows) - 1, num=windows_per_sequence)
    selected_indices = sorted({int(round(position)) for position in positions})
    return [windows[index] for index in selected_indices]


@lru_cache(maxsize=None)
def image_feature(path: Path, size: tuple[int, int] = (32, 18)) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    thumbnail = arr.reshape(-1)
    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(arr[..., channel], bins=16, range=(0.0, 1.0))
        values = hist.astype(np.float32)
        values /= max(float(values.sum()), 1.0)
        histograms.append(values)
    feature = np.concatenate([thumbnail, *histograms]).astype(np.float32)
    feature -= float(feature.mean())
    norm = float(np.linalg.norm(feature))
    if norm > 1.0e-6:
        feature /= norm
    return feature


def mean_feature(paths: Sequence[Path]) -> np.ndarray:
    return np.mean(np.stack([image_feature(path) for path in paths]), axis=0)


@lru_cache(maxsize=None)
def image_structure_score(path: Path) -> tuple[float, float]:
    with Image.open(path) as image:
        gray = image.convert("L")
        max_side = max(gray.size)
        if max_side > 384:
            scale = 384.0 / float(max_side)
            gray = gray.resize((max(1, int(gray.width * scale)), max(1, int(gray.height * scale))))
        arr = np.asarray(gray, dtype=np.float32) / 255.0
    dx = np.diff(arr, axis=1, prepend=arr[:, :1])
    dy = np.diff(arr, axis=0, prepend=arr[:1, :])
    grad = np.sqrt(dx * dx + dy * dy)
    hist, _ = np.histogram(arr, bins=64, range=(0.0, 1.0))
    probability = hist.astype(np.float64) / max(float(hist.sum()), 1.0)
    entropy = -float(np.sum(probability * np.log2(probability + 1.0e-12)))
    edge_density = float(np.mean(grad > 0.06))
    return entropy, edge_density


def mean_structure_score(paths: Sequence[Path]) -> tuple[float, float]:
    rows = [image_structure_score(path) for path in paths]
    return float(np.mean([row[0] for row in rows])), float(np.mean([row[1] for row in rows]))


def pose_row(path: Path) -> str:
    matrix = np.asarray(np.loadtxt(path), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"Expected finite 4x4 pose at {path}, got {matrix.shape}")
    return " ".join(f"{float(value):.12e}" for value in matrix.reshape(-1))


def build_sequence_payload_for_window(
    sequence_dir: str | Path, frame_indices: Sequence[int]
) -> SequencePayload:
    path = Path(sequence_dir).expanduser().resolve()
    segment, camera = parse_sequence_name(path)
    colors = indexed_files(path / "color")
    poses = indexed_files(path / "pose")
    frame_indices = tuple(int(index) for index in frame_indices)
    image_paths = tuple(colors[index] for index in frame_indices)
    pose_rows = tuple(pose_row(poses[index]) for index in frame_indices)
    resolutions = set()
    for image_path in image_paths:
        with Image.open(image_path) as image:
            resolutions.add((int(image.width), int(image.height)))
    if len(resolutions) != 1:
        raise ValueError(f"Mixed image resolutions in {path}: {sorted(resolutions)}")
    prefix_entropy, prefix_edge_density = mean_structure_score(image_paths[:PREFIX_VIEWS])
    tail_entropy, tail_edge_density = mean_structure_score(image_paths[PREFIX_VIEWS:])
    return SequencePayload(
        path=path,
        segment=segment,
        camera=camera,
        frame_indices=frame_indices,
        image_paths=image_paths,
        pose_rows=pose_rows,
        prefix_feature=mean_feature(image_paths[:PREFIX_VIEWS]),
        tail_feature=mean_feature(image_paths[PREFIX_VIEWS:]),
        prefix_entropy=prefix_entropy,
        prefix_edge_density=prefix_edge_density,
        tail_entropy=tail_entropy,
        tail_edge_density=tail_edge_density,
        resolution=next(iter(resolutions)),
    )


def build_sequence_payload(sequence_dir: str | Path, *, frame_stride: int = 4) -> SequencePayload:
    path = Path(sequence_dir).expanduser().resolve()
    frame_indices = centered_stride_window(
        common_frame_indices(path), total_views=TOTAL_VIEWS, frame_stride=int(frame_stride)
    )
    return build_sequence_payload_for_window(path, frame_indices)


def build_sequence_payloads(
    sequence_dir: str | Path, *, frame_stride: int = 4, windows_per_sequence: int = 1
) -> list[SequencePayload]:
    path = Path(sequence_dir).expanduser().resolve()
    windows = evenly_spaced_stride_windows(
        common_frame_indices(path),
        total_views=TOTAL_VIEWS,
        frame_stride=int(frame_stride),
        windows_per_sequence=int(windows_per_sequence),
    )
    return [build_sequence_payload_for_window(path, window) for window in windows]


def feature_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(lhs, dtype=np.float32) - np.asarray(rhs, dtype=np.float32))))


def eligible_distractor_pairs(
    clean: SequencePayload,
    payloads: Sequence[SequencePayload],
    *,
    min_entropy: float = 4.0,
    min_edge_density: float = 0.02,
    min_similarity_margin: float = 0.0,
) -> list[CandidatePair]:
    candidates: list[CandidatePair] = []
    clean_distance = feature_distance(clean.prefix_feature, clean.tail_feature)
    for wrong in payloads:
        if wrong.segment == clean.segment:
            continue
        if wrong.resolution != clean.resolution:
            continue
        if wrong.tail_entropy < float(min_entropy) or wrong.tail_edge_density < float(min_edge_density):
            continue
        distance = feature_distance(clean.prefix_feature, wrong.tail_feature)
        margin = clean_distance - distance
        if margin < float(min_similarity_margin) or margin <= 0.0:
            continue
        structure_bonus = 0.01 * wrong.tail_entropy + 0.1 * wrong.tail_edge_density
        candidates.append(
            CandidatePair(
                clean=clean,
                wrong=wrong,
                feature_distance=distance,
                clean_distance=clean_distance,
                wrong_more_like_prefix_margin=margin,
                wrong_entropy=wrong.tail_entropy,
                wrong_edge_density=wrong.tail_edge_density,
                plausibility_score=margin - distance + structure_bonus,
            )
        )
    candidates.sort(key=lambda row: (row.feature_distance, row.wrong.path.name))
    return candidates


def select_distractor_tail(
    clean: SequencePayload,
    payloads: Sequence[SequencePayload],
    *,
    min_entropy: float = 4.0,
    min_edge_density: float = 0.02,
    min_similarity_margin: float = 0.0,
) -> tuple[CandidatePair, list[CandidatePair]]:
    candidates = eligible_distractor_pairs(
        clean,
        payloads,
        min_entropy=min_entropy,
        min_edge_density=min_edge_density,
        min_similarity_margin=min_similarity_margin,
    )
    if not candidates:
        raise RuntimeError(f"No cross-segment distractor with positive similarity margin for {clean.path.name}")
    return candidates[0], candidates


def zscore(values: Sequence[float], value: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    sigma = float(np.std(array))
    if sigma < 1.0e-12:
        return 0.0
    return float((float(value) - float(np.mean(array))) / sigma)


def pair_score_fields(pair: CandidatePair) -> dict[str, float]:
    clean = pair.clean
    wrong = pair.wrong
    return {
        "prefix_minus_clean_entropy": clean.prefix_entropy - clean.tail_entropy,
        "prefix_minus_clean_edge_density": clean.prefix_edge_density - clean.tail_edge_density,
        "wrong_minus_clean_entropy": wrong.tail_entropy - clean.tail_entropy,
        "wrong_minus_clean_edge_density": wrong.tail_edge_density - clean.tail_edge_density,
        "wrong_tail_entropy": wrong.tail_entropy,
        "wrong_tail_edge_density": wrong.tail_edge_density,
        "wrong_more_like_prefix_margin": pair.wrong_more_like_prefix_margin,
        "prefix_wrong_distance": pair.feature_distance,
    }


def apply_historical_composite_score(pairs: Sequence[CandidatePair]) -> list[CandidatePair]:
    if not pairs:
        return []
    fields = [pair_score_fields(pair) for pair in pairs]
    keys = tuple(fields[0])
    values = {key: [row[key] for row in fields] for key in keys}
    output: list[CandidatePair] = []
    for pair, row in zip(pairs, fields):
        score = (
            1.0 * zscore(values["prefix_minus_clean_entropy"], row["prefix_minus_clean_entropy"])
            + 0.6 * zscore(values["prefix_minus_clean_edge_density"], row["prefix_minus_clean_edge_density"])
            + 1.0 * zscore(values["wrong_minus_clean_entropy"], row["wrong_minus_clean_entropy"])
            + 0.6 * zscore(values["wrong_minus_clean_edge_density"], row["wrong_minus_clean_edge_density"])
            + 0.4 * zscore(values["wrong_tail_entropy"], row["wrong_tail_entropy"])
            + 0.4 * zscore(values["wrong_tail_edge_density"], row["wrong_tail_edge_density"])
            + 0.4 * zscore(values["wrong_more_like_prefix_margin"], row["wrong_more_like_prefix_margin"])
            - 0.2 * zscore(values["prefix_wrong_distance"], row["prefix_wrong_distance"])
        )
        output.append(replace(pair, selection_score=float(score)))
    output.sort(key=lambda pair: (-pair.selection_score, pair.sample_id))
    return output


def link_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(source), str(destination))


def materialize_pair(pair: CandidatePair, output_root: Path, *, selection_rank: int) -> dict[str, Any]:
    output = Path(output_root)
    base_name = f"testing_rank{selection_rank:03d}_{pair.sample_id}"
    variant_rows: list[dict[str, Any]] = []
    for count, variant in enumerate(VARIANTS):
        sequence = output / variant / f"{base_name}__{variant}"
        if sequence.exists():
            raise FileExistsError(f"Refusing to overwrite {sequence}")
        color_dir = sequence / "color_90"
        color_dir.mkdir(parents=True)
        switch_index = TOTAL_VIEWS - count
        labels: list[str] = []
        output_pose_rows: list[str] = []
        for index in range(TOTAL_VIEWS):
            use_wrong = index >= switch_index
            source = pair.wrong if use_wrong else pair.clean
            source_path = source.image_paths[index]
            link_image(source_path, color_dir / f"frame_{index:04d}{source_path.suffix.lower()}")
            output_pose_rows.append(source.pose_rows[index])
            labels.append("distractor" if use_wrong else "clean")
        (sequence / "pose_90.txt").write_text("\n".join(output_pose_rows) + "\n", encoding="utf-8")
        meta = {
            "protocol": "waymo_testing_expanded_distractor_count_v1",
            "sample_id": base_name,
            "variant": variant,
            "distractor_count": count,
            "eval_frame_indices": list(range(PREFIX_VIEWS)),
            "source_labels": labels,
            "prefix_scene": pair.clean.segment,
            "prefix_camera": pair.clean.camera,
            "prefix_frame_indices": list(pair.clean.frame_indices[:PREFIX_VIEWS]),
            "clean_context_frame_indices": list(pair.clean.frame_indices[PREFIX_VIEWS:]),
            "distractor_scene": pair.wrong.segment,
            "distractor_camera": pair.wrong.camera,
            "distractor_frame_indices": list(pair.wrong.frame_indices[PREFIX_VIEWS:]),
            "clean_distance": pair.clean_distance,
            "feature_distance": pair.feature_distance,
            "wrong_more_like_prefix_margin": pair.wrong_more_like_prefix_margin,
            "plausibility_score": pair.plausibility_score,
            "selection_score": pair.selection_score,
            "selection_uses_model_outputs": False,
        }
        (sequence / "tuple_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        variant_rows.append({"variant": variant, "sequence_dir": str(sequence)})
    return {"selection_rank": selection_rank, "sample_id": base_name, "variants": variant_rows}


def candidate_row(pair: CandidatePair, *, clean_rank: int, selected: bool) -> dict[str, Any]:
    return {
        "clean_rank": clean_rank,
        "selected": bool(selected),
        "sample_id": pair.sample_id,
        "clean_sequence": pair.clean.path.name,
        "clean_segment": pair.clean.segment,
        "camera": pair.clean.camera,
        "clean_frame_indices": list(pair.clean.frame_indices),
        "wrong_sequence": pair.wrong.path.name,
        "wrong_segment": pair.wrong.segment,
        "wrong_frame_indices": list(pair.wrong.frame_indices),
        "clean_distance": pair.clean_distance,
        "feature_distance": pair.feature_distance,
        "wrong_more_like_prefix_margin": pair.wrong_more_like_prefix_margin,
        "wrong_entropy": pair.wrong_entropy,
        "wrong_edge_density": pair.wrong_edge_density,
        "plausibility_score": pair.plausibility_score,
        "selection_score": pair.selection_score,
    }


def build(
    waymo_root: Path,
    output_root: Path,
    *,
    selected_count: int,
    frame_stride: int,
    windows_per_sequence: int,
    min_entropy: float,
    min_edge_density: float,
    min_similarity_margin: float,
) -> dict[str, Any]:
    sequence_dirs = sorted(path for path in waymo_root.iterdir() if path.is_dir())
    payloads = [
        payload
        for path in sequence_dirs
        for payload in build_sequence_payloads(
            path,
            frame_stride=frame_stride,
            windows_per_sequence=windows_per_sequence,
        )
    ]
    eligible_pairs: list[CandidatePair] = []
    for clean in payloads:
        eligible_pairs.extend(
            eligible_distractor_pairs(
                clean,
                payloads,
                min_entropy=min_entropy,
                min_edge_density=min_edge_density,
                min_similarity_margin=min_similarity_margin,
            )
        )
    scored_pairs = apply_historical_composite_score(eligible_pairs)
    selected: list[CandidatePair] = []
    used_clean_anchors: set[tuple[Path, int]] = set()
    for pair in scored_pairs:
        clean_anchor = (pair.clean.path, int(pair.clean.frame_indices[0]))
        if clean_anchor in used_clean_anchors:
            continue
        selected.append(pair)
        used_clean_anchors.add(clean_anchor)
        if len(selected) >= selected_count:
            break
    if selected_count <= 0 or len(selected) < selected_count:
        raise ValueError(
            f"Need {selected_count} unique clean anchors, but only {len(selected)} satisfy the positive-margin rule"
        )
    selected_root = output_root / f"selected{selected_count}"
    materialized = [
        materialize_pair(pair, selected_root, selection_rank=rank)
        for rank, pair in enumerate(selected)
    ]
    selected_ids = {pair.sample_id for pair in selected}
    rows = [
        candidate_row(pair, clean_rank=rank, selected=pair.sample_id in selected_ids)
        for rank, pair in enumerate(scored_pairs)
    ]
    payload = {
        "protocol": "waymo_testing_expanded_distractor_count_v1",
        "waymo_root": str(waymo_root),
        "output_root": str(output_root),
        "num_sequence_anchors": len(payloads),
        "num_candidate_pairs": len(scored_pairs),
        "num_unique_eligible_clean_anchors": len(
            {(pair.clean.path, int(pair.clean.frame_indices[0])) for pair in scored_pairs}
        ),
        "selected_count": selected_count,
        "frame_stride": frame_stride,
        "windows_per_sequence": windows_per_sequence,
        "min_entropy": min_entropy,
        "min_edge_density": min_edge_density,
        "min_similarity_margin": min_similarity_margin,
        "hard_similarity_rule": "prefix_wrong_distance < prefix_clean_distance",
        "camera_constraint": "resolution_match_only",
        "ranking": "historical_composite_score",
        "selection_uses_model_outputs": False,
        "pairs": rows,
        "materialized": materialized,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "selection_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    args = parse_args()
    payload = build(
        Path(args.waymo_root).expanduser().resolve(),
        Path(args.output_root).expanduser().resolve(),
        selected_count=int(args.selected_count),
        frame_stride=int(args.frame_stride),
        windows_per_sequence=int(args.windows_per_sequence),
        min_entropy=float(args.min_entropy),
        min_edge_density=float(args.min_edge_density),
        min_similarity_margin=float(args.min_similarity_margin),
    )
    print(
        f"[waymo-testing-distractor] anchors={payload['num_sequence_anchors']} "
        f"pairs={payload['num_candidate_pairs']} selected={payload['selected_count']}"
    )


if __name__ == "__main__":
    main()
