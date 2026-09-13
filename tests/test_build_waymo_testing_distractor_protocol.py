#!/usr/bin/env python3
"""Tests for the Waymo2 testing distractor-pair builder."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for candidate in (REPO_ROOT / "tools", THIS_DIR.parent / "tools"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

import build_waymo_testing_distractor_protocol as builder


def write_sequence(root: Path, name: str, *, base_value: int, count: int = 64) -> Path:
    sequence = root / name
    color = sequence / "color"
    pose = sequence / "pose"
    color.mkdir(parents=True)
    pose.mkdir(parents=True)
    yy, xx = np.mgrid[:32, :64]
    for index in range(count):
        pattern = (base_value + xx + 3 * yy + index) % 255
        rgb = np.stack((pattern, (pattern + 17) % 255, (pattern + 31) % 255), axis=-1).astype(np.uint8)
        Image.fromarray(rgb).save(color / f"{index:03d}.jpg")
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 3] = float(index)
        np.savetxt(pose / f"{index:03d}.txt", matrix)
    return sequence


def test_centered_window_uses_common_color_pose_indices() -> None:
    with TemporaryDirectory() as tmpdir:
        sequence = write_sequence(Path(tmpdir), "segment_a_cam1", base_value=20)
        payload = builder.build_sequence_payload(sequence, frame_stride=4)

    assert payload.segment == "segment_a"
    assert payload.camera == "1"
    assert len(payload.frame_indices) == 10
    assert all(b - a == 4 for a, b in zip(payload.frame_indices, payload.frame_indices[1:]))
    assert len(payload.pose_rows) == 10
    assert all(len(row.split()) == 16 for row in payload.pose_rows)


def test_evenly_spaced_windows_return_requested_temporal_anchors() -> None:
    windows = builder.evenly_spaced_stride_windows(
        list(range(100)),
        total_views=10,
        frame_stride=4,
        windows_per_sequence=4,
    )
    assert len(windows) == 4
    assert len({window[0] for window in windows}) == 4
    assert all(len(window) == 10 for window in windows)
    assert all(all(b - a == 4 for a, b in zip(window, window[1:])) for window in windows)


def synthetic_payload(
    root: Path,
    *,
    segment: str,
    camera: str,
    prefix_feature: tuple[float, float],
    tail_feature: tuple[float, float],
) -> builder.SequencePayload:
    return builder.SequencePayload(
        path=root / f"{segment}_cam{camera}",
        segment=segment,
        camera=camera,
        frame_indices=tuple(range(10)),
        image_paths=tuple(root / f"frame_{index:04d}.jpg" for index in range(10)),
        pose_rows=tuple(" ".join(["1"] * 16) for _ in range(10)),
        prefix_feature=np.asarray(prefix_feature, dtype=np.float32),
        tail_feature=np.asarray(tail_feature, dtype=np.float32),
        prefix_entropy=5.0,
        prefix_edge_density=0.1,
        tail_entropy=5.0,
        tail_edge_density=0.1,
        resolution=(64, 32),
    )


def test_selection_requires_wrong_tail_closer_than_clean_context_and_allows_cross_camera() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        clean = synthetic_payload(
            root,
            segment="segment_a",
            camera="1",
            prefix_feature=(0.0, 0.0),
            tail_feature=(1.0, 1.0),
        )
        valid_cross_camera = synthetic_payload(
            root,
            segment="segment_b",
            camera="2",
            prefix_feature=(9.0, 9.0),
            tail_feature=(0.5, 0.5),
        )
        invalid_same_camera = synthetic_payload(
            root,
            segment="segment_c",
            camera="1",
            prefix_feature=(9.0, 9.0),
            tail_feature=(2.0, 2.0),
        )
        weak_margin = synthetic_payload(
            root,
            segment="segment_d",
            camera="1",
            prefix_feature=(9.0, 9.0),
            tail_feature=(0.9, 0.9),
        )
        selected, candidates = builder.select_distractor_tail(
            clean,
            [clean, invalid_same_camera, weak_margin, valid_cross_camera],
            min_entropy=0.0,
            min_edge_density=0.0,
            min_similarity_margin=0.5,
        )

    assert selected.wrong.segment == "segment_b"
    assert selected.wrong.camera == "2"
    assert selected.clean_distance > selected.feature_distance
    assert selected.wrong_more_like_prefix_margin > 0.0
    assert [row.wrong.segment for row in candidates] == ["segment_b"]


def test_selection_rejects_pool_without_positive_similarity_margin() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        clean = synthetic_payload(
            root,
            segment="segment_a",
            camera="1",
            prefix_feature=(0.0, 0.0),
            tail_feature=(1.0, 1.0),
        )
        invalid = synthetic_payload(
            root,
            segment="segment_b",
            camera="1",
            prefix_feature=(9.0, 9.0),
            tail_feature=(2.0, 2.0),
        )
        try:
            builder.select_distractor_tail(
                clean,
                [clean, invalid],
                min_entropy=0.0,
                min_edge_density=0.0,
            )
        except RuntimeError as exc:
            assert "positive similarity margin" in str(exc)
        else:
            raise AssertionError("Expected hard positive-margin rejection")


def test_materialization_builds_nested_five_level_protocol() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        clean = builder.build_sequence_payload(write_sequence(root, "segment_a_cam1", base_value=20), frame_stride=4)
        wrong = builder.build_sequence_payload(write_sequence(root, "segment_b_cam1", base_value=22), frame_stride=4)
        pair = builder.CandidatePair(
            clean=clean,
            wrong=wrong,
            feature_distance=0.1,
            clean_distance=0.2,
            wrong_more_like_prefix_margin=0.1,
            wrong_entropy=wrong.tail_entropy,
            wrong_edge_density=wrong.tail_edge_density,
            plausibility_score=1.0,
            selection_score=1.0,
        )
        output = root / "output"
        builder.materialize_pair(pair, output, selection_rank=0)

        prefix_payloads = []
        for count in range(5):
            variant = f"noise{count}"
            sequence = output / variant / f"testing_rank000_{pair.sample_id}__{variant}"
            images = sorted((sequence / "color_90").glob("*.jpg"))
            pose_rows = [line for line in (sequence / "pose_90.txt").read_text().splitlines() if line.strip()]
            meta = json.loads((sequence / "tuple_meta.json").read_text())
            assert len(images) == 10
            assert len(pose_rows) == 10
            assert meta["eval_frame_indices"] == [0, 1, 2, 3, 4, 5]
            assert meta["source_labels"] == ["clean"] * (10 - count) + ["distractor"] * count
            prefix_payloads.append([path.read_bytes() for path in images[:6]])

    assert all(payload == prefix_payloads[0] for payload in prefix_payloads[1:])


if __name__ == "__main__":
    test_centered_window_uses_common_color_pose_indices()
    test_evenly_spaced_windows_return_requested_temporal_anchors()
    test_selection_requires_wrong_tail_closer_than_clean_context_and_allows_cross_camera()
    test_selection_rejects_pool_without_positive_similarity_margin()
    test_materialization_builds_nested_five_level_protocol()
    print("ok")
