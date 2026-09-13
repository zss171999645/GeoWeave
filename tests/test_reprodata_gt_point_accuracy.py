#!/usr/bin/env python3
"""Tests for GT point accuracy helpers on reproduced rebuttal data."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
for candidate in (THIS_DIR, THIS_DIR.parent / "tools"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

import reprodata_gt_point_accuracy as gtacc


def test_scale_intrinsic_to_target_size() -> None:
    k = np.asarray([[100.0, 0.0, 50.0], [0.0, 200.0, 80.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    scaled = gtacc.scale_intrinsic(k, source_hw=(100, 200), target_hw=(50, 100))

    np.testing.assert_allclose(scaled, [[50.0, 0.0, 25.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])


def test_scannet_normalized_intrinsic_uses_target_pixels() -> None:
    normalized = np.asarray([[0.75, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)

    pixels = gtacc.normalized_intrinsic_to_pixels(normalized, target_hw=(200, 300))

    np.testing.assert_allclose(pixels, [[225.0, 0.0, 150.0], [0.0, 200.0, 100.0], [0.0, 0.0, 1.0]])


def test_load_scannet_intrinsics_accepts_list_literal_rows(tmp_path: Path) -> None:
    path = tmp_path / "scene_intrinsic.txt"
    path.write_text(
        "[[0.75, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]\n"
        "[[0.5, 0.0, 0.25], [0.0, 0.6, 0.25], [0.0, 0.0, 1.0]]\n",
        encoding="utf-8",
    )

    intrinsics = gtacc.load_scannet_normalized_intrinsics(path)

    assert intrinsics.shape == (2, 3, 3)
    np.testing.assert_allclose(intrinsics[1], [[0.5, 0.0, 0.25], [0.0, 0.6, 0.25], [0.0, 0.0, 1.0]])


def test_depth_to_world_points_uses_c2w_pose() -> None:
    depth = np.ones((2, 2), dtype=np.float32)
    intrinsic = np.eye(3, dtype=np.float64)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 3] = np.asarray([10.0, 20.0, 30.0])

    points = gtacc.depth_to_world_points(depth, intrinsic, c2w)

    np.testing.assert_allclose(points[0, 0], [10.0, 20.0, 31.0])
    np.testing.assert_allclose(points[1, 1], [11.0, 21.0, 31.0])


def test_waymo_manifest_maps_short_scene_and_camera_to_copied_dir(tmp_path: Path) -> None:
    gt_root = tmp_path / "gt"
    manifest = {
        "waymo_required_pairs": [
            {
                "short_scene": "163453191",
                "camN": 4,
                "frames": [18],
                "source_dir": "/mnt/bos/Waymo2/training/16345319168590318167_1420_000_1440_000_cam4",
            }
        ]
    }
    gt_root.mkdir()
    (gt_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    index = gtacc.load_waymo_source_index(gt_root)
    source = index.resolve(short_scene="163453191", cam_zero_based="03")

    assert source == gt_root / "Waymo2/training/16345319168590318167_1420_000_1440_000_cam4"


def test_eval_indices_prefix_or_all() -> None:
    assert gtacc.resolve_eval_indices(prefix_size=6, total_views=10, mode="prefix") == [0, 1, 2, 3, 4, 5]
    assert gtacc.resolve_eval_indices(prefix_size=6, total_views=10, mode="all") == list(range(10))


def test_flatten_metric_points_samples_sparse_valid_pixels_before_stride() -> None:
    points = np.zeros((1, 3, 3, 3), dtype=np.float32)
    points[0, 1, 1] = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    valid = np.zeros((1, 3, 3), dtype=bool)
    valid[0, 1, 1] = True

    flattened = gtacc.flatten_metric_points(points, valid=valid, stride=2, max_points=100)

    np.testing.assert_allclose(flattened, [[1.0, 2.0, 3.0]])


def test_summarize_rows_includes_normal_metrics() -> None:
    rows = [
        {
            "sample_id": "sample",
            "model": "geoweave_pi3",
            "variant": "weak_overlap",
            "status": "ok",
            "nc_acc_mean": 0.8,
            "nc_comp_mean": 0.9,
            "nc_mean": 0.85,
        }
    ]

    summary = gtacc.summarize_rows(rows)

    aggregate = summary["aggregate"][0]
    assert aggregate["nc_acc_mean"] == 0.8
    assert aggregate["nc_comp_mean"] == 0.9
    assert aggregate["nc_mean"] == 0.85


def test_world_points_to_camera_depth_uses_c2w_pose() -> None:
    points = np.asarray([[[[10.0, 20.0, 31.0], [11.0, 20.0, 32.0]]]], dtype=np.float64)
    c2w = np.eye(4, dtype=np.float64)[None]
    c2w[0, :3, 3] = np.asarray([10.0, 20.0, 30.0])

    depth = gtacc.world_points_to_camera_depth(points, c2w)

    np.testing.assert_allclose(depth, [[[1.0, 2.0]]])


def test_compute_aligned_depth_metrics_recovers_known_sim3() -> None:
    gt = np.asarray(
        [
            [
                [[-1.0, -1.0, 2.0], [0.0, -1.0, 2.5], [1.0, -1.0, 3.0]],
                [[-1.0, 0.0, 2.2], [0.0, 0.0, 3.2], [1.0, 0.0, 4.0]],
                [[-1.0, 1.0, 2.7], [0.0, 1.0, 3.5], [1.0, 1.0, 4.5]],
            ]
        ],
        dtype=np.float64,
    )
    translation = np.asarray([0.4, -0.7, 1.3], dtype=np.float64)
    pred = (gt - translation) / 1.7
    valid = np.ones(gt.shape[:-1], dtype=bool)
    c2w = np.eye(4, dtype=np.float64)[None]

    metrics = gtacc.compute_aligned_depth_metrics(
        pred,
        gt,
        gt_c2w=c2w,
        valid=valid,
        metric_stride=1,
        max_metric_points=100,
    )

    assert metrics["depth_status"] == "ok"
    assert metrics["depth_abs_rel"] < 1.0e-6
    assert metrics["depth_rmse"] < 1.0e-6
    assert metrics["depth_delta1"] == 1.0


if __name__ == "__main__":
    test_scale_intrinsic_to_target_size()
    test_scannet_normalized_intrinsic_uses_target_pixels()
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        test_load_scannet_intrinsics_accepts_list_literal_rows(Path(tmp))
    test_depth_to_world_points_uses_c2w_pose()

    with TemporaryDirectory() as tmp:
        test_waymo_manifest_maps_short_scene_and_camera_to_copied_dir(Path(tmp))
    test_eval_indices_prefix_or_all()
    test_flatten_metric_points_samples_sparse_valid_pixels_before_stride()
    test_summarize_rows_includes_normal_metrics()
    test_world_points_to_camera_depth_uses_c2w_pose()
    test_compute_aligned_depth_metrics_recovers_known_sim3()
    print("ok")
