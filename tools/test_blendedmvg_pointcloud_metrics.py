#!/usr/bin/env python3
"""Small behavior tests for BlendedMVG point-cloud metric helpers."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from blendedmvg_pointcloud_metrics import compute_aligned_cloud_metrics
from blendedmvg_point_degradation_protocol import sim3_error


def test_sim3_aligned_accuracy_completion_are_near_zero_for_transformed_cloud() -> None:
    xs, ys = np.meshgrid(np.linspace(-1.0, 1.0, 9), np.linspace(-0.5, 0.5, 7), indexing="xy")
    gt = np.stack([xs.reshape(-1), ys.reshape(-1), 0.2 * xs.reshape(-1) + 0.1], axis=1)

    theta = math.radians(25.0)
    rot = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pred = (gt @ rot.T) * 1.7 + np.array([3.0, -2.0, 0.4])

    metrics = compute_aligned_cloud_metrics(pred, gt, normal_metrics=False)

    assert metrics["status"] == "ok"
    assert metrics["acc_mean"] < 1.0e-10
    assert metrics["comp_mean"] < 1.0e-10
    assert metrics["acc_mean_norm"] < 1.0e-10
    assert metrics["comp_mean_norm"] < 1.0e-10


def test_existing_sim3_error_uses_row_vector_alignment() -> None:
    xs, ys = np.meshgrid(np.linspace(-1.0, 1.0, 9), np.linspace(-0.5, 0.5, 7), indexing="xy")
    gt = np.stack([xs.reshape(-1), ys.reshape(-1), 0.2 * xs.reshape(-1) + 0.1], axis=1)
    theta = math.radians(-18.0)
    rot = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pred = (gt @ rot.T) * 0.8 + np.array([-0.3, 0.6, 1.2])

    metrics = sim3_error(pred, gt)

    assert metrics["mean"] < 1.0e-10
    assert metrics["mean_norm"] < 1.0e-10


if __name__ == "__main__":
    test_sim3_aligned_accuracy_completion_are_near_zero_for_transformed_cloud()
    test_existing_sim3_error_uses_row_vector_alignment()
    print("ok")
