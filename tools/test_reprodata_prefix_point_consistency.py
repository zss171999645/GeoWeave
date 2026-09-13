#!/usr/bin/env python3
"""Tests for clean/noise prefix point-map consistency metrics."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

import reprodata_prefix_point_consistency as consistency


def test_compute_prefix_change_is_zero_after_global_sim3() -> None:
    clean = np.zeros((6, 4, 4, 3), dtype=np.float64)
    yy, xx = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    for view in range(6):
        clean[view, ..., 0] = xx + view * 0.2
        clean[view, ..., 1] = yy - view * 0.1
        clean[view, ..., 2] = 1.0 + view * 0.05
    theta = math.radians(25.0)
    rotation = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    noise = (clean @ rotation) * 3.0 + np.array([4.0, -2.0, 0.5])

    metrics = consistency.compute_prefix_change(clean, noise, stride=1, max_points=0)

    assert metrics["status"] == "ok"
    assert metrics["paired_mean_norm"] < 1.0e-10
    assert metrics["paired_p90_norm"] < 1.0e-10


def test_compute_rows_uses_model_sample_variant_layout() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        output_root = root / "outputs"
        sample_id = "sample_a"
        for variant, offset in (("clean_tail", 0.0), ("plausible_noise_tail", 0.1)):
            out_dir = output_root / "pi3_base" / sample_id / variant
            out_dir.mkdir(parents=True)
            points = np.zeros((6, 3, 3, 3), dtype=np.float32)
            points[..., 0] = np.arange(3)[None, None, :]
            points[..., 1] = np.arange(3)[None, :, None]
            points[..., 2] = 1.0 + offset
            if variant == "plausible_noise_tail":
                points[:, 1, 1, 2] += 0.3
            np.savez_compressed(out_dir / "points.npz", points=points)
        protocol = {"prefix_size": 6, "samples": [{"sample_id": sample_id, "variants": {}}]}

        rows = consistency.compute_rows(
            protocol,
            output_root,
            models=["pi3_base"],
            clean_variant="clean_tail",
            noise_variant="plausible_noise_tail",
            stride=1,
            max_points=0,
        )

        assert len(rows) == 1
        assert rows[0]["model"] == "pi3_base"
        assert rows[0]["sample_id"] == sample_id
        assert rows[0]["status"] == "ok"
        assert rows[0]["paired_mean_norm"] > 0.0


if __name__ == "__main__":
    test_compute_prefix_change_is_zero_after_global_sim3()
    test_compute_rows_uses_model_sample_variant_layout()
    print("ok")
