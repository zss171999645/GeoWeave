#!/usr/bin/env python3
"""Tests for the reproduced-data Pi3 protocol runner."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

import reprodata_pi3_protocol_infer as runner
from aidi.scripts.baselines import eval_pi3_mv_recon_core as mv_core


def test_iter_protocol_jobs_maps_variants_to_model_output_dirs() -> None:
    protocol = {
        "samples": [
            {
                "sample_id": "sample_a",
                "variants": {
                    "clean_tail": {"input_dir": "/data/a_clean"},
                    "plausible_noise_tail": {"input_dir": "/data/a_noise"},
                },
            },
            {
                "sample_id": "sample_b",
                "variants": {
                    "clean_tail": {"input_dir": "/data/b_clean"},
                    "plausible_noise_tail": {"input_dir": "/data/b_noise"},
                },
            },
        ]
    }

    jobs = list(
        runner.iter_protocol_jobs(
            protocol,
            output_root=Path("/out"),
            model_name="geoweave_pi3",
            variants=["plausible_noise_tail", "clean_tail"],
            sample_limit=1,
        )
    )

    assert jobs == [
        (
            "sample_a",
            "plausible_noise_tail",
            Path("/data/a_noise"),
            Path("/out/geoweave_pi3/sample_a/plausible_noise_tail"),
        ),
        (
            "sample_a",
            "clean_tail",
            Path("/data/a_clean"),
            Path("/out/geoweave_pi3/sample_a/clean_tail"),
        ),
    ]


def test_extract_pi3_points_and_poses_resizes_points_and_keeps_poses() -> None:
    points = torch.arange(1 * 2 * 2 * 3 * 3, dtype=torch.float32).reshape(1, 2, 2, 3, 3)
    poses = torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
    prediction = {
        "points": points,
        "camera_poses": poses,
    }

    result = mv_core.extract_pi3_points_and_poses(
        prediction,
        data_size=(4, 5),
        point_source="native",
    )

    assert result["points"].shape == (2, 4, 5, 3)
    assert result["camera_poses"].shape == (2, 4, 4)
    torch.testing.assert_close(torch.from_numpy(result["camera_poses"]), poses[0])


def test_prediction_cache_payload_omits_dense_points_in_camera_only_mode() -> None:
    predictions = {
        "points": np.ones((10, 8, 12, 3), dtype=np.float32),
        "camera_poses": np.repeat(np.eye(4, dtype=np.float32)[None], 10, axis=0),
    }

    payload = runner.prediction_cache_payload(
        predictions=predictions,
        image_paths=[Path(f"frame_{index:04d}.jpg") for index in range(10)],
        camera_only=True,
    )

    assert set(payload) == {"camera_poses", "image_paths"}
    assert payload["camera_poses"].shape == (10, 4, 4)


def test_prediction_cache_is_valid_rejects_interrupted_npz() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        interrupted = root / "interrupted.npz"
        interrupted.write_bytes(b"")
        assert not runner.prediction_cache_is_valid(interrupted)

        valid = root / "valid.npz"
        np.savez_compressed(
            valid,
            camera_poses=np.repeat(np.eye(4, dtype=np.float32)[None], 10, axis=0),
            image_paths=np.asarray([f"frame_{index:04d}.jpg" for index in range(10)]),
        )
        assert runner.prediction_cache_is_valid(valid)


if __name__ == "__main__":
    test_iter_protocol_jobs_maps_variants_to_model_output_dirs()
    test_extract_pi3_points_and_poses_resizes_points_and_keeps_poses()
    test_prediction_cache_payload_omits_dense_points_in_camera_only_mode()
    test_prediction_cache_is_valid_rejects_interrupted_npz()
    print("ok")
