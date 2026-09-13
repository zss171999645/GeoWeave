#!/usr/bin/env python3
"""Tests for cached Pi3 camera-pose evaluation helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "eval_reprodata_precomputed_pi3_relpose.py"


def load_module():
    spec = importlib.util.spec_from_file_location("eval_reprodata_precomputed_pi3_relpose", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_pred_c2w_reads_cached_camera_poses() -> None:
    module = load_module()
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "points.npz"
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], 10, axis=0)
        np.savez(path, camera_poses=poses)

        loaded = module.load_pred_c2w(path)

    np.testing.assert_allclose(loaded, poses)


def test_build_clean_delta_summary_uses_noise0_per_sample() -> None:
    module = load_module()
    rows = [
        {"sample_id": "a", "variant": "noise0", "ATE": 1.0, "RPE trans": 2.0, "RPE rot": 3.0},
        {"sample_id": "a", "variant": "noise2", "ATE": 1.4, "RPE trans": 2.5, "RPE rot": 4.0},
        {"sample_id": "b", "variant": "noise0", "ATE": 2.0, "RPE trans": 3.0, "RPE rot": 4.0},
        {"sample_id": "b", "variant": "noise2", "ATE": 2.2, "RPE trans": 3.5, "RPE rot": 4.5},
    ]

    summary = module.build_clean_delta_summary(rows)

    assert summary["noise2"]["num_pairs"] == 2
    assert np.isclose(summary["noise2"]["mean_delta_ATE"], 0.3)
    assert np.isclose(summary["noise2"]["mean_delta_RPE trans"], 0.5)
    assert np.isclose(summary["noise2"]["mean_delta_RPE rot"], 0.75)


def test_parse_eval_indices_reads_fixed_prefix() -> None:
    module = load_module()
    with TemporaryDirectory() as tmpdir:
        meta = Path(tmpdir) / "tuple_meta.json"
        meta.write_text(json.dumps({"eval_frame_indices": [0, 1, 2, 3, 4, 5]}), encoding="utf-8")

        assert module.parse_eval_indices({"tuple_meta_path": str(meta)}) == [0, 1, 2, 3, 4, 5]


def test_overlap_variants_keep_protocol_order_without_noise_parsing() -> None:
    module = load_module()
    order = ["high", "medium", "low", "near_zero"]
    rows = [
        {"sample_id": "a", "variant": variant, "ATE": float(index), "RPE trans": 0.1, "RPE rot": 0.2}
        for index, variant in enumerate(order)
    ]

    summary = module.build_variant_summary(rows, variant_order=order)

    assert list(summary) == order
    assert module.variant_distractor_count("high") is None
    assert module.variant_distractor_count("noise3") == 3


if __name__ == "__main__":
    test_load_pred_c2w_reads_cached_camera_poses()
    test_build_clean_delta_summary_uses_noise0_per_sample()
    test_parse_eval_indices_reads_fixed_prefix()
    test_overlap_variants_keep_protocol_order_without_noise_parsing()
    print("ok")
