#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "summarize_scannetpp_overlap_sweep.py"
LEVELS = ["high", "medium", "low", "near_zero"]


def load_module():
    spec = importlib.util.spec_from_file_location("summarize_scannetpp_overlap_sweep", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_rows(model: str) -> list[dict]:
    rows = []
    overlaps = {"high": 0.15, "medium": 0.06, "low": 0.02, "near_zero": 0.002}
    for sample_index, sample_id in enumerate(("scene_a__a000001", "scene_b__a000002")):
        for level_index, level in enumerate(LEVELS):
            base_ate = 0.10 + 0.02 * level_index + 0.01 * sample_index
            ate = base_ate if model == "pi3_base" else base_ate - (0.01 + 0.005 * level_index)
            rows.append(
                {
                    "model": model,
                    "sample_id": sample_id,
                    "variant": level,
                    "cross_overlap_mean": overlaps[level] + 0.001 * sample_index,
                    "ATE": ate,
                    "RPE trans": ate * 2,
                    "RPE rot": ate * 3,
                }
            )
    return rows


def test_build_paired_rows_uses_measured_overlap_and_signed_improvement() -> None:
    module = load_module()
    paired = module.build_paired_rows(synthetic_rows("pi3_base"), synthetic_rows("geoweave_pi3"), levels=LEVELS)

    assert len(paired) == 8
    first = paired[0]
    assert first["overlap_level"] == "high"
    assert np.isclose(first["cross_overlap_mean"], 0.15)
    assert np.isclose(first["delta_ATE"], 0.01)
    assert np.isclose(first["delta_RPE_trans"], 0.02)


def test_band_summary_bootstraps_paired_anchors_deterministically() -> None:
    module = load_module()
    paired = module.build_paired_rows(synthetic_rows("pi3_base"), synthetic_rows("geoweave_pi3"), levels=LEVELS)

    first = module.summarize_bands(paired, levels=LEVELS, bootstrap_repeats=200, seed=7)
    second = module.summarize_bands(paired, levels=LEVELS, bootstrap_repeats=200, seed=7)

    assert first == second
    assert [row["overlap_level"] for row in first] == LEVELS
    assert np.isclose(first[0]["mean_cross_overlap"], 0.1505)
    assert first[0]["num_anchors"] == 2
    assert first[0]["delta_ATE_ci_low"] <= first[0]["mean_delta_ATE"] <= first[0]["delta_ATE_ci_high"]
    assert np.isclose(first[0]["median_delta_ATE"], 0.01)
    assert first[0]["max_pi3_ATE"] >= first[0]["median_pi3_ATE"]


def test_build_paired_rows_rejects_missing_model_result() -> None:
    module = load_module()
    geoweave = synthetic_rows("geoweave_pi3")[:-1]
    try:
        module.build_paired_rows(synthetic_rows("pi3_base"), geoweave, levels=LEVELS)
    except ValueError as exc:
        assert "paired key mismatch" in str(exc)
    else:
        raise AssertionError("Expected paired key mismatch")


def test_parse_levels_supports_custom_order() -> None:
    module = load_module()
    assert module.parse_levels("highest,medium,low,lowest") == ("highest", "medium", "low", "lowest")


if __name__ == "__main__":
    test_build_paired_rows_uses_measured_overlap_and_signed_improvement()
    test_band_summary_bootstraps_paired_anchors_deterministically()
    test_build_paired_rows_rejects_missing_model_result()
    test_parse_levels_supports_custom_order()
    print("ok")
