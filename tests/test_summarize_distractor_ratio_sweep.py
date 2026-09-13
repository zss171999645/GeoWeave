#!/usr/bin/env python3
"""Tests for distractor-ratio sweep aggregation helpers."""

from __future__ import annotations

from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
for candidate in (THIS_DIR, THIS_DIR.parent / "tools"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

import summarize_distractor_ratio_sweep as summarize


def test_variant_to_context_ratio_maps_noise_count() -> None:
    assert summarize.variant_to_context_ratio("noise0") == 0.0
    assert summarize.variant_to_context_ratio("noise2") == 0.5
    assert summarize.variant_to_context_ratio("noise4") == 1.0


def test_compute_clean_relative_rows_respects_metric_direction() -> None:
    rows = [
        {"model": "pi3_base", "variant": "noise0", "ATE": 1.0, "nc_mean": 0.8},
        {"model": "pi3_base", "variant": "noise2", "ATE": 1.3, "nc_mean": 0.7},
        {"model": "geoweave_pi3", "variant": "noise0", "ATE": 0.9, "nc_mean": 0.82},
        {"model": "geoweave_pi3", "variant": "noise2", "ATE": 1.0, "nc_mean": 0.80},
    ]

    deltas = summarize.compute_clean_relative_rows(
        rows,
        lower_is_better=["ATE"],
        higher_is_better=["nc_mean"],
    )
    by_key = {(row["model"], row["variant"]): row for row in deltas}

    assert abs(by_key[("pi3_base", "noise2")]["degradation_ATE"] - 0.3) < 1.0e-12
    assert abs(by_key[("pi3_base", "noise2")]["degradation_nc_mean"] - 0.1) < 1.0e-12
    assert abs(by_key[("geoweave_pi3", "noise2")]["degradation_ATE"] - 0.1) < 1.0e-12
    assert abs(by_key[("geoweave_pi3", "noise2")]["degradation_nc_mean"] - 0.02) < 1.0e-12


if __name__ == "__main__":
    test_variant_to_context_ratio_maps_noise_count()
    test_compute_clean_relative_rows_respects_metric_direction()
    print("ok")
