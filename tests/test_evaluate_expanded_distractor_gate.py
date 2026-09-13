#!/usr/bin/env python3
"""Tests for staged expanded-distractor stopping gates."""

from __future__ import annotations

from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for candidate in (REPO_ROOT / "tools", THIS_DIR.parent / "tools"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

import evaluate_expanded_distractor_gate as gate


def rows(model: str, values: dict[str, tuple[float, float]], *, prefix: str) -> list[dict[str, object]]:
    output = []
    for sample_index in range(2):
        for variant, (ate, rpe) in values.items():
            output.append(
                {
                    "model": model,
                    "sample_id": f"{prefix}{sample_index}",
                    "variant": variant,
                    "ATE": ate + 0.01 * sample_index,
                    "RPE trans": rpe + 0.01 * sample_index,
                }
            )
    return output


def supporting_set(prefix: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pi3 = rows("pi3_base", {"noise0": (0.10, 0.15), "noise4": (0.30, 0.35)}, prefix=prefix)
    geo = rows("geoweave_pi3", {"noise0": (0.10, 0.15), "noise4": (0.20, 0.23)}, prefix=prefix)
    return pi3, geo


def test_endpoint_gate_proceeds_when_new_and_combined_support_both_metrics() -> None:
    new_pi3, new_geo = supporting_set("new")
    old_pi3, old_geo = supporting_set("old")
    result = gate.endpoint_gate(new_pi3, new_geo, old_pi3, old_geo)
    assert result["decision"] == "proceed"
    assert result["sets"]["new20"]["advantage_ATE"] > 0
    assert result["sets"]["combined40"]["advantage_RPE trans"] > 0


def test_endpoint_gate_stops_when_new_set_fails_both_metrics() -> None:
    new_pi3 = rows("pi3_base", {"noise0": (0.10, 0.15), "noise4": (0.20, 0.25)}, prefix="new")
    new_geo = rows("geoweave_pi3", {"noise0": (0.10, 0.15), "noise4": (0.30, 0.35)}, prefix="new")
    old_pi3, old_geo = supporting_set("old")
    result = gate.endpoint_gate(new_pi3, new_geo, old_pi3, old_geo)
    assert result["decision"] == "stop"
    assert "new20_both_metrics_nonpositive" in result["reasons"]


def test_endpoint_gate_stops_when_combined_rpe_advantage_is_nonpositive() -> None:
    new_pi3 = rows("pi3_base", {"noise0": (0.10, 0.15), "noise4": (0.35, 0.20)}, prefix="new")
    new_geo = rows("geoweave_pi3", {"noise0": (0.10, 0.15), "noise4": (0.15, 0.45)}, prefix="new")
    old_pi3, old_geo = supporting_set("old")
    result = gate.endpoint_gate(new_pi3, new_geo, old_pi3, old_geo)
    assert result["decision"] == "stop"
    assert "combined40_RPE trans_nonpositive" in result["reasons"]


def test_full_sweep_gate_requires_lower_endpoint_and_degradation_area() -> None:
    pi3_values = {f"noise{k}": (0.10 + 0.05 * k, 0.15 + 0.04 * k) for k in range(5)}
    geo_values = {f"noise{k}": (0.10 + 0.02 * k, 0.15 + 0.015 * k) for k in range(5)}
    result = gate.full_sweep_gate(
        rows("pi3_base", pi3_values, prefix="all"),
        rows("geoweave_pi3", geo_values, prefix="all"),
    )
    assert result["decision"] == "proceed"
    assert result["degradation_area_advantage_ATE"] > 0
    assert result["degradation_area_advantage_RPE trans"] > 0


if __name__ == "__main__":
    test_endpoint_gate_proceeds_when_new_and_combined_support_both_metrics()
    test_endpoint_gate_stops_when_new_set_fails_both_metrics()
    test_endpoint_gate_stops_when_combined_rpe_advantage_is_nonpositive()
    test_full_sweep_gate_requires_lower_endpoint_and_degradation_area()
    print("ok")
