#!/usr/bin/env python3
"""Tests for building a Waymo clean/noise protocol from reproduced data."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import reprodata_waymo_plausible_protocol as protocol_builder


def _make_variant(root: Path, base: str, variant: str) -> Path:
    seq = root / f"{base}__{variant}"
    color = seq / "color_90"
    color.mkdir(parents=True)
    for index in range(10):
        (color / f"frame_{index:04d}.jpg").write_bytes(b"")
    meta = {
        "protocol": "waymo_simple_plausible_wrong_context_v1",
        "sample_id": base,
        "variant": variant,
        "eval_frame_indices": [0, 1, 2, 3, 4, 5],
        "prefix_frame_names": [f"prefix_{index}" for index in range(6)],
        "tail_frame_names": [f"tail_{index}" for index in range(4)],
    }
    (seq / "tuple_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return seq


def test_build_protocol_pairs_clean_and_noise_variants() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = "simple_plausible_rank00_scene__anchor0001__wrong_other__tail0002"
        clean = _make_variant(root, base, "clean_tail")
        noise = _make_variant(root, base, "plausible_noise_tail")

        protocol = protocol_builder.build_protocol(root)

        assert protocol["prefix_size"] == 6
        assert protocol["context_size"] == 4
        assert protocol["variants"] == ["clean_tail", "plausible_noise_tail"]
        assert len(protocol["samples"]) == 1
        sample = protocol["samples"][0]
        assert sample["sample_id"] == base
        assert sample["variants"]["clean_tail"]["input_dir"] == str((clean / "color_90").resolve())
        assert sample["variants"]["plausible_noise_tail"]["input_dir"] == str((noise / "color_90").resolve())
        clean_roles = [frame["role"] for frame in sample["variants"]["clean_tail"]["frames"]]
        noise_roles = [frame["role"] for frame in sample["variants"]["plausible_noise_tail"]["frames"]]
        assert clean_roles == ["prefix"] * 6 + ["clean_context"] * 4
        assert noise_roles == ["prefix"] * 6 + ["distractor_context"] * 4


def test_build_protocol_reports_missing_pair() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = "simple_plausible_rank01_scene__anchor0001__wrong_other__tail0002"
        _make_variant(root, base, "clean_tail")

        protocol = protocol_builder.build_protocol(root)

        assert protocol["samples"] == []
        assert protocol["skipped_samples"][0]["sample_id"] == base
        assert protocol["skipped_samples"][0]["missing_variants"] == ["plausible_noise_tail"]


if __name__ == "__main__":
    test_build_protocol_pairs_clean_and_noise_variants()
    test_build_protocol_reports_missing_pair()
    print("ok")
