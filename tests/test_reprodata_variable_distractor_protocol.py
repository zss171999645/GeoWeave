#!/usr/bin/env python3
"""Tests for the five-level variable-distractor protocol builder."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

THIS_DIR = Path(__file__).resolve().parent
for candidate in (THIS_DIR, THIS_DIR.parent / "tools"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

import reprodata_variable_distractor_protocol as variable_protocol


VARIANTS = ["noise0", "noise1", "noise2", "noise3", "noise4"]


def write_variant(root: Path, variant: str, *, corrupt_prefix: bool = False) -> None:
    count = int(variant.removeprefix("noise"))
    seq = root / variant / f"sample_a__{variant}"
    color = seq / "color_90"
    color.mkdir(parents=True)
    for index in range(10):
        if index < 6:
            payload = f"prefix-{index}"
            if corrupt_prefix and index == 2:
                payload = "corrupt-prefix"
        else:
            payload = f"{variant}-context-{index}"
        (color / f"frame_{index:04d}.jpg").write_bytes(payload.encode("utf-8"))
    source_labels = ["clean"] * (10 - count) + ["distractor"] * count
    meta = {
        "eval_frame_indices": [0, 1, 2, 3, 4, 5],
        "source_labels": source_labels,
        "prefix_scene": "scene-a",
        "prefix_camera": "03",
        "prefix_frame_names": [0, 1, 2, 3, 4, 5],
    }
    pose_row = " ".join(["1"] * 16)
    (seq / "pose_90.txt").write_text("\n".join([pose_row] * 10) + "\n", encoding="utf-8")
    (seq / "tuple_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_build_protocol_groups_all_five_variants() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for variant in VARIANTS:
            write_variant(root, variant)

        protocol = variable_protocol.build_protocol(root)

    assert protocol["variants"] == VARIANTS
    assert protocol["num_samples"] == 1
    assert protocol["prefix_size"] == 6
    assert set(protocol["samples"][0]["variants"]) == set(VARIANTS)
    assert protocol["validation"]["prefix_hashes_match"] is True
    assert protocol["validation"]["num_variant_inputs"] == 5


def test_validate_fixed_prefix_rejects_changed_target_view() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for variant in VARIANTS:
            write_variant(root, variant, corrupt_prefix=variant == "noise3")

        try:
            variable_protocol.build_protocol(root)
        except RuntimeError as exc:
            assert "fixed prefix mismatch" in str(exc)
        else:
            raise AssertionError("Expected fixed-prefix validation failure")


if __name__ == "__main__":
    test_build_protocol_groups_all_five_variants()
    test_validate_fixed_prefix_rejects_changed_target_view()
    print("ok")
