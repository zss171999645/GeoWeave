#!/usr/bin/env python3
"""Tests for building unified protocols across reproduced paper settings."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
for candidate in (THIS_DIR, THIS_DIR.parent / "tools"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

import reprodata_unified_protocol as unified


def _touch_images(color_dir: Path, count: int) -> None:
    color_dir.mkdir(parents=True)
    for index in range(count):
        (color_dir / f"frame_{index:04d}.png").write_bytes(b"")


def test_build_single_variant_protocol_uses_five_view_groups() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        seq = root / "seq_a"
        _touch_images(seq / "color_90", 10)
        (seq / "tuple_meta.json").write_text(
            json.dumps({"views_per_cluster": 5, "scene_name": "scene_a"}),
            encoding="utf-8",
        )

        protocol = unified.build_single_variant_protocol(
            root,
            protocol_name="scannetpp_weak_overlap",
            variant_name="weak_overlap",
            default_prefix_size=5,
        )

        assert protocol["protocol_name"] == "scannetpp_weak_overlap"
        assert protocol["prefix_size"] == 5
        assert protocol["context_size"] == 5
        assert protocol["num_samples"] == 1
        sample = protocol["samples"][0]
        frames = sample["variants"]["weak_overlap"]["frames"]
        assert [frame["role"] for frame in frames[:5]] == ["group_a"] * 5
        assert [frame["role"] for frame in frames[5:]] == ["group_b"] * 5


def test_build_single_variant_protocol_keeps_tuple_frame_roles() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        seq = root / "seq_b"
        _touch_images(seq / "color_90", 4)
        (seq / "tuple_meta.json").write_text(
            json.dumps(
                {
                    "views_per_cluster": 2,
                    "frames": [
                        {"role": "anchor_a", "frame_id": "a0"},
                        {"role": "support_a", "frame_id": "a1"},
                        {"role": "anchor_b", "frame_id": "b0"},
                        {"role": "support_b", "frame_id": "b1"},
                    ],
                }
            ),
            encoding="utf-8",
        )

        protocol = unified.build_single_variant_protocol(
            root,
            protocol_name="scannetpp_weak_overlap",
            variant_name="weak_overlap",
            default_prefix_size=2,
        )

        frames = protocol["samples"][0]["variants"]["weak_overlap"]["frames"]
        assert [frame["role"] for frame in frames] == ["anchor_a", "support_a", "anchor_b", "support_b"]
        assert frames[0]["source_frame_name"] == "a0"


if __name__ == "__main__":
    test_build_single_variant_protocol_uses_five_view_groups()
    test_build_single_variant_protocol_keeps_tuple_frame_roles()
    print("ok")
