#!/usr/bin/env python3
"""Unit tests for the FastVGGT BlendedMVG rebuttal bridge."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

import fastvggt_blendedmvg_protocol as fastvggt


def test_iter_protocol_jobs_respects_limit_and_variants() -> None:
    protocol = {
        "samples": [
            {"sample_id": "sample_a", "variants": {"clean": {}, "tail": {}, "xscene": {}}},
            {"sample_id": "sample_b", "variants": {"clean": {}, "tail": {}, "xscene": {}}},
        ]
    }

    jobs = list(fastvggt.iter_protocol_jobs(protocol, variants=["tail", "xscene"], sample_limit=1))

    assert jobs == [
        ("sample_a", "tail", protocol["samples"][0]["variants"]["tail"], protocol["samples"][0]),
        ("sample_a", "xscene", protocol["samples"][0]["variants"]["xscene"], protocol["samples"][0]),
    ]


def test_remap_aggregator_special_token_keys() -> None:
    state = {
        "camera_token": 1,
        "patch_embed.cls_token": 2,
        "patch_embed.special_tokens.pos_embed": 3,
        "other": 4,
    }

    remapped = fastvggt.remap_aggregator_special_token_keys(state)

    assert "special_tokens.camera_token" in remapped
    assert "patch_embed.special_tokens.cls_token" in remapped
    assert remapped["patch_embed.special_tokens.pos_embed"] == 3
    assert remapped["other"] == 4
    assert "camera_token" not in remapped
    assert "patch_embed.cls_token" not in remapped


def test_row_sim3_alignment_maps_source_to_target() -> None:
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.5],
            [0.5, 0.25, 1.0],
            [1.2, -0.3, 0.7],
        ],
        dtype=np.float64,
    )
    theta = math.radians(35.0)
    rot = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    target = (source @ rot) * 2.5 + np.array([3.0, -1.0, 0.7])

    transform = fastvggt.estimate_row_sim3(source, target)
    aligned = fastvggt.apply_row_sim3(source, transform)

    assert np.max(np.abs(aligned - target)) < 1.0e-10


def test_load_protocol_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "protocol.json"
        payload = {"prefix_size": 6, "samples": []}
        path.write_text(json.dumps(payload), encoding="utf-8")

        assert fastvggt.load_protocol(path) == payload


def test_checkpoint_import_roots_include_script_parent_repo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        tools = repo / "tools"
        tools.mkdir(parents=True)
        script = tools / "fastvggt_blendedmvg_protocol.py"
        script.write_text("", encoding="utf-8")
        roots = fastvggt.checkpoint_import_roots(script)

        assert roots[0] == repo.resolve()


def test_choose_state_dict_candidate_prefers_matching_fastvggt_keys() -> None:
    target_keys = ["camera_token", "patch_embed.cls_token", "blocks.0.weight"]
    original = {
        "camera_token": 1,
        "patch_embed.cls_token": 2,
        "blocks.0.weight": 3,
    }
    remapped = fastvggt.remap_aggregator_special_token_keys(original)

    chosen, missing, unexpected = fastvggt.choose_state_dict_candidate(target_keys, [remapped, original])

    assert chosen == original
    assert missing == []
    assert unexpected == []


if __name__ == "__main__":
    test_iter_protocol_jobs_respects_limit_and_variants()
    test_remap_aggregator_special_token_keys()
    test_row_sim3_alignment_maps_source_to_target()
    test_load_protocol_roundtrip()
    test_checkpoint_import_roots_include_script_parent_repo()
    test_choose_state_dict_candidate_prefers_matching_fastvggt_keys()
    print("ok")
