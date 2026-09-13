#!/usr/bin/env python3
"""Unit tests for precomputed FastVGGT relpose evaluation helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


MODULE_PATH = Path(__file__).resolve().with_name("eval_fastvggt_precomputed_relpose.py")


def load_module():
    spec = importlib.util.spec_from_file_location("eval_fastvggt_precomputed_relpose", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_iter_protocol_jobs_respects_protocol_variant_order() -> None:
    module = load_module()
    protocol = {
        "variants": ["clean_tail", "plausible_noise_tail"],
        "samples": [
            {
                "sample_id": "sample_a",
                "variants": {
                    "plausible_noise_tail": {"variant": "plausible_noise_tail"},
                    "clean_tail": {"variant": "clean_tail"},
                },
            }
        ],
    }

    jobs = list(module.iter_protocol_jobs(protocol))

    assert [(sample_id, variant) for sample_id, variant, _payload, _sample in jobs] == [
        ("sample_a", "clean_tail"),
        ("sample_a", "plausible_noise_tail"),
    ]


def test_output_sequence_name_keeps_single_variant_names() -> None:
    module = load_module()

    assert module.output_sequence_name({"variants": ["weak_overlap"]}, "seq", "weak_overlap") == "seq"
    assert (
        module.output_sequence_name({"variants": ["clean_tail", "plausible_noise_tail"]}, "seq", "clean_tail")
        == "seq__clean_tail"
    )


def test_resolve_points_npz_uses_setting_model_sample_variant_layout() -> None:
    module = load_module()
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        expected = root / "waymo_weak" / "fastvggt_m0_r090" / "seq" / "weak_overlap" / "points.npz"
        expected.parent.mkdir(parents=True)
        np.savez(expected, extrinsic=np.zeros((1, 3, 4), dtype=np.float32))

        resolved = module.resolve_points_npz(root, "waymo_weak", "fastvggt_m0_r090", "seq", "weak_overlap")

    assert resolved == expected


def test_validate_image_order_rejects_mismatched_order() -> None:
    module = load_module()
    with TemporaryDirectory() as tmpdir:
        points = Path(tmpdir) / "points.npz"
        np.savez(points, image_paths=np.asarray(["/a/frame_0001.png", "/a/frame_0000.png"]))
        payload = {
            "frames": [
                {"image_path": "/a/frame_0000.png"},
                {"image_path": "/a/frame_0001.png"},
            ]
        }
        try:
            module.validate_image_order(points, payload)
        except RuntimeError as exc:
            assert "image order mismatch" in str(exc)
        else:
            raise AssertionError("Expected image-order validation to fail")


def test_build_variant_and_pair_delta_summary() -> None:
    module = load_module()
    rows = [
        {"sample_id": "a", "variant": "clean_tail", "ATE": 1.0, "RPE trans": 2.0, "RPE rot": 3.0},
        {"sample_id": "a", "variant": "plausible_noise_tail", "ATE": 1.5, "RPE trans": 2.5, "RPE rot": 4.0},
        {"sample_id": "b", "variant": "clean_tail", "ATE": 2.0, "RPE trans": 3.0, "RPE rot": 4.0},
        {"sample_id": "b", "variant": "plausible_noise_tail", "ATE": 2.25, "RPE trans": 4.0, "RPE rot": 5.0},
    ]

    variant_summary = module.build_variant_summary(rows)
    pair_delta = module.build_pair_delta_summary(rows)

    assert np.isclose(variant_summary["clean_tail"]["ATE"], 1.5)
    assert np.isclose(variant_summary["plausible_noise_tail"]["RPE trans"], 3.25)
    assert pair_delta["num_pairs"] == 2
    assert np.isclose(pair_delta["mean_delta_ATE"], 0.375)
    assert np.isclose(pair_delta["mean_delta_RPE trans"], 0.75)
    assert np.isclose(pair_delta["mean_delta_RPE rot"], 1.0)


def test_parse_eval_indices_reads_tuple_meta() -> None:
    module = load_module()
    with TemporaryDirectory() as tmpdir:
        meta = Path(tmpdir) / "tuple_meta.json"
        meta.write_text(json.dumps({"eval_frame_indices": [0, 2, 4]}), encoding="utf-8")
        payload = {"tuple_meta_path": str(meta)}

        assert module.parse_eval_indices(payload) == [0, 2, 4]


if __name__ == "__main__":
    test_iter_protocol_jobs_respects_protocol_variant_order()
    test_output_sequence_name_keeps_single_variant_names()
    test_resolve_points_npz_uses_setting_model_sample_variant_layout()
    test_validate_image_order_rejects_mismatched_order()
    test_build_variant_and_pair_delta_summary()
    test_parse_eval_indices_reads_tuple_meta()
