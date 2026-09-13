#!/usr/bin/env python3
"""Tests for reproduced-data runtime probe helpers."""

from __future__ import annotations

from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
for candidate in (THIS_DIR, THIS_DIR.parent / "tools"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

import reprodata_runtime_probe as runtime_probe


def test_iter_protocol_jobs_respects_variants_and_limit() -> None:
    protocol = {
        "samples": [
            {
                "sample_id": "a",
                "variants": {
                    "weak_overlap": {"input_dir": "/tmp/a"},
                    "other": {"input_dir": "/tmp/a2"},
                },
            },
            {
                "sample_id": "b",
                "variants": {"weak_overlap": {"input_dir": "/tmp/b"}},
            },
        ]
    }

    jobs = list(runtime_probe.iter_protocol_jobs(protocol, variants=["weak_overlap"], sample_limit=1))

    assert jobs == [("a", "weak_overlap", Path("/tmp/a"))]


def test_aggregate_rows_by_setting_and_model() -> None:
    rows = [
        {"setting": "s", "model": "pi3_base", "status": "ok", "elapsed_seconds": 1.0, "cuda_peak_allocated_gib": 3.0},
        {"setting": "s", "model": "pi3_base", "status": "ok", "elapsed_seconds": 3.0, "cuda_peak_allocated_gib": 5.0},
        {"setting": "s", "model": "geoweave_pi3", "status": "failed:boom"},
    ]

    aggregate = runtime_probe.aggregate_rows(rows)

    base = aggregate[("s", "pi3_base")]
    assert base["ok_jobs"] == 2
    assert base["mean_elapsed_seconds"] == 2.0
    assert base["median_elapsed_seconds"] == 2.0
    assert base["max_cuda_peak_allocated_gib"] == 5.0
    failed = aggregate[("s", "geoweave_pi3")]
    assert failed["ok_jobs"] == 0
    assert failed["failed_jobs"] == 1


if __name__ == "__main__":
    test_iter_protocol_jobs_respects_variants_and_limit()
    test_aggregate_rows_by_setting_and_model()
    print("ok")
