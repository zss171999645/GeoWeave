#!/usr/bin/env python3
"""Tests for GeoWeave-vs-dense Top-K alignment accounting."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
for candidate in (THIS_DIR, THIS_DIR.parent / "tools"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from reprodata_topk_alignment_analysis import summarize_alignment_numpy


def test_summarize_alignment_numpy_reports_exact_match() -> None:
    dense_probs = np.asarray(
        [
            [0.40, 0.30, 0.20, 0.10],
            [0.10, 0.20, 0.30, 0.40],
        ],
        dtype=np.float64,
    )
    learned = np.asarray([[0, 1], [3, 2]], dtype=np.int64)
    teacher = np.asarray([[0, 1], [3, 2]], dtype=np.int64)

    metrics = summarize_alignment_numpy(learned, teacher, dense_probs)

    assert metrics["queries"] == 2
    assert metrics["topk"] == 2
    assert metrics["mean_recall_at_k"] == 1.0
    assert metrics["mean_jaccard_at_k"] == 1.0
    assert metrics["mean_dense_mass_learned"] == 0.70
    assert metrics["mean_dense_mass_teacher"] == 0.70
    assert metrics["mean_dense_mass_ratio"] == 1.0


def test_summarize_alignment_numpy_reports_partial_overlap_and_mass() -> None:
    dense_probs = np.asarray(
        [
            [0.50, 0.20, 0.20, 0.10],
            [0.05, 0.45, 0.40, 0.10],
        ],
        dtype=np.float64,
    )
    learned = np.asarray([[0, 3], [0, 1]], dtype=np.int64)
    teacher = np.asarray([[0, 1], [1, 2]], dtype=np.int64)

    metrics = summarize_alignment_numpy(learned, teacher, dense_probs)

    assert metrics["queries"] == 2
    assert metrics["mean_overlap_count"] == 1.0
    assert metrics["mean_recall_at_k"] == 0.5
    assert np.isclose(metrics["mean_jaccard_at_k"], 1.0 / 3.0)
    assert np.isclose(metrics["mean_dense_mass_learned"], 0.55)
    assert np.isclose(metrics["mean_dense_mass_teacher"], 0.775)
    assert np.isclose(metrics["mean_dense_mass_ratio"], 0.55 / 0.775)


if __name__ == "__main__":
    test_summarize_alignment_numpy_reports_exact_match()
    test_summarize_alignment_numpy_reports_partial_overlap_and_mass()
    print("ok")
