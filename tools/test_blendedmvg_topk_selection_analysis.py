#!/usr/bin/env python3
"""Small tests for Top-K selection role accounting."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from blendedmvg_topk_selection_analysis import summarize_topk_roles


def test_summarize_topk_roles_for_prefix_queries() -> None:
    roles = ["ref", "prefix_support", "tail_context"]
    topk = np.zeros((1, 12, 3), dtype=np.int32)
    for query_index in [1, 2, 3, 5, 6, 7]:
        topk[0, query_index] = np.array([query_index % 4, 4 + (query_index % 4), 8 + (query_index % 4)], dtype=np.int32)

    rows = summarize_topk_roles(
        topk_indices=topk,
        roles=roles,
        tokens_per_view=4,
        patch_start_idx=1,
        prefix_size=2,
    )
    prefix_rows = {
        (row["query_scope"], row["key_role"]): row
        for row in rows
        if row["query_scope"] == "prefix_patch_queries" and row["granularity"] == "role_group"
    }

    assert prefix_rows[("prefix_patch_queries", "ref")]["count"] == 6
    assert prefix_rows[("prefix_patch_queries", "prefix_support")]["count"] == 6
    assert prefix_rows[("prefix_patch_queries", "tail_context")]["count"] == 6
    assert prefix_rows[("prefix_patch_queries", "tail_context")]["share"] == 1.0 / 3.0


if __name__ == "__main__":
    test_summarize_topk_roles_for_prefix_queries()
    print("ok")
