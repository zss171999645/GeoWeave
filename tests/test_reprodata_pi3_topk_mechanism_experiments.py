#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from reprodata_pi3_topk_mechanism_experiments import (  # noqa: E402
    _base_capture_forward,
    _geoweave_capture_forward,
    _selected_query_indices,
    aggregate_metric_rows,
    dense_topk_for_query_indices,
    frame_role_labels,
    infer_tokens_per_view,
    iter_protocol_samples,
    layer_names_from_indexer_layers,
    query_indices_numpy,
    summarize_role_target_ratio_numpy,
    summarize_same_view_group_ratio_numpy,
    summarize_topk_pair_numpy,
    topk_attention_chunked,
    write_topk_record_npz,
)


def test_layer_names_from_indexer_layers_maps_to_actual_decoder_attention_names() -> None:
    names = layer_names_from_indexer_layers("9-17")
    assert names == [
        "decoder.19.attn",
        "decoder.21.attn",
        "decoder.23.attn",
        "decoder.25.attn",
        "decoder.27.attn",
        "decoder.29.attn",
        "decoder.31.attn",
        "decoder.33.attn",
        "decoder.35.attn",
    ]


def test_query_indices_numpy_respects_prefix_patch_scope() -> None:
    indices = query_indices_numpy(
        num_tokens=12,
        tokens_per_view=4,
        patch_start_idx=1,
        prefix_size=2,
        scope="prefix_patch_queries",
        stride=2,
        max_queries=0,
    )
    assert indices.tolist() == [1, 3, 6]


def test_iter_protocol_samples_splits_samples_into_disjoint_shards() -> None:
    protocol = {"samples": [{"sample_id": f"s{i}"} for i in range(7)]}

    shard0 = iter_protocol_samples(protocol, sample_limit=0, sample_shard_index=0, sample_num_shards=3)
    shard1 = iter_protocol_samples(protocol, sample_limit=0, sample_shard_index=1, sample_num_shards=3)
    shard2 = iter_protocol_samples(protocol, sample_limit=0, sample_shard_index=2, sample_num_shards=3)

    assert [item["sample_id"] for item in shard0] == ["s0", "s3", "s6"]
    assert [item["sample_id"] for item in shard1] == ["s1", "s4"]
    assert [item["sample_id"] for item in shard2] == ["s2", "s5"]
    assert sorted(item["sample_id"] for shard in (shard0, shard1, shard2) for item in shard) == [
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
    ]


def test_selected_query_indices_supports_all_queries_and_fair_stride_downsampling() -> None:
    all_queries = _selected_query_indices(
        num_tokens=20,
        prefix_size=5,
        scope="all_patch_queries",
        stride=1,
        max_queries=0,
    )
    stride_four = _selected_query_indices(
        num_tokens=20,
        prefix_size=5,
        scope="all_patch_queries",
        stride=4,
        max_queries=0,
    )
    prefix_all = _selected_query_indices(
        num_tokens=20,
        prefix_size=5,
        scope="prefix_patch_queries",
        stride=1,
        max_queries=0,
    )

    assert all_queries.tolist() == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    assert stride_four.tolist() == [1, 9, 17]
    assert prefix_all.tolist() == [1, 3, 5, 7, 9]


def test_summarize_topk_pair_numpy_uses_base_attention_mass_on_geoweave_topk() -> None:
    base_topk = np.array([[0, 1], [2, 3]], dtype=np.int64)
    geoweave_topk = np.array([[1, 2], [2, 4]], dtype=np.int64)
    base_probs = np.array(
        [
            [0.50, 0.30, 0.10, 0.05, 0.05],
            [0.05, 0.05, 0.60, 0.20, 0.10],
        ],
        dtype=np.float64,
    )
    out = summarize_topk_pair_numpy(base_topk, geoweave_topk, base_probs)

    assert out["queries"] == 2
    assert out["topk"] == 2
    assert math.isclose(out["mean_overlap_count"], 1.0)
    assert math.isclose(out["mean_recall_at_k"], 0.5)
    assert math.isclose(out["mean_jaccard_at_k"], 1.0 / 3.0)
    assert math.isclose(out["mean_base_mass_on_base_topk"], 0.80)
    assert math.isclose(out["mean_base_mass_on_geoweave_topk"], 0.55)
    assert math.isclose(out["mean_base_mass_ratio_geoweave_over_base"], 0.55 / 0.80)


def test_infer_tokens_per_view_requires_even_view_layout() -> None:
    assert infer_tokens_per_view(num_tokens=40, num_views=10) == 4

    try:
        infer_tokens_per_view(num_tokens=41, num_views=10)
    except ValueError as exc:
        assert "Cannot infer tokens_per_view" in str(exc)
    else:
        raise AssertionError("Expected uneven token layout to raise ValueError")


def test_frame_role_labels_maps_weak_groups_and_context_roles() -> None:
    weak_frames = [
        {"index": 1, "role": "group_a"},
        {"index": 0, "group": "A", "role": "anchor"},
        {"index": 2, "role": "group_b"},
    ]
    assert frame_role_labels(weak_frames) == ["A", "A", "B"]

    context_frames = [
        {"index": 0, "role": "prefix"},
        {"index": 1, "role": "clean_context"},
        {"index": 2, "role": "distractor_context"},
    ]
    assert frame_role_labels(context_frames) == ["prefix", "clean_context", "distractor_context"]


def test_summarize_same_view_group_ratio_numpy_counts_keys_in_query_group() -> None:
    topk = np.array(
        [
            [0, 1, 6, 7],
            [4, 5, 8, 9],
        ],
        dtype=np.int64,
    )
    q_indices = np.array([0, 6], dtype=np.int64)
    out = summarize_same_view_group_ratio_numpy(
        topk=topk,
        q_indices=q_indices,
        tokens_per_view=2,
        view_labels=["A", "A", "B", "B", "B"],
    )

    assert out["A"]["queries"] == 1
    assert out["B"]["queries"] == 1
    assert math.isclose(out["A"]["mean_same_view_group_ratio"], 0.5)
    assert math.isclose(out["B"]["mean_same_view_group_ratio"], 1.0)


def test_summarize_role_target_ratio_numpy_counts_useful_context_keys() -> None:
    topk = np.array(
        [
            [0, 2, 4, 6],
            [1, 3, 5, 7],
        ],
        dtype=np.int64,
    )
    q_indices = np.array([0, 1], dtype=np.int64)
    out = summarize_role_target_ratio_numpy(
        topk=topk,
        q_indices=q_indices,
        tokens_per_view=2,
        view_labels=["prefix", "prefix", "clean_context", "distractor_context"],
        target_roles={"clean_context"},
    )

    assert out["queries"] == 2
    assert math.isclose(out["mean_target_role_ratio"], 0.25)


def test_aggregate_metric_rows_groups_by_setting_variant_scope_and_layer() -> None:
    rows = [
        {
            "setting": "s",
            "variant": "v",
            "query_scope": "prefix",
            "layer": "decoder.19.attn",
            "mean_recall_at_k": 0.5,
            "mean_jaccard_at_k": 0.25,
            "mean_base_mass_ratio_geoweave_over_base": 0.8,
            "queries": 2,
            "topk": 4,
        },
        {
            "setting": "s",
            "variant": "v",
            "query_scope": "prefix",
            "layer": "decoder.19.attn",
            "mean_recall_at_k": 1.0,
            "mean_jaccard_at_k": 0.75,
            "mean_base_mass_ratio_geoweave_over_base": 0.9,
            "queries": 4,
            "topk": 4,
        },
    ]

    grouped = aggregate_metric_rows(rows, group_keys=("setting", "variant", "query_scope", "layer"))
    assert len(grouped) == 1
    assert grouped[0]["rows"] == 2
    assert math.isclose(grouped[0]["mean_recall_at_k"], 0.75)
    assert math.isclose(grouped[0]["mean_jaccard_at_k"], 0.5)
    assert math.isclose(grouped[0]["mean_base_mass_ratio_geoweave_over_base"], 0.85)


def test_topk_attention_chunked_matches_manual_topk_softmax() -> None:
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    v = torch.tensor([[[[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]]])
    out, idx = topk_attention_chunked(q, k, v, topk=2, scale=1.0, query_chunk=1)

    full_scores = torch.einsum("bhqd,bhkd->bhqk", q, k)
    summary_scores = full_scores.float().mean(dim=1)
    _summary_scores, top_idx = torch.topk(summary_scores, k=2, dim=-1, largest=True, sorted=False)
    top_scores = full_scores.gather(-1, top_idx.unsqueeze(1))
    weights = torch.softmax(top_scores, dim=-1)
    gathered_v = torch.gather(
        v.unsqueeze(2).expand(-1, -1, q.shape[2], -1, -1),
        3,
        top_idx.unsqueeze(1).unsqueeze(-1).expand(-1, q.shape[1], -1, -1, v.shape[-1]),
    )
    expected = (weights.unsqueeze(-1) * gathered_v).sum(dim=-2)

    assert torch.equal(torch.sort(idx, dim=-1).values, torch.sort(top_idx.cpu(), dim=-1).values)
    assert torch.allclose(out, expected)


def test_dense_topk_for_query_indices_can_skip_probability_capture_for_role_only_runs() -> None:
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]]])
    q_indices = torch.tensor([0, 1, 2], dtype=torch.long)

    topk, probs = dense_topk_for_query_indices(
        q=q,
        k=k,
        q_indices=q_indices,
        topk=2,
        query_chunk=1,
        include_probs=False,
    )

    full_scores = torch.einsum("bhqd,bhkd->bhqk", q, k)
    dense_probs = full_scores.float().softmax(dim=-1).mean(dim=1)
    _scores, expected_topk = torch.topk(dense_probs, k=2, dim=-1, largest=True, sorted=False)

    assert torch.equal(topk, expected_topk)
    assert probs is None


def test_base_capture_forward_returns_original_forward_output_without_using_topk() -> None:
    class DummyAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_heads = 1
            self.scale = 1.0
            self.rope = None
            self.qkv = torch.nn.Linear(2, 6, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self._topk_prefix_size = 2
            self._topk_k = 2
            self._topk_scopes = ("all_patch_queries",)
            self._topk_query_token_stride = 1
            self._topk_max_query_tokens = 0
            self._topk_capture_query_chunk = 2
            self._topk_role_only = True

        def _topk_original_forward(self, x, attn_bias=None, xpos=None):
            return x + 7.0

    module = DummyAttention()
    x = torch.arange(40, dtype=torch.float32).reshape(1, 20, 2)

    out = _base_capture_forward(module, x)

    assert torch.equal(out, x + 7.0)
    assert len(module.last_base_dense_topk_records) == 1
    assert "base_probs" not in module.last_base_dense_topk_records[0]
    assert module.last_base_dense_topk_records[0]["q_indices"].tolist() == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]


def test_geoweave_capture_forward_records_scorer_and_dense_teacher_topk_without_changing_output() -> None:
    class DummyGeoAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_heads = 1
            self.scale = 1.0
            self.rope = None
            self.qkv = torch.nn.Linear(2, 6, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self._topk_prefix_size = 2
            self._topk_k = 2
            self._topk_scopes = ("all_patch_queries",)
            self._topk_query_token_stride = 2
            self._topk_max_query_tokens = 0
            self._topk_capture_query_chunk = 2

            with torch.no_grad():
                self.qkv.weight.zero_()
                self.qkv.weight[0, 0] = 1.0
                self.qkv.weight[1, 1] = 1.0
                self.qkv.weight[2, 0] = 1.0
                self.qkv.weight[3, 1] = 1.0

        def _topk_original_forward(self, x, pos=None):
            scorer_row = torch.tensor([18, 17], dtype=torch.long).view(1, 1, 2)
            self.last_topk_indices = scorer_row.expand(1, x.shape[1], 2).contiguous()
            return x + 11.0

    module = DummyGeoAttention()
    x = torch.arange(40, dtype=torch.float32).reshape(1, 20, 2)

    out = _geoweave_capture_forward(module, x)

    assert torch.equal(out, x + 11.0)
    assert len(module.last_geoweave_topk_records) == 1
    assert len(module.last_geoweave_dense_teacher_topk_records) == 1
    assert module.last_geoweave_topk_records[0]["q_indices"].tolist() == [1, 5, 9, 13, 17]
    assert module.last_geoweave_dense_teacher_topk_records[0]["q_indices"].tolist() == [1, 5, 9, 13, 17]
    assert module.last_geoweave_topk_records[0]["geoweave_topk"][0, 0].tolist() == [18, 17]
    assert set(module.last_geoweave_dense_teacher_topk_records[0]["teacher_topk"][0, 0].tolist()) == {18, 19}


def test_write_topk_record_npz_saves_reusable_topk_arrays_and_manifest() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_topk_record_npz(
            record_root=root,
            metadata={
                "setting": "s",
                "sample_id": "sample/with spaces",
                "variant": "v",
                "layer": "decoder.19.attn",
                "query_scope": "all_patch_queries",
                "batch_index": 0,
                "tokens_per_view": 4,
                "num_views": 2,
                "view_labels": ["A", "B"],
                "queries": 2,
                "topk": 3,
            },
            q_indices=np.array([1, 5], dtype=np.int64),
            pi3_dense_topk=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
            geoweave_scorer_topk=np.array([[6, 5, 4], [3, 2, 1]], dtype=np.int64),
            geoweave_dense_teacher_topk=np.array([[0, 1, 2], [6, 7, 8]], dtype=np.int64),
        )

        manifest_lines = (root / "topk_records_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(manifest_lines) == 1
        import json

        row = json.loads(manifest_lines[0])
        path = root / row["file"]
        assert path.exists()
        data = np.load(path)
        assert data["q_indices"].dtype == np.int32
        assert data["pi3_dense_topk"].tolist() == [[1, 2, 3], [4, 5, 6]]
        assert data["geoweave_scorer_topk"].tolist() == [[6, 5, 4], [3, 2, 1]]
        assert data["geoweave_dense_teacher_topk"].tolist() == [[0, 1, 2], [6, 7, 8]]
        assert row["geoweave_dense_teacher_topk_shape"] == [2, 3]
        assert row["contains_dense_probabilities"] is False
