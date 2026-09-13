import torch
import math

import easyvolcap.utils.custom_indexer as custom_indexer
from easyvolcap.official_vggt.layers.indexer import LightningIndexer
from easyvolcap.official_vggt.layers.rope import RotaryPositionEmbedding2D


def test_custom_indexer_exports_optional_inference_symbol():
    assert hasattr(custom_indexer, "sparse_topk_indexer_func")
    assert hasattr(custom_indexer, "sparse_topk_indexer_inference_func")


def test_compute_scores_uses_source_length_not_query_length():
    indexer = LightningIndexer(dim=16, n_heads=2, head_dim=4, use_topk_kernel=False)
    q = torch.randn(1, 3, 2, 4)
    k = torch.randn(1, 7, 2, 4)
    w = torch.randn(1, 7, 2)

    scores = indexer._compute_scores(q, k, w, mask=None)

    assert scores.shape == (1, 3, 7)


def test_project_casts_query_key_weight_to_score_dtype_consistently():
    indexer = LightningIndexer(
        dim=16,
        n_heads=2,
        head_dim=4,
        rope=RotaryPositionEmbedding2D(),
        qk_norm=True,
        score_dtype="float16",
        use_topk_kernel=False,
    )
    x = torch.randn(1, 10, 16)
    pos = torch.randint(0, 8, (1, 10, 2), dtype=torch.long)

    q, k, w = indexer.project(x, pos=pos)

    assert q.dtype == torch.float16
    assert k.dtype == torch.float16
    assert w.dtype == torch.float16


def test_project_key_weight_casts_key_weight_to_score_dtype_consistently():
    indexer = LightningIndexer(
        dim=16,
        n_heads=2,
        head_dim=4,
        rope=RotaryPositionEmbedding2D(),
        qk_norm=True,
        score_dtype="float16",
        use_topk_kernel=False,
    )
    x = torch.randn(1, 10, 16)
    pos = torch.randint(0, 8, (1, 10, 2), dtype=torch.long)

    k, w = indexer.project_key_weight(x, pos=pos)

    assert k.dtype == torch.float16
    assert w.dtype == torch.float16


def test_select_topk_projected_can_apply_view_bias_to_pre_topk_scores():
    indexer = LightningIndexer(dim=8, n_heads=1, head_dim=4, use_topk_kernel=False)
    q = torch.zeros(1, 2, 1, 4)
    k = torch.zeros(1, 4, 1, 4)
    w = torch.ones(1, 4, 1)

    q_view_ids = torch.tensor([[0, 1]], dtype=torch.long)
    s_view_ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    view_bias = torch.tensor(
        [[
            [0.0, 5.0],
            [7.0, 0.0],
        ]],
        dtype=torch.float32,
    )

    topk_indices, topk_scores = indexer.select_topk_projected(
        q,
        k,
        w,
        topk=1,
        return_scores=True,
        view_bias_data=dict(
            q_view_ids=q_view_ids,
            s_view_ids=s_view_ids,
            view_bias=view_bias,
        ),
    )

    assert topk_indices.shape == (1, 2, 1)
    assert topk_scores.shape == (1, 2, 1)
    assert int(topk_indices[0, 0, 0]) in (2, 3)
    assert int(topk_indices[0, 1, 0]) in (0, 1)


def test_select_topk_projected_keeps_view_bias_path_valid_when_topk_kernel_enabled():
    indexer = LightningIndexer(dim=8, n_heads=1, head_dim=4, use_topk_kernel=True)
    q = torch.zeros(1, 2, 1, 4)
    k = torch.zeros(1, 4, 1, 4)
    w = torch.ones(1, 4, 1)

    topk_indices, topk_scores = indexer.select_topk_projected(
        q,
        k,
        w,
        topk=1,
        return_scores=True,
        view_bias_data=dict(
            q_view_ids=torch.tensor([[0, 1]], dtype=torch.long),
            s_view_ids=torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
            view_bias=torch.tensor(
                [[
                    [0.0, 5.0],
                    [7.0, 0.0],
                ]],
                dtype=torch.float32,
            ),
        ),
    )

    assert topk_indices.shape == (1, 2, 1)
    assert topk_scores.shape == (1, 2, 1)
    assert int(topk_indices[0, 0, 0]) in (2, 3)
    assert int(topk_indices[0, 1, 0]) in (0, 1)


def test_sparse_topk_indexer_func_with_view_bias_matches_dense_topk():
    q = torch.tensor(
        [[
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ]],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [[
            [[1.0, 0.0]],
            [[0.2, 0.8]],
            [[0.0, 1.0]],
            [[0.8, 0.2]],
        ]],
        dtype=torch.float32,
    )
    w = torch.ones(1, 4, 1, dtype=torch.float32)
    mask = None
    view_bias_data = dict(
        q_view_ids=torch.tensor([[0, 1]], dtype=torch.long),
        s_view_ids=torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
        view_bias=torch.tensor(
            [[
                [0.0, 4.0],
                [3.0, 0.0],
            ]],
            dtype=torch.float32,
        ),
    )
    topk = 2
    scale = 1.0 / math.sqrt(q.shape[-1])

    indexer = LightningIndexer(dim=4, n_heads=1, head_dim=2, use_topk_kernel=False)
    dense_scores = indexer._compute_scores(q, k, w, mask, view_bias_data=view_bias_data)
    expected_scores, expected_indices = torch.topk(dense_scores, topk, dim=-1, sorted=False)

    got_indices, got_scores = custom_indexer.sparse_topk_indexer_func(
        q,
        k,
        w,
        q.new_empty(0),
        topk,
        float(scale),
        256,
        0,
        view_bias_data=view_bias_data,
    )

    exp_order = torch.argsort(expected_indices, dim=-1)
    got_order = torch.argsort(got_indices.to(torch.long), dim=-1)
    exp_indices_sorted = torch.gather(expected_indices, -1, exp_order)
    exp_scores_sorted = torch.gather(expected_scores, -1, exp_order)
    got_indices_sorted = torch.gather(got_indices.to(torch.long), -1, got_order)
    got_scores_sorted = torch.gather(got_scores, -1, got_order)

    assert torch.equal(got_indices_sorted, exp_indices_sorted)
    assert torch.allclose(got_scores_sorted, exp_scores_sorted, atol=1e-6, rtol=1e-6)


def test_sparse_topk_indexer_func_with_view_bias_preserves_gradients():
    q = torch.tensor(
        [[
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ]],
        dtype=torch.float32,
        requires_grad=True,
    )
    k = torch.tensor(
        [[
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[0.7, 0.3]],
            [[0.3, 0.7]],
        ]],
        dtype=torch.float32,
        requires_grad=True,
    )
    w = torch.ones(1, 4, 1, dtype=torch.float32, requires_grad=True)
    view_bias = torch.tensor(
        [[
            [0.0, 2.0],
            [1.5, 0.0],
        ]],
        dtype=torch.float32,
        requires_grad=True,
    )
    view_bias_data = dict(
        q_view_ids=torch.tensor([[0, 1]], dtype=torch.long),
        s_view_ids=torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
        view_bias=view_bias,
    )

    _, topk_scores = custom_indexer.sparse_topk_indexer_func(
        q,
        k,
        w,
        q.new_empty(0),
        2,
        1.0 / math.sqrt(q.shape[-1]),
        256,
        0,
        view_bias_data=view_bias_data,
    )
    loss = topk_scores.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert w.grad is not None
    assert view_bias.grad is not None
    assert float(view_bias.grad.abs().sum()) > 0.0


def test_sparse_topk_indexer_func_with_view_bias_matches_dense_gradients():
    q_dense = torch.tensor(
        [[
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ]],
        dtype=torch.float32,
        requires_grad=True,
    )
    k_dense = torch.tensor(
        [[
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[0.7, 0.3]],
            [[0.3, 0.7]],
        ]],
        dtype=torch.float32,
        requires_grad=True,
    )
    w_dense = torch.tensor([[[1.0], [0.9], [1.1], [0.8]]], dtype=torch.float32, requires_grad=True)
    view_bias_dense = torch.tensor(
        [[
            [0.0, 1.2],
            [1.7, 0.0],
        ]],
        dtype=torch.float32,
        requires_grad=True,
    )
    view_bias_data_dense = dict(
        q_view_ids=torch.tensor([[0, 1]], dtype=torch.long),
        s_view_ids=torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
        view_bias=view_bias_dense,
    )

    indexer = LightningIndexer(dim=4, n_heads=1, head_dim=2, use_topk_kernel=False)
    dense_scores = indexer._compute_scores(q_dense, k_dense, w_dense, mask=None, view_bias_data=view_bias_data_dense)
    dense_topk_scores, _ = torch.topk(dense_scores, 2, dim=-1, sorted=False)
    dense_loss = dense_topk_scores.sum()
    dense_loss.backward()
    dense_grads = (
        q_dense.grad.detach().clone(),
        k_dense.grad.detach().clone(),
        w_dense.grad.detach().clone(),
        view_bias_dense.grad.detach().clone(),
    )

    q = q_dense.detach().clone().requires_grad_(True)
    k = k_dense.detach().clone().requires_grad_(True)
    w = w_dense.detach().clone().requires_grad_(True)
    view_bias = view_bias_dense.detach().clone().requires_grad_(True)
    view_bias_data = dict(
        q_view_ids=torch.tensor([[0, 1]], dtype=torch.long),
        s_view_ids=torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
        view_bias=view_bias,
    )

    _, topk_scores = custom_indexer.sparse_topk_indexer_func(
        q,
        k,
        w,
        q.new_empty(0),
        2,
        1.0 / math.sqrt(q.shape[-1]),
        256,
        0,
        view_bias_data=view_bias_data,
    )
    sparse_loss = topk_scores.sum()
    sparse_loss.backward()
    sparse_grads = (
        q.grad.detach(),
        k.grad.detach(),
        w.grad.detach(),
        view_bias.grad.detach(),
    )

    for got, expected in zip(sparse_grads, dense_grads):
        assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)
