try:
    import torch
except Exception as exc:  # pragma: no cover - runtime env may miss torch
    torch = None
    TORCH_ERROR = exc
else:
    TORCH_ERROR = None

if torch is not None:
    from easyvolcap.utils.base_utils import dotdict
    from easyvolcap.official_vggt.layers.dsa_attention import DSAAttention
    from easyvolcap.official_vggt.models.aggregator import Aggregator


def _build_attn():
    return DSAAttention(dim=1, num_heads=1, indexer_cfg=dotdict(enabled=False))


def _build_attn_with_indexer():
    return DSAAttention(
        dim=1,
        num_heads=1,
        indexer_cfg=dotdict(
            enabled=True,
            n_heads=1,
            head_dim=1,
            topk_block=256,
            topk_merge_blocks=0,
            use_topk_kernel=False,
        ),
    )


def _build_tiny_aggregator_with_indexer():
    return Aggregator(
        img_size=42,
        patch_size=14,
        embed_dim=8,
        depth=1,
        num_heads=1,
        mlp_ratio=1.0,
        num_register_tokens=4,
        patch_embed="conv",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=False,
        rope_freq=100,
        init_values=0.01,
        fused_attn=False,
        use_checkpoint=False,
        indexer_cfg=dotdict(
            enabled=True,
            topk=8,
            n_heads=1,
            head_dim=8,
            topk_block=256,
            topk_merge_blocks=0,
            use_topk_kernel=False,
        ),
    ).eval()


def _expected_round_robin(layout, coarse_indices, topk, fine_src_len, pad_window):
    target_topk = min(int(topk), int(fine_src_len))
    bsz, q_len, _ = coarse_indices.shape
    out = torch.empty((bsz, q_len, target_topk), dtype=torch.long)
    for b in range(bsz):
        for q in range(q_len):
            blocks = []
            for idx in coarse_indices[b, q].tolist():
                row = layout.source_to_fine[idx]
                valid = layout.source_valid[idx]
                blocks.append(row[valid].tolist())
            rr = []
            if bool(layout.get("expand_round_robin", True)):
                max_len = max((len(x) for x in blocks), default=0)
                for step in range(max_len):
                    for block in blocks:
                        if step < len(block):
                            rr.append(block[step])
            else:
                for block in blocks:
                    rr.extend(block)
            if not rr:
                raise RuntimeError("empty coarse selection in synthetic check")
            if len(rr) < target_topk:
                safe_window = min(len(rr), int(pad_window))
                tail = rr[-safe_window:]
                base_len = len(rr)
                for i in range(target_topk - base_len):
                    rr.append(tail[i % safe_window])
            out[b, q] = torch.tensor(rr[:target_topk], dtype=torch.long)
    return out


def _run_case(name, num_views, patch_start_idx, grid_h, grid_w, factor, representative_points=False):
    attn = _build_attn()
    image_tokens = grid_h * grid_w
    tokens_per_view = patch_start_idx + image_tokens
    src_len = tokens_per_view * num_views
    state = dotdict(
        tokens_per_view=tokens_per_view,
        num_views=num_views,
        patch_start_idx=patch_start_idx,
        patch_grid_height=grid_h,
        patch_grid_width=grid_w,
        source_downsample_factor=factor,
        source_downsample_pad_window=64,
        source_downsample_representative_points=representative_points,
    )
    k = torch.arange(src_len, dtype=torch.float32).view(1, src_len, 1, 1)
    w = (torch.arange(src_len, dtype=torch.float32) + 1000).view(1, src_len, 1)

    layout = attn._get_source_downsample_layout(state, src_len, k.device)
    assert layout is not None, f"{name}: layout is None"
    prepared = attn._prepare_source_downsample_projected(k, w, state)
    assert prepared is not None, f"{name}: prepared is None"
    assert prepared.source_k.shape[1] == layout.source_len, f"{name}: source_k len mismatch"
    assert prepared.source_w.shape[1] == layout.source_len, f"{name}: source_w len mismatch"

    for idx in range(layout.source_len):
        fine = layout.source_to_fine[idx][layout.source_valid[idx]]
        assert fine.numel() > 0, f"{name}: empty fine block at {idx}"
        exp_k = k[0, fine, 0, 0].mean().item()
        got_k = prepared.source_k[0, idx, 0, 0].item()
        exp_w = w[0, fine, 0].mean().item()
        got_w = prepared.source_w[0, idx, 0].item()
        assert abs(exp_k - got_k) < 1e-5, f"{name}: pooled k mismatch idx={idx} exp={exp_k} got={got_k}"
        assert abs(exp_w - got_w) < 1e-5, f"{name}: pooled w mismatch idx={idx} exp={exp_w} got={got_w}"

    sel_count = min(4, layout.source_len)
    query0 = list(range(sel_count))
    query1 = list(range(layout.source_len - 1, max(layout.source_len - 1 - sel_count, -1), -1))
    coarse_indices = torch.tensor([query0, query1], dtype=torch.long).unsqueeze(0)
    expanded = attn._expand_source_downsample_chunk(
        coarse_indices,
        layout,
        state,
        topk=min(10, src_len),
        fine_src_len=src_len,
    )
    expected = _expected_round_robin(layout, coarse_indices, topk=min(10, src_len), fine_src_len=src_len, pad_window=64)
    assert torch.equal(expanded.cpu(), expected.cpu()), (
        f"{name}: expanded mismatch expected={expected.tolist()} actual={expanded.tolist()}"
    )


def test_source_downsample_layout_without_edge_tokens():
    if torch is None:
        raise RuntimeError(f"torch unavailable: {TORCH_ERROR}")
    _run_case("no_edge_4x4_factor2", num_views=2, patch_start_idx=2, grid_h=4, grid_w=4, factor=2)


def test_source_downsample_layout_with_edge_tokens():
    if torch is None:
        raise RuntimeError(f"torch unavailable: {TORCH_ERROR}")
    _run_case("with_edge_5x5_factor2", num_views=2, patch_start_idx=2, grid_h=5, grid_w=5, factor=2)


def test_source_downsample_layout_with_full_block_factor4():
    if torch is None:
        raise RuntimeError(f"torch unavailable: {TORCH_ERROR}")
    attn = _build_attn()
    patch_start_idx = 2
    grid_h = 9
    grid_w = 9
    num_views = 2
    tokens_per_view = patch_start_idx + grid_h * grid_w
    src_len = tokens_per_view * num_views
    state = dotdict(
        tokens_per_view=tokens_per_view,
        num_views=num_views,
        patch_start_idx=patch_start_idx,
        patch_grid_height=grid_h,
        patch_grid_width=grid_w,
        source_downsample_factor=4,
        source_downsample_pad_window=64,
        source_downsample_representative_points=False,
    )
    layout = attn._get_source_downsample_layout(state, src_len, torch.device("cpu"))
    assert layout is not None
    assert not bool(layout.use_representative_points)
    assert int(layout.max_block_tokens) == 16
    first_coarse = int(layout.special_count) + int(layout.edge_count)
    first_block = layout.source_to_fine[first_coarse][layout.source_valid[first_coarse]]
    assert first_block.numel() == 16, f"factor4 full block should expand to 16, got {first_block.numel()}"
    _run_case("with_edge_9x9_factor4_full_block", num_views=2, patch_start_idx=2, grid_h=9, grid_w=9, factor=4)


def test_source_downsample_layout_with_representative_factor4():
    if torch is None:
        raise RuntimeError(f"torch unavailable: {TORCH_ERROR}")
    attn = _build_attn()
    patch_start_idx = 2
    grid_h = 9
    grid_w = 9
    num_views = 2
    tokens_per_view = patch_start_idx + grid_h * grid_w
    src_len = tokens_per_view * num_views
    state = dotdict(
        tokens_per_view=tokens_per_view,
        num_views=num_views,
        patch_start_idx=patch_start_idx,
        patch_grid_height=grid_h,
        patch_grid_width=grid_w,
        source_downsample_factor=4,
        source_downsample_pad_window=64,
        source_downsample_representative_points=True,
    )
    layout = attn._get_source_downsample_layout(state, src_len, torch.device("cpu"))
    assert layout is not None
    assert bool(layout.use_representative_points)
    assert int(layout.max_block_tokens) == 4
    _run_case(
        "with_edge_9x9_factor4_representative",
        num_views=2,
        patch_start_idx=2,
        grid_h=9,
        grid_w=9,
        factor=4,
        representative_points=True,
    )


def test_source_downsample_coarse_topk_reserves_special_budget_for_factor4():
    if torch is None:
        raise RuntimeError(f"torch unavailable: {TORCH_ERROR}")
    attn = _build_attn()
    state = dotdict(
        tokens_per_view=5 + 21 * 37,
        num_views=10,
        patch_start_idx=5,
        patch_grid_height=21,
        patch_grid_width=37,
        source_downsample_factor=4,
        source_downsample_coarse_ratio=1.1,
    )
    src_len = int(state.tokens_per_view) * int(state.num_views)
    layout = attn._get_source_downsample_layout(state, src_len, torch.device("cpu"))
    assert layout is not None
    coarse_topk = attn._resolve_source_downsample_coarse_topk(state, layout, topk=256)
    assert coarse_topk == 80, f"expected factor4 auto coarse_topk=80, got {coarse_topk}"


def test_source_downsample_coarse_topk_keeps_factor2_budget_unchanged():
    if torch is None:
        raise RuntimeError(f"torch unavailable: {TORCH_ERROR}")
    attn = _build_attn()
    state = dotdict(
        tokens_per_view=5 + 21 * 37,
        num_views=10,
        patch_start_idx=5,
        patch_grid_height=21,
        patch_grid_width=37,
        source_downsample_factor=2,
        source_downsample_coarse_ratio=1.1,
    )
    src_len = int(state.tokens_per_view) * int(state.num_views)
    layout = attn._get_source_downsample_layout(state, src_len, torch.device("cpu"))
    assert layout is not None
    coarse_topk = attn._resolve_source_downsample_coarse_topk(state, layout, topk=256)
    assert coarse_topk == 88, f"expected factor2 auto coarse_topk=88, got {coarse_topk}"


def test_source_downsample_exact_rerank_matches_full_exact_when_recall_covers_all_blocks():
    if torch is None:
        raise RuntimeError(f"torch unavailable: {TORCH_ERROR}")
    attn = _build_attn_with_indexer()
    patch_start_idx = 0
    grid_h = 4
    grid_w = 4
    num_views = 1
    tokens_per_view = patch_start_idx + grid_h * grid_w
    src_len = tokens_per_view * num_views
    state = dotdict(
        tokens_per_view=tokens_per_view,
        num_views=num_views,
        patch_start_idx=patch_start_idx,
        patch_grid_height=grid_h,
        patch_grid_width=grid_w,
        source_downsample_enabled=True,
        source_downsample_factor=4,
        source_downsample_strategy="exact_rerank",
        source_downsample_coarse_topk=1,
        source_downsample_rerank_query_chunk=4,
    )
    q = torch.ones((1, 2, 1, 1), dtype=torch.float32)
    k = torch.arange(src_len, dtype=torch.float32).view(1, src_len, 1, 1)
    w = torch.ones((1, src_len, 1), dtype=torch.float32)
    prepared = attn._prepare_source_downsample_projected(k, w, state)
    assert prepared is not None
    exact = torch.tensor([[[12, 13, 14, 15], [12, 13, 14, 15]]], dtype=torch.long)
    actual = attn._select_topk_source_downsample_chunk(q, prepared, state, topk=4)
    assert torch.equal(torch.sort(actual, dim=-1).values, torch.sort(exact, dim=-1).values), (
        f"exact_rerank mismatch exact={exact.tolist()} actual={actual.tolist()}"
    )


def test_source_downsample_subcell_top1_keeps_best_scored_2x2_pack():
    if torch is None:
        raise RuntimeError(f"torch unavailable: {TORCH_ERROR}")
    attn = _build_attn_with_indexer()
    patch_start_idx = 0
    grid_h = 4
    grid_w = 4
    num_views = 1
    tokens_per_view = patch_start_idx + grid_h * grid_w
    src_len = tokens_per_view * num_views
    state = dotdict(
        tokens_per_view=tokens_per_view,
        num_views=num_views,
        patch_start_idx=patch_start_idx,
        patch_grid_height=grid_h,
        patch_grid_width=grid_w,
        source_downsample_enabled=True,
        source_downsample_factor=4,
        source_downsample_strategy="subcell_top1",
        source_downsample_coarse_topk=1,
        source_downsample_rerank_query_chunk=4,
    )
    q = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    k = torch.tensor(
        [0, 1, 2, 3,
         4, 5, 6, 7,
         8, 9, 10, 11,
         12, 13, 14, 15],
        dtype=torch.float32,
    ).view(1, src_len, 1, 1)
    w = torch.ones((1, src_len, 1), dtype=torch.float32)
    prepared = attn._prepare_source_downsample_projected(k, w, state)
    assert prepared is not None
    actual = attn._select_topk_source_downsample_chunk(q, prepared, state, topk=4)
    expected = torch.tensor([[[10, 11, 14, 15]]], dtype=torch.long)
    assert torch.equal(torch.sort(actual, dim=-1).values, torch.sort(expected, dim=-1).values), (
        f"subcell_top1 mismatch expected={expected.tolist()} actual={actual.tolist()}"
    )


def test_source_downsample_subcell_static_pack4_uses_highest_weight_subcell():
    if torch is None:
        raise RuntimeError(f"torch unavailable: {TORCH_ERROR}")
    attn = _build_attn_with_indexer()
    patch_start_idx = 0
    grid_h = 4
    grid_w = 4
    num_views = 1
    tokens_per_view = patch_start_idx + grid_h * grid_w
    src_len = tokens_per_view * num_views
    state = dotdict(
        tokens_per_view=tokens_per_view,
        num_views=num_views,
        patch_start_idx=patch_start_idx,
        patch_grid_height=grid_h,
        patch_grid_width=grid_w,
        source_downsample_enabled=True,
        source_downsample_factor=4,
        source_downsample_strategy="subcell_static_pack4",
        source_downsample_coarse_topk=1,
        source_downsample_rerank_query_chunk=4,
    )
    q = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    k = torch.arange(src_len, dtype=torch.float32).view(1, src_len, 1, 1)
    w = torch.ones((1, src_len, 1), dtype=torch.float32)
    w[0, torch.tensor([10, 11, 14, 15]), 0] = 100.0
    prepared = attn._prepare_source_downsample_projected(k, w, state)
    assert prepared is not None
    actual = attn._select_topk_source_downsample_chunk(q, prepared, state, topk=4)
    expected = torch.tensor([[[10, 11, 14, 15]]], dtype=torch.long)
    assert torch.equal(torch.sort(actual, dim=-1).values, torch.sort(expected, dim=-1).values), (
        f"subcell_static_pack4 mismatch expected={expected.tolist()} actual={actual.tolist()}"
    )



def test_qk_sym_x2_outer_query_chunk_uses_source_downsample_query_chunk():
    if torch is None:
        raise RuntimeError(f"torch unavailable: {TORCH_ERROR}")
    attn = _build_attn_with_indexer()
    state = dotdict(source_downsample_query_chunk=4096)
    resolved = attn._resolve_sparse_stream_qk_sym_x2_query_chunk(state, tgt_len=50000, base_query_chunk=16384)
    assert resolved == 4096, f"qk outer chunk should respect source_downsample_query_chunk, got {resolved}"


def test_qk_sym_x2_outer_query_chunk_clamps_to_target_length():
    if torch is None:
        raise RuntimeError(f"torch unavailable: {TORCH_ERROR}")
    attn = _build_attn_with_indexer()
    state = dotdict(source_downsample_query_chunk=4096)
    resolved = attn._resolve_sparse_stream_qk_sym_x2_query_chunk(state, tgt_len=2048, base_query_chunk=16384)
    assert resolved == 2048, f"qk outer chunk should clamp to tgt_len, got {resolved}"

def test_source_downsample_qk_sym_x2_broadcast_reuses_coarse_query_selection():
    if torch is None:
        raise RuntimeError(f"torch unavailable: {TORCH_ERROR}")
    attn = _build_attn_with_indexer()
    patch_start_idx = 0
    grid_h = 4
    grid_w = 4
    num_views = 1
    tokens_per_view = patch_start_idx + grid_h * grid_w
    src_len = tokens_per_view * num_views
    state = dotdict(
        tokens_per_view=tokens_per_view,
        num_views=num_views,
        patch_start_idx=patch_start_idx,
        patch_grid_height=grid_h,
        patch_grid_width=grid_w,
        source_downsample_enabled=True,
        source_downsample_factor=2,
        source_downsample_strategy="qk_sym_x2_broadcast",
        source_downsample_coarse_topk=1,
    )
    q = torch.ones((1, src_len, 1, 1), dtype=torch.float32)
    k = torch.arange(src_len, dtype=torch.float32).view(1, src_len, 1, 1)
    w = torch.ones((1, src_len, 1), dtype=torch.float32)
    prepared = attn._prepare_source_downsample_projected(k, w, state)
    assert prepared is not None
    actual = attn._select_topk_source_downsample_chunk(q, prepared, state, topk=4, q_offset=0)
    expected = torch.tensor(
        [[
            [10, 11, 14, 15],
            [10, 11, 14, 15],
            [10, 11, 14, 15],
            [10, 11, 14, 15],
            [10, 11, 14, 15],
            [10, 11, 14, 15],
            [10, 11, 14, 15],
            [10, 11, 14, 15],
            [10, 11, 14, 15],
            [10, 11, 14, 15],
            [10, 11, 14, 15],
            [10, 11, 14, 15],
            [10, 11, 14, 15],
            [10, 11, 14, 15],
            [10, 11, 14, 15],
            [10, 11, 14, 15],
        ]],
        dtype=torch.long,
    )
    assert torch.equal(torch.sort(actual, dim=-1).values, torch.sort(expected, dim=-1).values), (
        f"qk_sym_x2_broadcast mismatch expected={expected.tolist()} actual={actual.tolist()}"
    )


def test_source_downsample_qk_sym_x2_broadcast_supports_query_slice():
    if torch is None:
        raise RuntimeError(f"torch unavailable: {TORCH_ERROR}")
    attn = _build_attn_with_indexer()
    patch_start_idx = 0
    grid_h = 4
    grid_w = 4
    num_views = 1
    tokens_per_view = patch_start_idx + grid_h * grid_w
    src_len = tokens_per_view * num_views
    state = dotdict(
        tokens_per_view=tokens_per_view,
        num_views=num_views,
        patch_start_idx=patch_start_idx,
        patch_grid_height=grid_h,
        patch_grid_width=grid_w,
        source_downsample_enabled=True,
        source_downsample_factor=2,
        source_downsample_strategy="qk_sym_x2_broadcast",
        source_downsample_coarse_topk=1,
    )
    q = torch.ones((1, src_len, 1, 1), dtype=torch.float32)
    k = torch.arange(src_len, dtype=torch.float32).view(1, src_len, 1, 1)
    w = torch.ones((1, src_len, 1), dtype=torch.float32)
    prepared = attn._prepare_source_downsample_projected(k, w, state)
    assert prepared is not None
    query_prepared = attn._prepare_query_downsample_chunk(q, prepared.layout, 0)
    coarse_indices = attn.indexer.select_topk_projected(
        query_prepared.coarse_q,
        prepared.source_k,
        prepared.source_w,
        mask=None,
        topk=1,
        return_scores=False,
    )
    fine_source_indices = attn._expand_source_downsample_chunk(
        coarse_indices,
        prepared.layout,
        state,
        topk=4,
        fine_src_len=src_len,
    )
    full = attn._broadcast_source_downsample_chunk_to_queries(
        fine_source_indices,
        query_prepared,
        fine_q_len=src_len,
    )
    sliced = attn._broadcast_source_downsample_chunk_to_queries(
        fine_source_indices,
        query_prepared,
        fine_q_len=src_len,
        fine_start=4,
        fine_end=12,
    )
    assert torch.equal(sliced, full[:, 4:12]), (
        f"qk_sym_x2_broadcast slice mismatch expected={full[:, 4:12].tolist()} actual={sliced.tolist()}"
    )


def test_qk_sym_x2_query_downsample_keeps_integer_block_positions():
    if torch is None:
        raise RuntimeError(f"torch unavailable: {TORCH_ERROR}")
    attn = _build_attn_with_indexer()
    state = dotdict(
        tokens_per_view=16,
        num_views=1,
        patch_start_idx=0,
        patch_grid_height=4,
        patch_grid_width=4,
        source_downsample_enabled=True,
        source_downsample_factor=2,
        source_downsample_strategy="qk_sym_x2_broadcast",
    )
    layout = attn._get_source_downsample_layout(state, src_len=16, device=torch.device("cpu"))
    assert layout is not None
    x = torch.arange(16, dtype=torch.float32).view(1, 16, 1)
    pos = torch.cartesian_prod(torch.arange(4, dtype=torch.long), torch.arange(4, dtype=torch.long)).view(1, 16, 2)
    prepared = attn._prepare_query_downsample_inputs_chunk(x, pos, layout, q_offset=0)
    assert prepared.coarse_pos is not None
    assert prepared.coarse_pos.dtype == torch.long, f"expected integer coarse positions, got {prepared.coarse_pos.dtype}"
    expected = torch.tensor(
        [[[0, 0], [0, 2], [2, 0], [2, 2]]],
        dtype=torch.long,
    )
    assert torch.equal(prepared.coarse_pos.cpu(), expected), (
        f"unexpected coarse positions expected={expected.tolist()} actual={prepared.coarse_pos.tolist()}"
    )


def test_aggregator_indexer_state_records_non_square_patch_grid():
    if torch is None:
        raise RuntimeError(f"torch unavailable: {TORCH_ERROR}")
    agg = _build_tiny_aggregator_with_indexer()
    images = torch.rand((1, 2, 3, 28, 42), dtype=torch.float32)
    with torch.inference_mode():
        _ = agg(images)
    assert int(agg.indexer_state.tokens_per_view) == 11
    assert int(agg.indexer_state.num_views) == 2
    assert int(agg.indexer_state.patch_start_idx) == 5
    assert int(agg.indexer_state.patch_grid_height) == 2
    assert int(agg.indexer_state.patch_grid_width) == 3


if __name__ == "__main__":
    test_source_downsample_layout_without_edge_tokens()
    test_source_downsample_layout_with_edge_tokens()
    test_source_downsample_layout_with_full_block_factor4()
    test_source_downsample_layout_with_representative_factor4()
    test_source_downsample_coarse_topk_reserves_special_budget_for_factor4()
    test_source_downsample_coarse_topk_keeps_factor2_budget_unchanged()
    test_source_downsample_exact_rerank_matches_full_exact_when_recall_covers_all_blocks()
    test_source_downsample_subcell_top1_keeps_best_scored_2x2_pack()
    test_source_downsample_subcell_static_pack4_uses_highest_weight_subcell()
    test_source_downsample_qk_sym_x2_broadcast_reuses_coarse_query_selection()
    test_source_downsample_qk_sym_x2_broadcast_supports_query_slice()
    test_qk_sym_x2_query_downsample_keeps_integer_block_positions()
    test_aggregator_indexer_state_records_non_square_patch_grid()
    print("source_downsample_layout_tests passed")
