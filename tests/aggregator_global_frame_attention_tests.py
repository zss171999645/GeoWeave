try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - runtime env may miss torch
    torch = None
    nn = None
    TORCH_ERROR = exc
else:
    TORCH_ERROR = None

if torch is not None:
    from easyvolcap.official_vggt.models.aggregator import Aggregator


class ShapeRecorderBlock(nn.Module if nn is not None else object):
    def __init__(self):
        super().__init__()
        self.input_shapes = []
        self.pos_shapes = []

    def forward(self, x, pos=None):
        self.input_shapes.append(tuple(x.shape))
        self.pos_shapes.append(None if pos is None else tuple(pos.shape))
        return x


def _build_tiny_aggregator(global_frame_attention_layers):
    return Aggregator(
        img_size=28,
        patch_size=14,
        embed_dim=8,
        depth=2,
        num_heads=1,
        mlp_ratio=1.0,
        num_register_tokens=4,
        patch_embed="conv",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=False,
        rope_freq=0,
        init_values=0.01,
        fused_attn=False,
        use_checkpoint=False,
        global_frame_attention_layers=global_frame_attention_layers,
    ).eval()


def test_global_frame_attention_layers_keep_selected_global_blocks_frame_local():
    if torch is None:
        raise RuntimeError(f"torch unavailable: {TORCH_ERROR}")

    agg = _build_tiny_aggregator(global_frame_attention_layers="0")
    frame_local_global = ShapeRecorderBlock()
    normal_global = ShapeRecorderBlock()
    agg.global_blocks[0] = frame_local_global
    agg.global_blocks[1] = normal_global

    bsz, views, tokens_per_view, channels = 2, 3, 5, 8
    tokens = torch.randn(bsz * views, tokens_per_view, channels)
    pos = torch.randn(bsz * views, tokens_per_view, 2)

    tokens, global_idx, intermediates, indexer_loss = agg._process_global_attention(
        tokens, bsz, views, tokens_per_view, channels, 0, pos=pos
    )

    assert global_idx == 1
    assert indexer_loss is None
    assert frame_local_global.input_shapes == [(bsz * views, tokens_per_view, channels)]
    assert frame_local_global.pos_shapes == [(bsz * views, tokens_per_view, 2)]
    assert intermediates[0].shape == (bsz, views, tokens_per_view, channels)

    tokens, global_idx, intermediates, indexer_loss = agg._process_global_attention(
        tokens, bsz, views, tokens_per_view, channels, global_idx, pos=pos
    )

    assert global_idx == 2
    assert indexer_loss is None
    assert normal_global.input_shapes == [(bsz, views * tokens_per_view, channels)]
    assert normal_global.pos_shapes == [(bsz, views * tokens_per_view, 2)]
    assert intermediates[0].shape == (bsz, views, tokens_per_view, channels)
