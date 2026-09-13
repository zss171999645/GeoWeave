# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from easyvolcap.official_vggt.layers import PatchEmbed
from easyvolcap.official_vggt.layers.block import Block, DSABlock
from easyvolcap.official_vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from easyvolcap.official_vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import log, yellow

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class SpecialTokens(nn.Module):
    def __init__(self, embed_dim: int, num_register_tokens: int):
        super().__init__()
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        fused_attn=True,
        use_checkpoint=True,
        patch_embed_chunk_size=0,
        indexer_cfg: dict = None,
        output_list_cfg: dict = None,
        global_frame_attention_layers=None,
        global_dense_checkpoint_reentrant=False,
    ):
        super().__init__()
        self.indexer_cfg = dotdict(indexer_cfg or {})
        self.indexer_state = dotdict(enabled=False)
        self.indexer_loss = None
        self.indexer_aux_stats = dotdict()
        self.indexer_loss_backwarded = False
        self.indexer_layerwise_sync_loss = None
        self.use_checkpoint = use_checkpoint
        self.patch_embed_chunk_size = int(patch_embed_chunk_size) if patch_embed_chunk_size else 0
        self.fused_attn = fused_attn
        self.output_list_cfg = dotdict(output_list_cfg or {})
        self.output_list_keep_layers = self._parse_keep_layers(self.output_list_cfg.get("keep_layers", None))
        self.output_list_keep_last = bool(self.output_list_cfg.get("keep_last", True))
        self.global_dense_checkpoint_reentrant = bool(global_dense_checkpoint_reentrant)
        if self.output_list_keep_layers is not None:
            self.output_list_keep_layers = {idx for idx in self.output_list_keep_layers if 0 <= idx < depth}
        self.global_frame_attention_layers = self._parse_global_frame_attention_layers(
            global_frame_attention_layers,
            depth,
        )

        self.__build_patch_embed__(
            patch_embed,
            img_size,
            patch_size,
            num_register_tokens,
            embed_dim=embed_dim,
            fused_attn=fused_attn,
            use_checkpoint=use_checkpoint,
        )

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    fused_attn=fused_attn,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.indexer_layers = None
        self.reuse_topk_source_layer = None
        self.reuse_topk_layers = None
        self.reuse_topk_source_by_layer = {}
        self.reuse_topk_source_layers = set()
        self.reuse_topk_allow_loss = bool(self.indexer_cfg.get("reuse_topk_allow_loss", False))
        self.reuse_topk_debug = bool(self.indexer_cfg.get("reuse_topk_debug", False))
        self._reuse_topk_indices = {}
        self._reuse_topk_scores = {}
        self._reuse_topk_warned_missing = set()
        if self.indexer_cfg.get("enabled", False):
            layers_cfg = self.indexer_cfg.get("indexer_layers", None)
            self.indexer_layers = self._parse_indexer_layers(layers_cfg)
            if self.indexer_layers is not None:
                self.indexer_layers = {idx for idx in self.indexer_layers if 0 <= idx < self.depth}
            groups_cfg = self.indexer_cfg.get("reuse_topk_groups", None)
            parsed_groups = self._parse_reuse_topk_groups(groups_cfg)
            if parsed_groups is not None:
                for layer_idx, source_idx in parsed_groups.items():
                    if 0 <= source_idx < self.depth and 0 <= layer_idx < self.depth and layer_idx != source_idx:
                        self.reuse_topk_source_by_layer[layer_idx] = source_idx
                        self.reuse_topk_source_layers.add(source_idx)
            src_layer_cfg = self.indexer_cfg.get("reuse_topk_source_layer", None)
            if src_layer_cfg is not None and str(src_layer_cfg).strip() != "":
                self.reuse_topk_source_layer = int(src_layer_cfg)
                if not (0 <= self.reuse_topk_source_layer < self.depth):
                    self.reuse_topk_source_layer = None
            reuse_layers_cfg = self.indexer_cfg.get("reuse_topk_layers", None)
            self.reuse_topk_layers = self._parse_indexer_layers(reuse_layers_cfg)
            if self.reuse_topk_layers is not None:
                self.reuse_topk_layers = {idx for idx in self.reuse_topk_layers if 0 <= idx < self.depth}

        global_block_kwargs = dict(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            init_values=init_values,
            qk_norm=qk_norm,
            rope=self.rope,
            fused_attn=fused_attn,
        )
        global_block_kwargs_indexer = dict(global_block_kwargs)
        global_block_kwargs_indexer["indexer_cfg"] = self.indexer_cfg

        global_blocks = []
        for layer_idx in range(depth):
            use_indexer = self.indexer_cfg.get("enabled", False) and (
                self.indexer_layers is None or layer_idx in self.indexer_layers
            )
            if use_indexer:
                global_blocks.append(DSABlock(**global_block_kwargs_indexer))
            else:
                global_blocks.append(block_fn(**global_block_kwargs))

        self.global_blocks = nn.ModuleList(global_blocks)
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.special_tokens = SpecialTokens(embed_dim, num_register_tokens)

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.special_tokens.camera_token, std=1e-6)
        nn.init.normal_(self.special_tokens.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def _should_layerwise_backward(self) -> bool:
        if not self.indexer_cfg.get("layerwise_backward", False):
            return False
        if not self.training:
            return False
        if self.indexer_cfg.get("layerwise_backward_warmup_only", True):
            return bool(self.indexer_state.get("warmup", False))
        return True

    def _backward_indexer_loss(
        self,
        indexer_loss: torch.Tensor,
        layer_module: Optional[nn.Module] = None,
        layer_idx: Optional[int] = None,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        if not self._should_layerwise_backward():
            return indexer_loss
        loss_scaler = float(self.indexer_state.get("loss_scaler", 1.0))
        sync_context = self.indexer_state.get("no_sync_context", None) or nullcontext
        with sync_context():
            (indexer_loss * loss_scaler).backward()

        aux_loss = None
        if layer_module is not None:
            for param in layer_module.parameters():
                if not param.requires_grad:
                    continue
                term = param.mean() * 0.0
                aux_loss = term if aux_loss is None else aux_loss + term
        if aux_loss is not None:
            if self.indexer_layerwise_sync_loss is None:
                self.indexer_layerwise_sync_loss = aux_loss
            else:
                self.indexer_layerwise_sync_loss = self.indexer_layerwise_sync_loss + aux_loss
        self.indexer_loss_backwarded = True
        return indexer_loss.detach()

    @staticmethod
    def _parse_indexer_layers(indexer_layers):
        if indexer_layers is None:
            return None
        if isinstance(indexer_layers, str):
            text = indexer_layers.strip()
            if not text:
                return set()
            lowered = text.lower()
            if lowered in ("all", "*"):
                return None
            parsed = set()
            for item in (item.strip() for item in text.split(",") if item.strip()):
                if "-" not in item:
                    parsed.add(int(item))
                    continue
                left, right = item.split("-", 1)
                left = left.strip()
                right = right.strip()
                if left and right:
                    parsed.update(range(int(left), int(right) + 1))
                    continue
                parsed.add(int(item))
            return parsed
        if isinstance(indexer_layers, (list, tuple, set)):
            return set(int(item) for item in indexer_layers)
        return None

    @staticmethod
    def _parse_global_frame_attention_layers(layers, depth: int):
        if layers is None:
            return set()
        parsed = Aggregator._parse_indexer_layers(layers)
        if parsed is None:
            return set(range(depth))
        return {idx for idx in parsed if 0 <= idx < depth}

    @staticmethod
    def _parse_keep_layers(keep_layers):
        if keep_layers is None:
            return None
        if isinstance(keep_layers, str):
            text = keep_layers.strip()
            if not text:
                return set()
            lowered = text.lower()
            if lowered in ("all", "*"):
                return None
            if "-" in text:
                left, right = text.split("-", 1)
                left = left.strip()
                right = right.strip()
                if left and right:
                    return set(range(int(left), int(right) + 1))
            items = [item.strip() for item in text.split(",") if item.strip()]
            return set(int(item) for item in items)
        if isinstance(keep_layers, (list, tuple, set)):
            return set(int(item) for item in keep_layers)
        return None

    @staticmethod
    def _parse_reuse_topk_groups(groups):
        if groups is None:
            return None
        if isinstance(groups, str):
            text = groups.strip()
            if not text:
                return {}
            items = [item.strip() for item in text.split(",") if item.strip()]
        elif isinstance(groups, (list, tuple, set)):
            items = [str(item).strip() for item in groups if str(item).strip()]
        else:
            return None

        source_by_layer = {}
        for item in items:
            if ":" in item:
                src_text, layers_text = item.split(":", 1)
                source = int(src_text.strip())
                layers = Aggregator._parse_indexer_layers(layers_text.strip())
                if layers is None:
                    continue
                for layer_idx in layers:
                    layer_idx = int(layer_idx)
                    if layer_idx != source:
                        source_by_layer[layer_idx] = source
                continue
            if "-" in item:
                left, right = item.split("-", 1)
                source = int(left.strip())
                right_val = int(right.strip())
                if right_val < source:
                    source, right_val = right_val, source
                for layer_idx in range(source + 1, right_val + 1):
                    source_by_layer[layer_idx] = source
                continue
            _ = int(item)
        return source_by_layer

    @staticmethod
    def _unwrap_block(block: nn.Module) -> nn.Module:
        inner = block
        for _ in range(4):
            next_inner = (
                getattr(inner, "_checkpoint_wrapped_module", None)
                or getattr(inner, "module", None)
            )
            if next_inner is None or next_inner is inner:
                break
            inner = next_inner
        return inner

    def _indexer_state_for_layer(self, layer_idx: int) -> dotdict:
        state = dotdict(self.indexer_state)
        state.layer_idx = layer_idx
        if not state.get("enabled", False):
            return state
        if self.indexer_layers is None or layer_idx in self.indexer_layers:
            return state
        state.enabled = False
        state.sparse = False
        state.compute_loss = False
        return state

    def _should_reuse_topk_for_layer(self, layer_idx: int) -> bool:
        if self.reuse_topk_source_by_layer:
            return layer_idx in self.reuse_topk_source_by_layer
        if self.reuse_topk_source_layer is None:
            return False
        if self.reuse_topk_layers is None:
            return False
        if layer_idx == self.reuse_topk_source_layer:
            return False
        return layer_idx in self.reuse_topk_layers

    def _inject_reuse_topk_state(self, state: dotdict, layer_idx: int) -> dotdict:
        source_layer = None
        if self.reuse_topk_source_by_layer:
            source_layer = self.reuse_topk_source_by_layer.get(layer_idx, None)
            if source_layer is None:
                return state
        else:
            if not self._should_reuse_topk_for_layer(layer_idx):
                return state
            source_layer = self.reuse_topk_source_layer
        cached_indices = self._reuse_topk_indices.get(source_layer, None)
        if cached_indices is None:
            if self.reuse_topk_debug and layer_idx not in self._reuse_topk_warned_missing:
                log(yellow(f"Aggregator: missing cached topk from layer {source_layer}, fallback to recompute at layer {layer_idx}."))
                self._reuse_topk_warned_missing.add(layer_idx)
            return state
        state.reuse_topk_indices = cached_indices
        state.reuse_topk_allow_loss = self.reuse_topk_allow_loss
        if self.reuse_topk_allow_loss:
            cached_scores = self._reuse_topk_scores.get(source_layer, None)
            if cached_scores is not None:
                state.reuse_topk_scores = cached_scores
        return state

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
        fused_attn=True,
        use_checkpoint=True,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
                fused_attn=fused_attn,
                use_checkpoint=use_checkpoint,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "special_tokens") and hasattr(self.patch_embed.special_tokens, "mask_token"):
                self.patch_embed.special_tokens.mask_token.requires_grad_(False)
            elif hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape
        self.indexer_loss_backwarded = False
        self.indexer_layerwise_sync_loss = None

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        if self.patch_embed_chunk_size > 0 and images.shape[0] > self.patch_embed_chunk_size:
            token_chunks = []
            for start in range(0, images.shape[0], self.patch_embed_chunk_size):
                chunk = self.patch_embed(images[start:start + self.patch_embed_chunk_size])
                if isinstance(chunk, dict):
                    chunk = chunk["x_norm_patchtokens"]
                token_chunks.append(chunk)
            patch_tokens = torch.cat(token_chunks, dim=0)
        else:
            patch_tokens = self.patch_embed(images)
            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.special_tokens.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.special_tokens.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape
        if self.indexer_cfg.get("enabled", False):
            patch_grid_height = int(H // self.patch_size)
            patch_grid_width = int(W // self.patch_size)
            self.indexer_state.tokens_per_view = P
            self.indexer_state.num_views = S
            self.indexer_state.patch_start_idx = self.patch_start_idx
            self.indexer_state.camera_tokens_per_view = int(self.special_tokens.camera_token.shape[2])
            self.indexer_state.patch_grid_height = patch_grid_height
            self.indexer_state.patch_grid_width = patch_grid_width
            self._reuse_topk_indices.clear()
            self._reuse_topk_scores.clear()
            self._reuse_topk_warned_missing.clear()

        frame_idx = 0
        global_idx = 0
        if self.output_list_keep_layers is None:
            output_list = []
            keep_layers = None
        else:
            keep_layers = set(self.output_list_keep_layers)
            if self.output_list_keep_last:
                keep_layers.add(self.depth - 1)
            output_list = [None] * self.depth
        indexer_loss_total = None
        indexer_aux_stats = dotdict()
        layer_idx = 0
        concat_inter = None
        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates, indexer_loss = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                    for key, value in getattr(self, "_current_indexer_aux_stats", dotdict()).items():
                        indexer_aux_stats[key] = indexer_aux_stats.get(key, value.new_zeros(())) + value
                    if indexer_loss is not None:
                        indexer_loss_total = indexer_loss if indexer_loss_total is None else indexer_loss_total + indexer_loss
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                if keep_layers is None or layer_idx in keep_layers:
                    concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                    if keep_layers is None:
                        output_list.append(concat_inter)
                    else:
                        output_list[layer_idx] = concat_inter
                layer_idx += 1

        if concat_inter is not None:
            del concat_inter
        del frame_intermediates
        del global_intermediates
        self.indexer_loss = indexer_loss_total
        if (
            "epipolar_selector_candidate_queries" in indexer_aux_stats
            and "epipolar_selector_valid_queries" in indexer_aux_stats
        ):
            indexer_aux_stats.epipolar_selector_valid_query_ratio = (
                indexer_aux_stats.epipolar_selector_valid_queries
                / indexer_aux_stats.epipolar_selector_candidate_queries.clamp_min(1.0)
            )
        self.indexer_aux_stats = indexer_aux_stats
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training and self.use_checkpoint:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        intermediates = []
        indexer_loss_total = None
        self._current_indexer_aux_stats = dotdict()

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            layer_idx = global_idx
            block = self.global_blocks[layer_idx]
            inner_block = self._unwrap_block(block)
            is_dsa = isinstance(inner_block, DSABlock)
            use_frame_range = layer_idx in self.global_frame_attention_layers
            if use_frame_range:
                if is_dsa:
                    raise ValueError(
                        f"global_frame_attention_layers cannot include DSA/indexer layer {layer_idx}; "
                        "remove the overlap with indexer_cfg.indexer_layers."
                    )
                if tokens.shape != (B * S, P, C):
                    tokens = tokens.view(B, S, P, C).view(B * S, P, C)
                if pos is not None and pos.shape != (B * S, P, 2):
                    pos = pos.view(B, S, P, 2).view(B * S, P, 2)
            else:
                if tokens.shape != (B, S * P, C):
                    tokens = tokens.view(B, S, P, C).view(B, S * P, C)
                if pos is not None and pos.shape != (B, S * P, 2):
                    pos = pos.view(B, S, P, 2).view(B, S * P, 2)
            tgt_len = tokens.shape[1]
            if is_dsa:
                layer_state = self._indexer_state_for_layer(layer_idx)
                layer_state = self._inject_reuse_topk_state(layer_state, layer_idx)
                inner_block.indexer_state = layer_state
                inner_block.return_indexer_loss = True
            use_layer_checkpoint = self.training and self.use_checkpoint
            if is_dsa and self._should_layerwise_backward():
                # The warm-up selector loss is backwarded immediately after this
                # layer. Checkpointing the DSA block would recompute the full
                # dense-attention teacher and top-k support loss during that
                # local backward, which is both slower and higher peak-memory.
                use_layer_checkpoint = False
            if use_layer_checkpoint:
                use_reentrant = self.use_reentrant
                if self.global_dense_checkpoint_reentrant and (not is_dsa) and (not use_frame_range):
                    use_reentrant = True
                if is_dsa:
                    tokens, indexer_loss = checkpoint(block, tokens, pos, use_reentrant=use_reentrant)
                else:
                    tokens = checkpoint(block, tokens, pos, use_reentrant=use_reentrant)
                    indexer_loss = None
            else:
                if is_dsa:
                    tokens, indexer_loss = block(tokens, pos=pos)
                else:
                    tokens = block(tokens, pos=pos)
                    indexer_loss = None
            should_cache_source = False
            if self.reuse_topk_source_by_layer:
                should_cache_source = layer_idx in self.reuse_topk_source_layers
            elif self.reuse_topk_source_layer is not None:
                should_cache_source = layer_idx == self.reuse_topk_source_layer
            if is_dsa and should_cache_source:
                cached_indices = getattr(inner_block.attn, "last_topk_indices", None)
                if isinstance(cached_indices, torch.Tensor):
                    self._reuse_topk_indices[layer_idx] = cached_indices.detach()
                    cached_scores = getattr(inner_block.attn, "last_topk_scores", None)
                    if isinstance(cached_scores, torch.Tensor):
                        self._reuse_topk_scores[layer_idx] = cached_scores.detach()
                    if self.reuse_topk_debug:
                        log(yellow(f"Aggregator: cached topk indices from layer {layer_idx} (shape={tuple(cached_indices.shape)})."))
            if is_dsa:
                stats = getattr(inner_block.attn, "last_epipolar_selector_stats", None)
                if stats is not None:
                    for key in ("epipolar_selector_valid_queries", "epipolar_selector_candidate_queries"):
                        value = stats.get(key, None)
                        if isinstance(value, torch.Tensor):
                            value = value.detach().to(torch.float32)
                            self._current_indexer_aux_stats[key] = (
                                self._current_indexer_aux_stats.get(key, value.new_zeros(())) + value
                            )
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
            if indexer_loss is not None:
                indexer_module = inner_block.attn.indexer if (is_dsa and getattr(inner_block.attn, "indexer", None) is not None) else None
                indexer_loss = self._backward_indexer_loss(
                    indexer_loss,
                    layer_module=indexer_module,
                    layer_idx=global_idx - 1,
                    seq_len=tgt_len,
                )
                indexer_loss_total = indexer_loss if indexer_loss_total is None else indexer_loss_total + indexer_loss

        return tokens, global_idx, intermediates, indexer_loss_total

    def set_indexer_state(self, state: dict):
        self.indexer_state = dotdict(state)


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
