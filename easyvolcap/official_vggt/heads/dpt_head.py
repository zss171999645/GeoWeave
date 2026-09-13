# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


import os
from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from torch.utils.checkpoint import checkpoint
from .head_act import activate_head
from .utils import create_uv_grid, position_grid_to_embed
from easyvolcap.official_vggt.training.loss import compute_depth_loss, compute_point_loss


class DPTHead(nn.Module):
    """
    DPT  Head for dense prediction tasks.

    This implementation follows the architecture described in "Vision Transformers for Dense Prediction"
    (https://arxiv.org/abs/2103.13413). The DPT head processes features from a vision transformer
    backbone and produces dense predictions by fusing multi-scale features.

    Args:
        dim_in (int): Input dimension (channels).
        patch_size (int, optional): Patch size. Default is 14.
        output_dim (int, optional): Number of output channels. Default is 4.
        activation (str, optional): Activation type. Default is "inv_log".
        conf_activation (str, optional): Confidence activation type. Default is "expp1".
        features (int, optional): Feature channels for intermediate representations. Default is 256.
        out_channels (List[int], optional): Output channels for each intermediate layer.
        intermediate_layer_idx (List[int], optional): Indices of layers from aggregated tokens used for DPT.
        pos_embed (bool, optional): Whether to use positional embedding. Default is True.
        feature_only (bool, optional): If True, return features only without the last several layers and activation head. Default is False.
        down_ratio (int, optional): Downscaling factor for the output resolution. Default is 1.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
        use_checkpoint: bool = False,
        checkpoint_use_reentrant: bool = False,
    ) -> None:
        super(DPTHead, self).__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx
        self.use_checkpoint = bool(use_checkpoint)
        self.checkpoint_use_reentrant = bool(checkpoint_use_reentrant)

        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features, expand=False)

        # Attach additional modules to scratch.
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32

        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
            )
            conv2_in_channels = head_features_1 // 2

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )

    @staticmethod
    def _get_layer_tokens(aggregated_tokens_list: Union[List[torch.Tensor], Dict[int, torch.Tensor]], layer_idx: int) -> torch.Tensor:
        if isinstance(aggregated_tokens_list, dict):
            tokens = aggregated_tokens_list.get(layer_idx, None)
        else:
            if layer_idx >= len(aggregated_tokens_list):
                raise IndexError(f"Layer index {layer_idx} out of range for aggregated tokens.")
            tokens = aggregated_tokens_list[layer_idx]
        if tokens is None:
            raise ValueError(f"Missing aggregated tokens for layer {layer_idx}.")
        return tokens

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the DPT head, supports processing by chunking frames.
        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            patch_start_idx (int): Starting index for patch tokens in the token sequence.
                Used to separate patch tokens from other tokens (e.g., camera or register tokens).
            frames_chunk_size (int, optional): Number of frames to process in each chunk.
                If None or larger than S, all frames are processed at once. Default: 8.

        Returns:
            Tensor or Tuple[Tensor, Tensor]:
                - If feature_only=True: Feature maps with shape [B, S, C, H, W]
                - Otherwise: Tuple of (predictions, confidence) both with shape [B, S, 1, H, W]
        """
        B, S, _, H, W = images.shape

        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0

        # Process frames in batches
        all_preds = []
        all_conf = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            # Process batch of frames
            if self.feature_only:
                chunk_output = self._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_output)
            else:
                chunk_preds, chunk_conf = self._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_preds)
                all_conf.append(chunk_conf)

        # Concatenate results along the sequence dimension
        if self.feature_only:
            return torch.cat(all_preds, dim=1)
        else:
            return torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
        batch_start_idx: int = None,
        batch_end_idx: int = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()
        if batch_start_idx is not None and batch_end_idx is not None:
            images = images[batch_start_idx:batch_end_idx].contiguous()

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = self._get_layer_tokens(aggregated_tokens_list, layer_idx)[:, :, patch_start_idx:]

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
            if batch_start_idx is not None and batch_end_idx is not None:
                x = x[batch_start_idx:batch_end_idx]

            x = x.reshape(B * S, -1, x.shape[-1])

            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        if self.training and self.use_checkpoint:
            def _scratch_forward(layer_1, layer_2, layer_3, layer_4):
                return self.scratch_forward([layer_1, layer_2, layer_3, layer_4])

            out = checkpoint(
                _scratch_forward,
                out[0],
                out[1],
                out[2],
                out[3],
                use_reentrant=self.checkpoint_use_reentrant,
            )
        else:
            out = self.scratch_forward(out)
        # Interpolate fused output to match target image resolution.
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        if self.feature_only:
            return out.view(B, S, *out.shape[1:])

        out = self.scratch.output_conv2(out)
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)

        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])
        return preds, conf

    @staticmethod
    def _add_axis_pos_embed_(
        x: torch.Tensor,
        pos: torch.Tensor,
        channel_offset: int,
        freq_count: int,
        axis_embed_dim: int,
        height: int,
        width: int,
        ratio: float,
        omega_0: float = 100.0,
        freq_chunk_size: int = 8,
    ) -> None:
        device = pos.device
        omega_dtype = torch.float32 if device.type == "mps" else torch.double
        pos = pos.reshape(-1, 1).to(dtype=omega_dtype)
        denom = axis_embed_dim / 2.0
        for freq_start in range(0, freq_count, freq_chunk_size):
            freq_end = min(freq_start + freq_chunk_size, freq_count)
            omega = torch.arange(freq_start, freq_end, dtype=omega_dtype, device=device)
            omega /= denom
            omega = 1.0 / omega_0**omega

            phase = pos * omega.reshape(1, -1)
            phase.sin_()
            embed = phase.to(dtype=x.dtype).view(height, width, -1).permute(2, 0, 1).unsqueeze(0)
            x[:, channel_offset + freq_start : channel_offset + freq_end].add_(embed, alpha=ratio)
            del phase, embed

            phase = pos * omega.reshape(1, -1)
            phase.cos_()
            embed = phase.to(dtype=x.dtype).view(height, width, -1).permute(2, 0, 1).unsqueeze(0)
            x[:, channel_offset + freq_count + freq_start : channel_offset + freq_count + freq_end].add_(
                embed,
                alpha=ratio,
            )
            del phase, embed, omega

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x without materializing the full HxWxC map.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        channels = x.shape[1]
        if channels % 4 != 0:
            pos_embed = position_grid_to_embed(pos_embed, channels)
            pos_embed = pos_embed.permute(2, 0, 1).unsqueeze(0)
            pos_embed = pos_embed.mul(ratio)
            return x.add_(pos_embed)

        height, width = pos_embed.shape[:2]
        pos_flat = pos_embed.reshape(-1, 2)
        axis_embed_dim = channels // 2
        freq_count = axis_embed_dim // 2
        self._add_axis_pos_embed_(x, pos_flat[:, 0], 0, freq_count, axis_embed_dim, height, width, ratio)
        self._add_axis_pos_embed_(
            x,
            pos_flat[:, 1],
            axis_embed_dim,
            freq_count,
            axis_embed_dim,
            height,
            width,
            ratio,
        )
        return x

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out


class DPTHeadWithChunkwiseBP(DPTHead):
    def __init__(
        self,
        *args,
        loss_type: str = "depth",
        chunk_cfg: dict = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_type = str(loss_type)
        self.chunk_cfg = chunk_cfg or {}
        self.batch_chunk_size = int(self.chunk_cfg.get("batch_chunk_size", 1))
        self.frames_chunk_size = int(self.chunk_cfg.get("frames_chunk_size", 4))
        self.enabled = bool(self.chunk_cfg.get("enabled", True))
        self.return_predictions = bool(self.chunk_cfg.get("return_predictions", True))
        self.backward_auxiliary_immediately = bool(self.chunk_cfg.get("backward_auxiliary_immediately", False))
        self.forward_checkpoint = bool(self.chunk_cfg.get("forward_checkpoint", False))
        self.forward_checkpoint_use_reentrant = bool(self.chunk_cfg.get("forward_checkpoint_use_reentrant", False))
        self.empty_cache_before_backward_mb = int(self.chunk_cfg.get("empty_cache_before_backward_mb", 0))
        self.loss_cfg = {}
        self.last_loss_dict = None
        self.chunkwise_bp = True
        if self.feature_only:
            raise ValueError("DPTHeadWithChunkwiseBP only supports feature_only=False")

    def set_loss_cfg(self, loss_cfg: dict) -> None:
        self.loss_cfg = dict(loss_cfg or {})

    def _compute_chunk_loss(self, pred: torch.Tensor, conf: torch.Tensor, batch_data: dict) -> dict:
        if self.loss_type == "depth":
            return compute_depth_loss(
                {"depth": pred, "depth_conf": conf},
                batch_data,
                **self.loss_cfg,
            )
        if self.loss_type == "point":
            return compute_point_loss(
                {"world_points": pred, "world_points_conf": conf},
                batch_data,
                **self.loss_cfg,
            )
        raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    @staticmethod
    def _is_cuda_oom(exc: RuntimeError) -> bool:
        return "out of memory" in str(exc).lower()

    @staticmethod
    def _slice_detached_tokens(
        aggregated_tokens_list: Union[List[torch.Tensor], Dict[int, torch.Tensor]],
        batch_start: int,
        batch_end: int,
        frames_start: int,
        frames_end: int,
    ) -> Union[List[torch.Tensor], Dict[int, torch.Tensor]]:
        def _slice_token(token: torch.Tensor):
            if token is None:
                return None
            token = token[batch_start:batch_end, frames_start:frames_end].contiguous()
            return token.detach().requires_grad_(token.requires_grad)

        if isinstance(aggregated_tokens_list, dict):
            return {key: _slice_token(token) for key, token in aggregated_tokens_list.items()}
        return [_slice_token(token) for token in aggregated_tokens_list]

    @staticmethod
    def _accumulate_token_auxiliary_loss(
        token_auxiliary_loss: torch.Tensor,
        detached_tokens_list: Union[List[torch.Tensor], Dict[int, torch.Tensor]],
        aggregated_tokens_list: Union[List[torch.Tensor], Dict[int, torch.Tensor]],
        batch_start: int,
        batch_end: int,
        frames_start: int,
        frames_end: int,
        loss_scaler: float,
    ) -> torch.Tensor:
        if isinstance(detached_tokens_list, dict):
            token_items = detached_tokens_list.items()
        else:
            token_items = enumerate(detached_tokens_list)

        for idx, token in token_items:
            if token is None or not token.requires_grad or token.grad is None:
                continue
            target = aggregated_tokens_list[idx][batch_start:batch_end, frames_start:frames_end]
            grad = token.grad.detach()
            token.grad = None
            token_auxiliary_loss = token_auxiliary_loss + (target * grad).sum() / loss_scaler
        return token_auxiliary_loss

    def _maybe_empty_cache_before_backward(self, tensor: torch.Tensor) -> bool:
        if self.empty_cache_before_backward_mb <= 0 or not tensor.is_cuda:
            return False
        free_bytes, _ = torch.cuda.mem_get_info(tensor.device)
        if free_bytes >= self.empty_cache_before_backward_mb * 1024 * 1024:
            return False
        torch.cuda.empty_cache()
        return True

    def _forward_chunk(
        self,
        detached_tokens_list: Union[List[torch.Tensor], Dict[int, torch.Tensor]],
        chunk_images: torch.Tensor,
        patch_start_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.forward_checkpoint:
            return self._forward_impl(
                detached_tokens_list,
                chunk_images,
                patch_start_idx,
            )

        if isinstance(detached_tokens_list, dict):
            token_keys = [key for key, token in detached_tokens_list.items() if token is not None]
            token_args = tuple(detached_tokens_list[key] for key in token_keys)

            def _checkpointed_forward(*values):
                token_dict = dict(detached_tokens_list)
                for key, value in zip(token_keys, values):
                    token_dict[key] = value
                return self._forward_impl(
                    token_dict,
                    chunk_images,
                    patch_start_idx,
                )

        else:
            token_indices = [idx for idx, token in enumerate(detached_tokens_list) if token is not None]
            token_args = tuple(detached_tokens_list[idx] for idx in token_indices)

            def _checkpointed_forward(*values):
                token_list = list(detached_tokens_list)
                for idx, value in zip(token_indices, values):
                    token_list[idx] = value
                return self._forward_impl(
                    token_list,
                    chunk_images,
                    patch_start_idx,
                )

        if not any(token.requires_grad for token in token_args):
            return _checkpointed_forward(*token_args)
        return checkpoint(
            _checkpointed_forward,
            *token_args,
            use_reentrant=self.forward_checkpoint_use_reentrant,
        )

    def forward(
        self,
        aggregated_tokens_list: Union[List[torch.Tensor], Dict[int, torch.Tensor]],
        images: torch.Tensor,
        patch_start_idx: int,
        batch=None,
        official_batch: dict = None,
        loss_scaler: float = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self.last_loss_dict = None
        if not self.training or not self.enabled or batch is None or official_batch is None:
            return super().forward(aggregated_tokens_list, images, patch_start_idx)

        B, S, _, H, W = images.shape
        batch_chunk_size = max(int(self.batch_chunk_size), 1)
        frames_chunk_size = max(int(self.frames_chunk_size), 1)
        frames_chunk_size = max(frames_chunk_size // max(B, 1), 1)

        if loss_scaler is None:
            loss_scaler = float(getattr(batch, "loss_scaler", 1.0))
        sync_context = getattr(batch, "no_sync_context", nullcontext)

        mask = official_batch["point_masks"].to(dtype=torch.bool)
        mask_sum = mask.sum().clamp(min=1).to(dtype=images.dtype)

        if self.loss_type == "depth":
            loss_keys = ("loss_conf_depth", "loss_reg_depth", "loss_grad_depth")
        else:
            loss_keys = ("loss_conf_point", "loss_reg_point", "loss_grad_point")
        loss_accum = {key: torch.zeros((), device=images.device, dtype=images.dtype) for key in loss_keys}

        all_preds = []
        all_conf = []
        token_auxiliary_loss = images.new_zeros(())

        for batch_start in range(0, B, batch_chunk_size):
            batch_end = min(batch_start + batch_chunk_size, B)
            batch_preds = []
            batch_conf = []

            frames_start = 0
            while frames_start < S:
                current_chunk_size = min(frames_chunk_size, S - frames_start)
                while True:
                    frames_end = min(frames_start + current_chunk_size, S)
                    chunk_tokens = None
                    chunk_images = None
                    try:
                        chunk_tokens = self._slice_detached_tokens(
                            aggregated_tokens_list,
                            batch_start,
                            batch_end,
                            frames_start,
                            frames_end,
                        )
                        chunk_images = images[batch_start:batch_end, frames_start:frames_end].contiguous()
                        chunk_preds, chunk_conf = self._forward_chunk(
                            chunk_tokens,
                            chunk_images,
                            patch_start_idx,
                        )
                        break
                    except RuntimeError as exc:
                        if (not self._is_cuda_oom(exc)) or current_chunk_size <= 1:
                            raise
                        del chunk_tokens, chunk_images
                        torch.cuda.empty_cache()
                        current_chunk_size = max(1, current_chunk_size // 2)

                mask_chunk = mask[batch_start:batch_end, frames_start:frames_end]
                data_ratio = (mask_chunk.sum() / mask_sum).to(dtype=images.dtype)

                if self.loss_type == "depth":
                    batch_data = {
                        "depths": official_batch["depths"][batch_start:batch_end, frames_start:frames_end],
                        "point_masks": mask_chunk,
                    }
                else:
                    batch_data = {
                        "world_points": official_batch["world_points"][batch_start:batch_end, frames_start:frames_end],
                        "point_masks": mask_chunk,
                    }

                loss_dict = self._compute_chunk_loss(chunk_preds, chunk_conf, batch_data)
                if chunk_preds.requires_grad:
                    chunk_loss = sum(loss_dict[key] for key in loss_keys) * data_ratio * loss_scaler
                    self._maybe_empty_cache_before_backward(chunk_preds)
                    with sync_context():
                        chunk_loss.backward()
                    token_auxiliary_loss = self._accumulate_token_auxiliary_loss(
                        token_auxiliary_loss,
                        chunk_tokens,
                        aggregated_tokens_list,
                        batch_start,
                        batch_end,
                        frames_start,
                        frames_end,
                        loss_scaler,
                    )
                for key in loss_keys:
                    loss_accum[key] = loss_accum[key] + loss_dict[key].detach() * data_ratio

                if self.return_predictions:
                    batch_preds.append(chunk_preds.detach())
                    batch_conf.append(chunk_conf.detach())

                frames_start = frames_end
                del chunk_tokens, chunk_images, chunk_preds, chunk_conf, loss_dict

            if self.return_predictions:
                all_preds.append(torch.cat(batch_preds, dim=1))
                all_conf.append(torch.cat(batch_conf, dim=1))

        if self.return_predictions:
            all_preds = torch.cat(all_preds, dim=0)
            all_conf = torch.cat(all_conf, dim=0)
        else:
            all_preds = None
            all_conf = None

        param_sync_loss = images.new_zeros(())
        for param in self.parameters():
            if param.requires_grad:
                param_sync_loss = param_sync_loss + param.mean() * 0.0

        if self.backward_auxiliary_immediately and token_auxiliary_loss.requires_grad:
            with sync_context():
                (token_auxiliary_loss * loss_scaler).backward(retain_graph=True)
            batch.chunkwise_bp_loss = batch.get("chunkwise_bp_loss", 0.0) + param_sync_loss
        else:
            batch.chunkwise_bp_loss = batch.get("chunkwise_bp_loss", 0.0) + param_sync_loss + token_auxiliary_loss
        self.last_loss_dict = loss_accum

        return all_preds, all_conf


################################################################################
# Modules
################################################################################


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, groups=1):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
