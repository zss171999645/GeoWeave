# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


import os
from typing import List, Dict, Tuple, Union

from easyvolcap.engine import SUPERVISORS
from easyvolcap.utils.base_utils import dotdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .head_act import activate_head
from .utils import create_uv_grid, position_grid_to_embed
from easyvolcap.utils.dist_utils import get_rank


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
        intermediate_layer_idx (List[int], Dict, optional): Indices of layers from aggregated tokens used for DPT.
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
    ) -> None:
        super(DPTHead, self).__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx
        self.use_checkpoint = use_checkpoint

        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=oc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for oc in out_channels
            ]
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

        self.scratch = _make_scratch(
            out_channels,
            features,
            expand=False,
        )

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

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
        **kwargs,
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
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
            if batch_start_idx is not None and batch_end_idx is not None:
                x = x[batch_start_idx:batch_end_idx]

            x = x.cuda()

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
        if self.use_checkpoint:
            out = torch.utils.checkpoint.checkpoint(
                self.scratch_forward,
                out[0], out[1], out[2], out[3],
                use_reentrant=False
            )
        else:
            out = self.scratch_forward(out[0], out[1], out[2], out[3])

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
            return out.reshape(B, S, *out.shape[1:])

        out = self.scratch.output_conv2(out)
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)

        preds = preds.reshape(B, S, *preds.shape[1:])
        conf = conf.reshape(B, S, *conf.shape[1:])
        return preds, conf

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1).to(x.dtype)
        return x + pos_embed

    def scratch_forward(self, layer_1, layer_2, layer_3, layer_4) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
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
    def __init__(self, *args, supervisor_type: str = "DepthSupervisor", **kwargs):
        super(DPTHeadWithChunkwiseBP, self).__init__(*args, **kwargs)
        self.supervisor_type = supervisor_type
        if supervisor_type == "DepthSupervisor":
            self.loss_cfg = dotdict(type="DepthSupervisor", dpt_conf_grad_loss_weight=1.0, dpt_conf_grad_loss_range=0.98, dpt_conf_grad_loss_separate=False)
        elif supervisor_type == "PointSupervisor":
            self.loss_cfg = dotdict(type="PointSupervisor", xyz_conf_grad_loss_weight=1.0, xyz_conf_grad_loss_range=0.98, xyz_conf_grad_loss_separate=False)
        else:
            raise ValueError(f"Invalid supervisor type: {supervisor_type}")
        self.loss_fn = SUPERVISORS.build(self.loss_cfg, network=nn.Identity())
        assert not self.feature_only, "DPTHeadWithChunkwiseBP only supports feature_only=False"

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        batch_chunk_size: int = 4,
        frames_chunk_size: int = 4,
        batch: dotdict = None,
        loss_scaler: float = 1.0,
        verbose: bool = False,
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
        frames_chunk_size = max(frames_chunk_size // B, 1)
        verbose = verbose and self.training and get_rank() == 0

        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or not self.training or batch.get("debug_skip_chunkwise_bp", False):
            return super(DPTHeadWithChunkwiseBP, self).forward(aggregated_tokens_list, images, patch_start_idx)

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0
        assert self.training, "only supports training"

        # Process frames in batches
        all_preds = []
        all_conf = []
        assert batch is not None, "batch is required for chunkwise BP"

        def print_mem(tag):
            mem = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
            mem_max = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            print(f"{tag}: mem: {mem:.2f} GB, mem_max: {mem_max:.2f} GB")

        if verbose:
            print(f"dpt with chunkwise BP, frames_chunk_size: {frames_chunk_size}, training: {self.training}, images.shape: {images.shape}")
        detached_aggregated_tokens_list = {k: v.detach().requires_grad_(v.requires_grad) for k, v in aggregated_tokens_list.items() if int(k) >= 0}

        B, S = batch.dpt.shape[:2]
        H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
        msk = batch.msk.clone().reshape(B, S, H, W) > 0  # (B, S, H, W)    
        msk_sum = msk.sum() + 1e-6
        total_loss = 0.0

        for batch_start_idx in range(0, B, batch_chunk_size):
            batch_end_idx = min(batch_start_idx + batch_chunk_size, B)
            batch_preds = []
            batch_conf = []

            for frames_start_idx in range(0, S, frames_chunk_size):
                frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

                chunk_preds, chunk_conf = self._forward_impl(
                    detached_aggregated_tokens_list, images, patch_start_idx, 
                    frames_start_idx, frames_end_idx, 
                    batch_start_idx, batch_end_idx,
                )

                if chunk_preds.requires_grad:
                    if self.supervisor_type == "DepthSupervisor":
                        chunk_output = dotdict(
                            dpt_map=chunk_preds,
                            dpt_cnf=chunk_conf,
                        )
                        chunk_batch = dotdict(
                            dpt=batch.dpt[batch_start_idx:batch_end_idx, frames_start_idx:frames_end_idx],
                            msk=batch.msk[batch_start_idx:batch_end_idx, frames_start_idx:frames_end_idx],
                            meta=batch.meta,
                        )
                    else:
                        chunk_output = dotdict(
                            xyz_map=chunk_preds,
                            xyz_cnf=chunk_conf,
                        )
                        chunk_batch = dotdict(
                            xyz=batch.xyz[batch_start_idx:batch_end_idx, frames_start_idx:frames_end_idx],
                            msk=batch.msk[batch_start_idx:batch_end_idx, frames_start_idx:frames_end_idx],
                            meta=batch.meta,
                        )
                    loss = 0.0
                    scalar_stats = dotdict()
                    image_stats = dotdict()
                    data_ratio = msk[batch_start_idx:batch_end_idx, frames_start_idx:frames_end_idx].sum() / msk_sum
                    chunk_loss = self.loss_fn.compute_loss(chunk_output, chunk_batch, loss, scalar_stats, image_stats) * loss_scaler * data_ratio
                    total_loss += chunk_loss.item()

                    with batch.no_sync_context():
                        chunk_loss.backward()
                    if verbose:
                        print_mem(f"[batch {batch_start_idx}-{batch_end_idx}, chunk {frames_start_idx}-{frames_end_idx}], chunk_loss: {chunk_loss.item()}, scaler: {loss_scaler}, data_ratio: {data_ratio}")

                batch_preds.append(chunk_preds.detach())
                batch_conf.append(chunk_conf.detach())

            all_preds.append(torch.cat(batch_preds, dim=1))
            all_conf.append(torch.cat(batch_conf, dim=1))
        
        all_preds = torch.cat(all_preds, dim=0)
        all_conf = torch.cat(all_conf, dim=0)
        
        if verbose:
            print(f"DPTHeadWithChunkwiseBP <{self.supervisor_type}> total_loss: {total_loss / loss_scaler}, loss_scaler: {loss_scaler}")

        # xiaoyang: a trick to ensure final loss will bp to dpt_head, pls do not remove this        
        auxiliary_loss = images.new_zeros(())
        for k, v in self.named_parameters():
            if v.requires_grad:
                auxiliary_loss += v.mean() * 0.0
        for k, v in detached_aggregated_tokens_list.items():
            if v.requires_grad and v.grad is not None:
                target_v = aggregated_tokens_list[k]
                target_grad = v.grad.clone()
                v.grad = None
                auxiliary_loss += (target_v * target_grad).sum() / loss_scaler
        batch.chunkwise_bp_loss = batch.get("chunkwise_bp_loss", 0.0) + auxiliary_loss
        
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
