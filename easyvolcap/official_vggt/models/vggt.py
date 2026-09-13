# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from easyvolcap.official_vggt.models.aggregator import Aggregator
from easyvolcap.official_vggt.heads.camera_head import CameraHead
from easyvolcap.official_vggt.heads.dpt_head import DPTHead, DPTHeadWithChunkwiseBP
from easyvolcap.official_vggt.heads.track_head import TrackHead
from easyvolcap.utils.base_utils import dotdict


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True,
                 head_autocast_enabled=False,
                 head_autocast_dtype="float16",
                 indexer_cfg: dict = None,
                 memory_cfg: dict = None,
                 aggregator_cfg: dict = None,
                 depth_head_cfg: dict = None,
                 point_head_cfg: dict = None):
        super().__init__()
        self.head_autocast_enabled = bool(head_autocast_enabled)
        self.head_autocast_dtype = self._resolve_autocast_dtype(head_autocast_dtype)

        mem_cfg = dotdict(memory_cfg or {})
        agg_cfg = dotdict(aggregator_cfg or {})
        agg_extra = {
            k: v for k, v in agg_cfg.items()
            if k not in (
                "fused_attn",
                "use_checkpoint",
                "patch_embed_chunk_size",
                "output_list_cfg",
            )
        }
        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            indexer_cfg=indexer_cfg,
            fused_attn=agg_cfg.get("fused_attn", mem_cfg.get("fused_attn", True)),
            use_checkpoint=agg_cfg.get("use_checkpoint", mem_cfg.get("use_checkpoint", True)),
            patch_embed_chunk_size=agg_cfg.get("patch_embed_chunk_size", mem_cfg.get("patch_embed_chunk_size", 0)),
            output_list_cfg=agg_cfg.get("output_list_cfg", mem_cfg.get("output_list_cfg", {})),
            **agg_extra,
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None

        point_cfg = dotdict(point_head_cfg or {})
        point_type = point_cfg.pop("type", "DPTHead")
        if point_type == "DPTHead":
            point_cls = DPTHead
        elif point_type == "DPTHeadWithChunkwiseBP":
            point_cls = DPTHeadWithChunkwiseBP
        else:
            raise ValueError(f"Unsupported point_head type: {point_type}")
        if enable_point:
            if point_cls is DPTHeadWithChunkwiseBP:
                point_cfg.setdefault("loss_type", "point")
            point_defaults = dict(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
            point_defaults.update(point_cfg)
            self.point_head = point_cls(**point_defaults)
        else:
            self.point_head = None

        depth_cfg = dotdict(depth_head_cfg or {})
        depth_type = depth_cfg.pop("type", "DPTHead")
        if depth_type == "DPTHead":
            depth_cls = DPTHead
        elif depth_type == "DPTHeadWithChunkwiseBP":
            depth_cls = DPTHeadWithChunkwiseBP
        else:
            raise ValueError(f"Unsupported depth_head type: {depth_type}")
        if enable_depth:
            if depth_cls is DPTHeadWithChunkwiseBP:
                depth_cfg.setdefault("loss_type", "depth")
            depth_defaults = dict(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
            depth_defaults.update(depth_cfg)
            self.depth_head = depth_cls(**depth_defaults)
        else:
            self.depth_head = None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    @staticmethod
    def _resolve_autocast_dtype(dtype):
        if isinstance(dtype, torch.dtype):
            return dtype
        text = str(dtype).strip().lower()
        if text in ("float16", "fp16", "half", "torch.float16"):
            return torch.float16
        if text in ("bfloat16", "bf16", "torch.bfloat16"):
            return torch.bfloat16
        if text in ("float32", "fp32", "torch.float32"):
            return torch.float32
        raise ValueError(f"Unsupported head_autocast_dtype={dtype!r}")

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
        batch=None,
        official_batch: dict = None,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=self.head_autocast_enabled, dtype=self.head_autocast_dtype):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                if getattr(self.depth_head, "chunkwise_bp", False) and batch is not None and official_batch is not None and self.training:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens_list,
                        images=images,
                        patch_start_idx=patch_start_idx,
                        batch=batch,
                        official_batch=official_batch,
                    )
                    if getattr(self.depth_head, "last_loss_dict", None) is not None:
                        predictions["depth_loss_dict"] = self.depth_head.last_loss_dict
                else:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                    )
                if depth is not None:
                    predictions["depth"] = depth
                if depth_conf is not None:
                    predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                if getattr(self.point_head, "chunkwise_bp", False) and batch is not None and official_batch is not None and self.training:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens_list,
                        images=images,
                        patch_start_idx=patch_start_idx,
                        batch=batch,
                        official_batch=official_batch,
                    )
                    if getattr(self.point_head, "last_loss_dict", None) is not None:
                        predictions["point_loss_dict"] = self.point_head.last_loss_dict
                else:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                    )
                if pts3d is not None:
                    predictions["world_points"] = pts3d
                if pts3d_conf is not None:
                    predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions
