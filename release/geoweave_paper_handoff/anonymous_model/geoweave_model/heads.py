import torch
import torch.nn.functional as F
from torch import nn


class GeometryHeads(nn.Module):
    def __init__(self, dim: int, patch_start_idx: int):
        super().__init__()
        self.patch_start_idx = patch_start_idx
        self.point_head = nn.Linear(dim, 3)
        self.conf_head = nn.Linear(dim, 1)
        self.camera_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 9),
        )

    def _dense_map(self, values: torch.Tensor, grid_hw, image_hw):
        grid_h, grid_w = grid_hw
        bsz, views, _, channels = values.shape
        values = values.transpose(2, 3).reshape(bsz * views, channels, grid_h, grid_w)
        values = F.interpolate(values, size=image_hw, mode="bilinear", align_corners=False)
        return values.reshape(bsz, views, channels, *image_hw).permute(0, 1, 3, 4, 2)

    def forward(self, tokens: torch.Tensor, grid_hw, image_hw):
        patch_tokens = tokens[:, :, self.patch_start_idx:]
        world_points = self._dense_map(self.point_head(patch_tokens), grid_hw, image_hw)
        confidence = self._dense_map(self.conf_head(patch_tokens), grid_hw, image_hw)

        summary = patch_tokens.mean(dim=2)
        pose_delta = self.camera_head(summary)
        bsz, views = pose_delta.shape[:2]
        camera_poses = torch.eye(4, device=tokens.device, dtype=tokens.dtype).view(1, 1, 4, 4).repeat(bsz, views, 1, 1)
        camera_poses[:, :, :3, 3] = pose_delta[:, :, :3]
        camera_poses[:, :, 0, 0] = 1.0 + 0.01 * pose_delta[:, :, 3]
        camera_poses[:, :, 1, 1] = 1.0 + 0.01 * pose_delta[:, :, 4]
        camera_poses[:, :, 2, 2] = 1.0 + 0.01 * pose_delta[:, :, 5]

        return {
            "world_points": world_points,
            "confidence": confidence,
            "camera_poses": camera_poses,
        }
