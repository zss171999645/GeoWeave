from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import torch


def _stack_view_tensors(views: Sequence[Mapping[str, Any]], key: str) -> torch.Tensor:
    missing = [idx for idx, view in enumerate(views) if key not in view or view[key] is None]
    if missing:
        raise KeyError(f"Missing {key!r} in views {missing}")
    return torch.stack([view[key] for view in views], dim=1)


def _stack_z_far(views: Sequence[Mapping[str, Any]], depths: torch.Tensor) -> torch.Tensor:
    values = []
    for view in views:
        value = view.get("z_far", 0.0)
        if torch.is_tensor(value):
            value = value.to(device=depths.device, dtype=depths.dtype)
            if value.ndim == 0:
                value = value.expand(depths.shape[0])
            values.append(value.reshape(depths.shape[0]))
        else:
            values.append(torch.full((depths.shape[0],), float(value), device=depths.device, dtype=depths.dtype))
    return torch.stack(values, dim=1)


def _compute_masks(
    views: Sequence[Mapping[str, Any]],
    depths: torch.Tensor,
) -> torch.Tensor:
    if all("valid_mask" in view and view["valid_mask"] is not None for view in views):
        masks = _stack_view_tensors(views, "valid_mask").bool()
        return masks & torch.isfinite(depths) & (depths > 0)

    masks = torch.isfinite(depths) & (depths > 0)
    z_far = _stack_z_far(views, depths)
    return masks & ((z_far <= 0)[:, :, None, None] | (depths < z_far[:, :, None, None]))


def _points_from_depth_camera(
    views: Sequence[Mapping[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    depths = _stack_view_tensors(views, "depthmap").float()
    intrinsics = _stack_view_tensors(views, "camera_intrinsics").to(device=depths.device, dtype=depths.dtype)
    poses = _stack_view_tensors(views, "camera_pose").to(device=depths.device, dtype=depths.dtype)
    masks = _compute_masks(views, depths)

    bsz, num_views, height, width = depths.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=depths.device, dtype=depths.dtype),
        torch.arange(width, device=depths.device, dtype=depths.dtype),
        indexing="ij",
    )
    x = x.view(1, 1, height, width)
    y = y.view(1, 1, height, width)

    fx = intrinsics[..., 0, 0].view(bsz, num_views, 1, 1).clamp_min(1e-12)
    fy = intrinsics[..., 1, 1].view(bsz, num_views, 1, 1).clamp_min(1e-12)
    cx = intrinsics[..., 0, 2].view(bsz, num_views, 1, 1)
    cy = intrinsics[..., 1, 2].view(bsz, num_views, 1, 1)

    x_cam = (x - cx) * depths / fx
    y_cam = (y - cy) * depths / fy
    points_cam = torch.stack((x_cam, y_cam, depths), dim=-1)

    rotation = poses[..., :3, :3]
    translation = poses[..., :3, 3]
    points_world = torch.einsum("bnij,bnhwj->bnhwi", rotation, points_cam) + translation[:, :, None, None, :]
    masks = masks & torch.isfinite(points_world).all(dim=-1)
    points_world = torch.where(masks[..., None], points_world, torch.zeros_like(points_world))
    return points_world, masks


def stack_gt_points_and_masks(
    views: Sequence[Mapping[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if all("pts3d" in view and view["pts3d"] is not None for view in views):
        points = _stack_view_tensors(views, "pts3d")
        masks = _stack_view_tensors(views, "valid_mask").bool()
        return points, masks
    return _points_from_depth_camera(views)
