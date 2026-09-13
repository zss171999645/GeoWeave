from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from easyvolcap.utils.base_utils import dotdict


def augment_indexer_state_with_camera(
    state: dotdict,
    official_batch: Optional[dict],
    image_height: int,
    image_width: int,
) -> dotdict:
    merged = dotdict(state)
    if official_batch is None:
        return merged
    extrinsics = official_batch.get("extrinsics", None)
    intrinsics = official_batch.get("intrinsics", None)
    if extrinsics is not None:
        merged.extrinsics = extrinsics
    if intrinsics is not None:
        merged.intrinsics = intrinsics
    depths = official_batch.get("depths", None)
    point_masks = official_batch.get("point_masks", None)
    if depths is not None:
        merged.depths = depths
    if point_masks is not None:
        merged.point_masks = point_masks
    merged.image_height = int(image_height)
    merged.image_width = int(image_width)
    return merged


def build_image_token_indices(
    num_views: int,
    tokens_per_view: int,
    patch_start_idx: int,
    device: torch.device,
    view_indices: Optional[torch.Tensor] = None,
    patch_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    patch_token_count = int(tokens_per_view) - int(patch_start_idx)
    if patch_token_count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if patch_indices is None:
        patch_local = torch.arange(int(patch_start_idx), int(tokens_per_view), dtype=torch.long, device=device)
    else:
        patch_indices = patch_indices.to(device=device, dtype=torch.long).reshape(-1)
        if patch_indices.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if bool(((patch_indices < 0) | (patch_indices >= patch_token_count)).any()):
            raise ValueError(
                f"patch_indices out of range for patch_token_count={patch_token_count}: {patch_indices.tolist()}"
            )
        patch_local = int(patch_start_idx) + patch_indices
    if view_indices is None:
        view_ids = torch.arange(int(num_views), dtype=torch.long, device=device)
    else:
        view_ids = view_indices.to(device=device, dtype=torch.long).reshape(-1)
        if bool(((view_ids < 0) | (view_ids >= int(num_views))).any()):
            raise ValueError(f"view_indices out of range for num_views={num_views}: {view_ids.tolist()}")
    view_offsets = view_ids * int(tokens_per_view)
    return (view_offsets[:, None] + patch_local[None, :]).reshape(-1)


def build_downsampled_patch_indices(
    patch_grid_height: int,
    patch_grid_width: int,
    factor: int,
    device: torch.device,
) -> torch.Tensor:
    patch_count = int(patch_grid_height) * int(patch_grid_width)
    if patch_count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    factor = max(int(factor), 1)
    if factor <= 1:
        return torch.arange(patch_count, dtype=torch.long, device=device)

    row_count = (int(patch_grid_height) + factor - 1) // factor
    col_count = (int(patch_grid_width) + factor - 1) // factor
    offset = factor // 2
    rows = torch.arange(row_count, dtype=torch.long, device=device) * factor + offset
    cols = torch.arange(col_count, dtype=torch.long, device=device) * factor + offset
    rows = rows.clamp(max=int(patch_grid_height) - 1)
    cols = cols.clamp(max=int(patch_grid_width) - 1)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    return (grid_y * int(patch_grid_width) + grid_x).reshape(-1)


def build_patch_token_centers(
    image_height: int,
    image_width: int,
    patch_grid_height: int,
    patch_grid_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ys = (torch.arange(int(patch_grid_height), device=device, dtype=dtype) + 0.5) * (float(image_height) / float(patch_grid_height))
    xs = (torch.arange(int(patch_grid_width), device=device, dtype=dtype) + 0.5) * (float(image_width) / float(patch_grid_width))
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones_like(grid_x)
    return torch.stack([grid_x, grid_y, ones], dim=-1).reshape(-1, 3)


def reshape_epipolar_teacher_probs_to_view_patch_maps(
    probs: torch.Tensor,
    *,
    num_views: int,
    patch_grid_height: int,
    patch_grid_width: int,
) -> torch.Tensor:
    if probs.dim() != 3:
        raise ValueError(f"Expected probs [B,Tq,S], got {tuple(probs.shape)}")
    patch_token_count = int(patch_grid_height) * int(patch_grid_width)
    expected_src = int(num_views) * patch_token_count
    if probs.shape[-1] != expected_src:
        raise ValueError(
            f"Expected src dim {expected_src} for num_views={num_views}, "
            f"patch_grid={patch_grid_height}x{patch_grid_width}, got {probs.shape[-1]}"
        )
    return probs.view(probs.shape[0], probs.shape[1], int(num_views), int(patch_grid_height), int(patch_grid_width))


def pixel_to_patch_token_index(
    *,
    x: float,
    y: float,
    image_width: int,
    image_height: int,
    patch_grid_width: int,
    patch_grid_height: int,
    tokens_per_view: int,
    patch_start_idx: int,
    view_idx: int = 0,
) -> int:
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size: {image_width}x{image_height}")
    if patch_grid_width <= 0 or patch_grid_height <= 0:
        raise ValueError(f"Invalid patch grid: {patch_grid_height}x{patch_grid_width}")
    if tokens_per_view <= patch_start_idx:
        raise ValueError(
            f"tokens_per_view={tokens_per_view} must be greater than patch_start_idx={patch_start_idx}"
        )
    x_clamped = min(max(float(x), 0.0), float(image_width) - 1.0)
    y_clamped = min(max(float(y), 0.0), float(image_height) - 1.0)
    col = min(int(x_clamped * float(patch_grid_width) / float(image_width)), int(patch_grid_width) - 1)
    row = min(int(y_clamped * float(patch_grid_height) / float(image_height)), int(patch_grid_height) - 1)
    patch_local = row * int(patch_grid_width) + col
    return int(view_idx) * int(tokens_per_view) + int(patch_start_idx) + patch_local


def _skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(v[..., 0])
    return torch.stack(
        [
            torch.stack([zeros, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], zeros, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], zeros], dim=-1),
        ],
        dim=-2,
    )


def compute_query_to_source_fundamental_matrices(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    if extrinsics.dim() != 4 or extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Expected extrinsics [B,V,3,4], got {tuple(extrinsics.shape)}")
    if intrinsics.dim() != 4 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsics [B,V,3,3], got {tuple(intrinsics.shape)}")

    extrinsics = extrinsics.to(torch.float32)
    intrinsics = intrinsics.to(torch.float32)
    rot = extrinsics[..., :3]
    trans = extrinsics[..., 3]

    rot_query = rot[:, :, None]
    rot_source = rot[:, None, :]
    rot_rel = torch.matmul(rot_source, rot_query.transpose(-1, -2))

    trans_query = trans[:, :, None, :, None]
    trans_source = trans[:, None, :, :, None]
    trans_rel = trans_source - torch.matmul(rot_rel, trans_query)

    intrinsics_inv = torch.linalg.inv(intrinsics)
    intrinsics_query_inv = intrinsics_inv[:, :, None]
    intrinsics_source_inv_t = intrinsics_inv.transpose(-1, -2)[:, None, :]

    essential = torch.matmul(_skew_symmetric(trans_rel.squeeze(-1)), rot_rel)
    fundamental = torch.matmul(
        intrinsics_source_inv_t,
        torch.matmul(essential, intrinsics_query_inv),
    )
    return fundamental


def _compute_epipolar_geometry(
    query_token_indices: torch.Tensor,
    *,
    num_views: int,
    tokens_per_view: int,
    patch_start_idx: int,
    patch_grid_height: int,
    patch_grid_width: int,
    image_height: int,
    image_width: int,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    exclude_self_view: bool = True,
    source_mask: Optional[torch.Tensor] = None,
    source_view_indices: Optional[torch.Tensor] = None,
    source_patch_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if query_token_indices.dim() == 1:
        query_token_indices = query_token_indices.unsqueeze(0)
    if query_token_indices.dim() != 2:
        raise ValueError(f"Expected query_token_indices [B,Tq], got {tuple(query_token_indices.shape)}")

    device = query_token_indices.device
    dtype = extrinsics.dtype if extrinsics.is_floating_point() else torch.float32
    bsz, tgt_len = query_token_indices.shape
    patch_token_count = int(tokens_per_view) - int(patch_start_idx)
    if patch_token_count <= 0:
        distances = torch.zeros((bsz, tgt_len, 0), device=device, dtype=dtype)
        valid_mask = torch.zeros((bsz, tgt_len, 0), device=device, dtype=torch.bool)
        indices = torch.empty(0, dtype=torch.long, device=device)
        return distances, valid_mask, indices

    if source_view_indices is None:
        active_source_views = torch.arange(int(num_views), dtype=torch.long, device=device)
    else:
        active_source_views = source_view_indices.to(device=device, dtype=torch.long).reshape(-1)
        if bool(((active_source_views < 0) | (active_source_views >= int(num_views))).any()):
            raise ValueError(
                f"source_view_indices out of range for num_views={num_views}: {active_source_views.tolist()}"
            )
    source_view_count = int(active_source_views.numel())
    source_token_indices = build_image_token_indices(
        num_views,
        tokens_per_view,
        patch_start_idx,
        device=device,
        view_indices=active_source_views,
        patch_indices=source_patch_indices,
    )
    patch_centers = build_patch_token_centers(
        image_height=image_height,
        image_width=image_width,
        patch_grid_height=patch_grid_height,
        patch_grid_width=patch_grid_width,
        device=device,
        dtype=dtype,
    )

    local_query_patch = torch.remainder(query_token_indices, int(tokens_per_view)) - int(patch_start_idx)
    if bool((local_query_patch < 0).any()) or bool((local_query_patch >= patch_token_count).any()):
        raise ValueError("query_token_indices must point to image tokens only.")

    query_view_ids = torch.div(query_token_indices, int(tokens_per_view), rounding_mode="floor")
    query_centers = patch_centers[local_query_patch.reshape(-1)].reshape(bsz, tgt_len, 3)

    fundamental = compute_query_to_source_fundamental_matrices(
        extrinsics=extrinsics.to(device=device, dtype=dtype),
        intrinsics=intrinsics.to(device=device, dtype=dtype),
    )

    if source_patch_indices is None:
        source_patch_indices = torch.arange(patch_token_count, dtype=torch.long, device=device)
    else:
        source_patch_indices = source_patch_indices.to(device=device, dtype=torch.long).reshape(-1)
        if source_patch_indices.numel() == 0:
            distances = torch.zeros((bsz, tgt_len, 0), device=device, dtype=dtype)
            valid_mask = torch.zeros((bsz, tgt_len, 0), device=device, dtype=torch.bool)
            return distances, valid_mask, source_token_indices
        if bool(((source_patch_indices < 0) | (source_patch_indices >= patch_token_count)).any()):
            raise ValueError(
                f"source_patch_indices out of range for patch_token_count={patch_token_count}: "
                f"{source_patch_indices.tolist()}"
            )

    source_patch_count = int(source_patch_indices.numel())
    source_patch_centers = patch_centers[source_patch_indices].view(1, 1, 1, source_patch_count, 3)
    source_view_ids = active_source_views.view(1, 1, source_view_count, 1).expand(
        bsz, tgt_len, source_view_count, source_patch_count
    )

    batch_ids = torch.arange(bsz, device=device)[:, None, None]
    source_ids = active_source_views.view(1, 1, source_view_count).expand(bsz, tgt_len, -1)
    query_f = fundamental[
        batch_ids,
        query_view_ids[..., None].expand(-1, -1, source_view_count),
        source_ids,
    ]
    lines = torch.einsum("btvij,btj->btvi", query_f, query_centers.to(dtype=query_f.dtype))
    numer = torch.abs((source_patch_centers * lines.unsqueeze(-2)).sum(dim=-1))
    denom = lines[..., :2].square().sum(dim=-1).sqrt().clamp_min(1e-6).unsqueeze(-1)
    distances = numer / denom
    distances = distances.reshape(bsz, tgt_len, source_view_count * source_patch_count)
    invalid = torch.zeros_like(distances, dtype=torch.bool)
    if exclude_self_view:
        flat_source_view_ids = source_view_ids.reshape(bsz, tgt_len, source_view_count * source_patch_count)
        invalid = invalid | (flat_source_view_ids == query_view_ids[..., None])
    if source_mask is not None:
        if source_mask.shape != distances.shape:
            raise ValueError(f"source_mask shape {tuple(source_mask.shape)} mismatches distances {tuple(distances.shape)}")
        invalid = invalid | (~source_mask.bool())
    return distances, (~invalid), source_token_indices


def build_epipolar_teacher_probs(
    query_token_indices: torch.Tensor,
    *,
    num_views: int,
    tokens_per_view: int,
    patch_start_idx: int,
    patch_grid_height: int,
    patch_grid_width: int,
    image_height: int,
    image_width: int,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    sigma_px: float,
    exclude_self_view: bool = True,
    source_mask: Optional[torch.Tensor] = None,
    source_view_indices: Optional[torch.Tensor] = None,
    source_patch_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if sigma_px <= 0:
        raise ValueError(f"sigma_px must be positive, got {sigma_px}")
    distances, valid_mask, source_token_indices = _compute_epipolar_geometry(
        query_token_indices=query_token_indices,
        num_views=num_views,
        tokens_per_view=tokens_per_view,
        patch_start_idx=patch_start_idx,
        patch_grid_height=patch_grid_height,
        patch_grid_width=patch_grid_width,
        image_height=image_height,
        image_width=image_width,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        exclude_self_view=exclude_self_view,
        source_mask=source_mask,
        source_view_indices=source_view_indices,
        source_patch_indices=source_patch_indices,
    )
    logits = -0.5 * (distances / float(sigma_px)).square()

    fill_value = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~valid_mask, fill_value)
    probs = F.softmax(masked_logits, dim=-1)
    probs = probs.masked_fill(~valid_mask, 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return probs, source_token_indices


def build_epipolar_band_mask(
    query_token_indices: torch.Tensor,
    *,
    num_views: int,
    tokens_per_view: int,
    patch_start_idx: int,
    patch_grid_height: int,
    patch_grid_width: int,
    image_height: int,
    image_width: int,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    band_px: float,
    exclude_self_view: bool = True,
    source_mask: Optional[torch.Tensor] = None,
    source_view_indices: Optional[torch.Tensor] = None,
    source_patch_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if band_px <= 0:
        raise ValueError(f"band_px must be positive, got {band_px}")
    distances, valid_mask, source_token_indices = _compute_epipolar_geometry(
        query_token_indices=query_token_indices,
        num_views=num_views,
        tokens_per_view=tokens_per_view,
        patch_start_idx=patch_start_idx,
        patch_grid_height=patch_grid_height,
        patch_grid_width=patch_grid_width,
        image_height=image_height,
        image_width=image_width,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        exclude_self_view=exclude_self_view,
        source_mask=source_mask,
        source_view_indices=source_view_indices,
        source_patch_indices=source_patch_indices,
    )
    band_mask = (distances <= float(band_px)) & valid_mask
    return band_mask, valid_mask, source_token_indices


def build_depth_reprojection_support_mask(
    query_token_indices: torch.Tensor,
    *,
    num_views: int,
    tokens_per_view: int,
    patch_start_idx: int,
    patch_grid_height: int,
    patch_grid_width: int,
    image_height: int,
    image_width: int,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    depths: torch.Tensor,
    point_masks: Optional[torch.Tensor] = None,
    radius_px: float = 12.0,
    min_depth: float = 1e-4,
    exclude_self_view: bool = True,
    source_mask: Optional[torch.Tensor] = None,
    source_view_indices: Optional[torch.Tensor] = None,
    source_patch_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if radius_px <= 0:
        raise ValueError(f"radius_px must be positive, got {radius_px}")
    if query_token_indices.dim() == 1:
        query_token_indices = query_token_indices.unsqueeze(0)
    if query_token_indices.dim() != 2:
        raise ValueError(f"Expected query_token_indices [B,Tq], got {tuple(query_token_indices.shape)}")
    if depths.dim() != 4:
        raise ValueError(f"Expected depths [B,V,H,W], got {tuple(depths.shape)}")

    device = query_token_indices.device
    dtype = extrinsics.dtype if extrinsics.is_floating_point() else torch.float32
    bsz, tgt_len = query_token_indices.shape
    patch_token_count = int(tokens_per_view) - int(patch_start_idx)
    if patch_token_count <= 0:
        empty = torch.zeros((bsz, tgt_len, 0), device=device, dtype=torch.bool)
        return empty, empty, torch.empty(0, dtype=torch.long, device=device), empty[..., 0]

    if source_view_indices is None:
        active_source_views = torch.arange(int(num_views), dtype=torch.long, device=device)
    else:
        active_source_views = source_view_indices.to(device=device, dtype=torch.long).reshape(-1)
        if bool(((active_source_views < 0) | (active_source_views >= int(num_views))).any()):
            raise ValueError(
                f"source_view_indices out of range for num_views={num_views}: {active_source_views.tolist()}"
            )
    source_view_count = int(active_source_views.numel())
    source_token_indices = build_image_token_indices(
        num_views,
        tokens_per_view,
        patch_start_idx,
        device=device,
        view_indices=active_source_views,
        patch_indices=source_patch_indices,
    )

    if source_patch_indices is None:
        source_patch_indices = torch.arange(patch_token_count, dtype=torch.long, device=device)
    else:
        source_patch_indices = source_patch_indices.to(device=device, dtype=torch.long).reshape(-1)
        if source_patch_indices.numel() == 0:
            empty = torch.zeros((bsz, tgt_len, 0), device=device, dtype=torch.bool)
            return empty, empty, source_token_indices, torch.zeros((bsz, tgt_len), device=device, dtype=torch.bool)
        if bool(((source_patch_indices < 0) | (source_patch_indices >= patch_token_count)).any()):
            raise ValueError(
                f"source_patch_indices out of range for patch_token_count={patch_token_count}: "
                f"{source_patch_indices.tolist()}"
            )
    source_patch_count = int(source_patch_indices.numel())

    patch_centers = build_patch_token_centers(
        image_height=image_height,
        image_width=image_width,
        patch_grid_height=patch_grid_height,
        patch_grid_width=patch_grid_width,
        device=device,
        dtype=dtype,
    )
    local_query_patch = torch.remainder(query_token_indices, int(tokens_per_view)) - int(patch_start_idx)
    if bool((local_query_patch < 0).any()) or bool((local_query_patch >= patch_token_count).any()):
        raise ValueError("query_token_indices must point to image tokens only.")

    query_view_ids = torch.div(query_token_indices, int(tokens_per_view), rounding_mode="floor")
    query_centers = patch_centers[local_query_patch.reshape(-1)].reshape(bsz, tgt_len, 3)
    source_patch_centers = patch_centers[source_patch_indices][..., :2]

    depth_h, depth_w = int(depths.shape[-2]), int(depths.shape[-1])
    sample_x = query_centers[..., 0] * (float(depth_w) / float(image_width))
    sample_y = query_centers[..., 1] * (float(depth_h) / float(image_height))
    sample_x = sample_x.round().long().clamp(min=0, max=max(depth_w - 1, 0))
    sample_y = sample_y.round().long().clamp(min=0, max=max(depth_h - 1, 0))

    depths = depths.to(device=device, dtype=dtype)
    batch_ids = torch.arange(bsz, device=device)[:, None]
    query_depth = depths[batch_ids, query_view_ids, sample_y, sample_x]
    query_depth_valid = torch.isfinite(query_depth) & (query_depth > float(min_depth))
    if point_masks is not None:
        point_masks = point_masks.to(device=device, dtype=torch.bool)
        query_depth_valid = query_depth_valid & point_masks[batch_ids, query_view_ids, sample_y, sample_x]

    intrinsics = intrinsics.to(device=device, dtype=dtype)
    extrinsics = extrinsics.to(device=device, dtype=dtype)
    intrinsics_inv = torch.linalg.inv(intrinsics)
    query_k_inv = intrinsics_inv[batch_ids, query_view_ids]
    query_rays = torch.matmul(query_k_inv, query_centers.unsqueeze(-1)).squeeze(-1)
    query_cam_points = query_rays * query_depth.clamp_min(float(min_depth)).unsqueeze(-1)

    rot = extrinsics[..., :3]
    trans = extrinsics[..., 3]
    query_rot = rot[batch_ids, query_view_ids]
    query_trans = trans[batch_ids, query_view_ids]
    world_points = torch.matmul(
        query_rot.transpose(-1, -2),
        (query_cam_points - query_trans).unsqueeze(-1),
    ).squeeze(-1)

    source_rot = rot[:, active_source_views]
    source_trans = trans[:, active_source_views]
    source_cam_points = torch.einsum("bvij,btj->btvi", source_rot, world_points) + source_trans[:, None]
    source_z = source_cam_points[..., 2]
    source_intrinsics = intrinsics[:, active_source_views]
    source_pixels_h = torch.einsum("bvij,btvj->btvi", source_intrinsics, source_cam_points)
    projected_xy = source_pixels_h[..., :2] / source_pixels_h[..., 2:].clamp_min(float(min_depth))
    projected_inside = (
        query_depth_valid[..., None]
        & torch.isfinite(projected_xy).all(dim=-1)
        & (source_z > float(min_depth))
        & (projected_xy[..., 0] >= 0.0)
        & (projected_xy[..., 0] <= float(image_width) - 1.0)
        & (projected_xy[..., 1] >= 0.0)
        & (projected_xy[..., 1] <= float(image_height) - 1.0)
    )

    source_patch_xy = source_patch_centers.view(1, 1, 1, source_patch_count, 2)
    dist_sq = (source_patch_xy[..., 0] - projected_xy[..., 0].unsqueeze(-1)).square()
    dist_sq = dist_sq + (source_patch_xy[..., 1] - projected_xy[..., 1].unsqueeze(-1)).square()
    support_mask = (dist_sq <= float(radius_px) * float(radius_px)) & projected_inside.unsqueeze(-1)

    source_view_ids = active_source_views.view(1, 1, source_view_count, 1).expand(
        bsz, tgt_len, source_view_count, source_patch_count
    )
    invalid = torch.zeros((bsz, tgt_len, source_view_count, source_patch_count), device=device, dtype=torch.bool)
    if exclude_self_view:
        invalid = invalid | (source_view_ids == query_view_ids[..., None, None])
    valid_mask = (~invalid) & query_depth_valid[..., None, None]
    support_mask = support_mask & valid_mask

    support_mask = support_mask.reshape(bsz, tgt_len, source_view_count * source_patch_count)
    valid_mask = valid_mask.reshape(bsz, tgt_len, source_view_count * source_patch_count)
    if source_mask is not None:
        if source_mask.shape != valid_mask.shape:
            raise ValueError(f"source_mask shape {tuple(source_mask.shape)} mismatches valid_mask {tuple(valid_mask.shape)}")
        source_mask = source_mask.to(device=device, dtype=torch.bool)
        valid_mask = valid_mask & source_mask
        support_mask = support_mask & source_mask

    return support_mask, valid_mask, source_token_indices, query_depth_valid


def compute_epipolar_band_logsumexp_terms(
    student_scores: torch.Tensor,
    band_mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if student_scores.dim() != 3:
        raise ValueError(f"Expected student_scores [B,Tq,S], got {tuple(student_scores.shape)}")
    if band_mask.shape != student_scores.shape:
        raise ValueError(f"band_mask shape {tuple(band_mask.shape)} mismatches scores {tuple(student_scores.shape)}")
    if valid_mask is None:
        valid_mask = torch.ones_like(band_mask, dtype=torch.bool)
    elif valid_mask.shape != student_scores.shape:
        raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} mismatches scores {tuple(student_scores.shape)}")

    fill_value = torch.finfo(student_scores.dtype).min
    masked_scores = student_scores.masked_fill(~valid_mask, fill_value)
    band_scores = masked_scores.masked_fill(~band_mask, fill_value)
    total_lse = torch.logsumexp(masked_scores, dim=-1).to(torch.float32)
    band_lse = torch.logsumexp(band_scores, dim=-1).to(torch.float32)
    valid_queries = valid_mask.any(dim=-1) & band_mask.any(dim=-1)
    return total_lse, band_lse, valid_queries


def compute_epipolar_band_loss_from_scores(
    student_scores: torch.Tensor,
    band_mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    total_lse, band_lse, valid_queries = compute_epipolar_band_logsumexp_terms(
        student_scores=student_scores,
        band_mask=band_mask,
        valid_mask=valid_mask,
    )
    if not bool(valid_queries.any()):
        return student_scores.new_tensor(0.0)
    return (total_lse[valid_queries] - band_lse[valid_queries]).mean()
