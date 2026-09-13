from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping

import numpy as np


def build_point_cloud_labels(primary_label: str = "Primary", secondary_label: str = "Secondary") -> Dict[str, str]:
    return {
        "show_gt_xyz": "GT XYZ",
        "show_gt_reprojected": "GT Reprojected",
        "show_primary_xyz": f"{primary_label} XYZ",
        "show_primary_reprojected": f"{primary_label} Reprojected",
        "show_secondary_xyz": f"{secondary_label} XYZ",
        "show_secondary_reprojected": f"{secondary_label} Reprojected",
    }


def build_camera_labels(primary_label: str = "Primary", secondary_label: str = "Secondary") -> Dict[str, str]:
    return {
        "show_gt_camera": "Show GT Camera",
        "show_primary_camera": f"Show {primary_label} Camera",
        "show_secondary_camera": f"Show {secondary_label} Camera",
    }


def merge_secondary_visualization_data(
    primary_data: MutableMapping,
    secondary_data: Mapping,
) -> MutableMapping:
    for target_key, source_key in (
        ("depth_preds2", "depth_preds"),
        ("c2ws_pred2", "c2ws_pred"),
        ("Ks_pred2", "Ks_pred"),
        ("xyzs2", "xyzs"),
    ):
        if source_key in secondary_data:
            primary_data[target_key] = secondary_data[source_key]
    return primary_data


def get_meta_value(meta: Mapping[str, Any], key: str, default: Any) -> Any:
    try:
        return meta[key]
    except Exception:
        return default


def compute_valid_depth_ratio(depth: np.ndarray, max_depth: float = 100.0) -> float:
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape (H, W), got {depth.shape}")
    valid_mask = np.isfinite(depth) & (depth > 0) & (depth < max_depth)
    return float(valid_mask.mean())


def flatten_depth_samples(
    depth: np.ndarray,
    rgbs: np.ndarray,
    downsample: int = 2,
    preserve_valid: bool = False,
    max_depth: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(depth)
    rgbs = np.asarray(rgbs)
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape (H, W), got {depth.shape}")
    if rgbs.shape != depth.shape + (3,):
        raise ValueError(f"Expected rgbs shape {depth.shape + (3,)}, got {rgbs.shape}")
    if downsample < 1:
        raise ValueError(f"downsample must be >= 1, got {downsample}")

    valid_mask = np.isfinite(depth) & (depth > 0) & (depth < max_depth)
    if preserve_valid:
        v, u = np.nonzero(valid_mask)
        z = depth[v, u]
        colors = rgbs[v, u]
    else:
        depth_ds = depth[::downsample, ::downsample]
        rgbs_ds = rgbs[::downsample, ::downsample]
        valid_mask_ds = np.isfinite(depth_ds) & (depth_ds > 0) & (depth_ds < max_depth)
        v_ds, u_ds = np.nonzero(valid_mask_ds)
        u = u_ds.astype(np.float32) * downsample
        v = v_ds.astype(np.float32) * downsample
        z = depth_ds[v_ds, u_ds]
        colors = rgbs_ds[v_ds, u_ds]
        return u.astype(np.float32), v.astype(np.float32), z.astype(np.float32), colors.astype(np.float32)

    return u.astype(np.float32), v.astype(np.float32), z.astype(np.float32), colors.astype(np.float32)


def normalize_xyz_point_cloud(xyzs: np.ndarray, num_views: int, height: int, width: int) -> np.ndarray:
    xyzs = np.asarray(xyzs)
    if xyzs.shape == (num_views, height, width, 3):
        return xyzs
    if xyzs.shape == (num_views, height * width, 3):
        return xyzs.reshape(num_views, height, width, 3)
    raise ValueError(
        f"Expected xyzs shape {(num_views, height, width, 3)} or {(num_views, height * width, 3)}, got {xyzs.shape}"
    )


def flatten_xyz_point_cloud(
    xyzs: np.ndarray,
    rgbs: np.ndarray,
    downsample: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    xyzs = np.asarray(xyzs)
    rgbs = np.asarray(rgbs)
    if xyzs.ndim != 3 or xyzs.shape[-1] != 3:
        raise ValueError(f"Expected xyzs shape (H, W, 3), got {xyzs.shape}")
    if rgbs.shape != xyzs.shape:
        raise ValueError(f"Expected rgbs shape {xyzs.shape}, got {rgbs.shape}")
    if downsample < 1:
        raise ValueError(f"downsample must be >= 1, got {downsample}")

    xyzs_ds = xyzs[::downsample, ::downsample]
    rgbs_ds = rgbs[::downsample, ::downsample]
    points = xyzs_ds.reshape(-1, 3)
    colors = rgbs_ds.reshape(-1, 3)
    valid_mask = np.isfinite(points).all(axis=1)
    return points[valid_mask], colors[valid_mask]
