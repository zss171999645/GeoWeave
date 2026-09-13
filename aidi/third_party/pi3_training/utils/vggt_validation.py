from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch

from pi3.utils.gt_points import stack_gt_points_and_masks
import torch.distributed as dist


EXPECTED_VGGT_STYLE_TAGS = (
    "cam:pose_auc_30",
    "dpt:abs_rel",
    "xyz:rmse",
    "training:loss",
)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().float().mean().item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _make_eye(batch: int, num_views: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(4, device=device, dtype=dtype).reshape(1, 1, 4, 4).repeat(batch, num_views, 1, 1)


def _invert_se3(poses: torch.Tensor) -> torch.Tensor:
    inv = torch.eye(4, device=poses.device, dtype=poses.dtype).expand_as(poses).clone()
    rot = poses[..., :3, :3]
    trans = poses[..., :3, 3:4]
    inv[..., :3, :3] = rot.transpose(-1, -2)
    inv[..., :3, 3:4] = -rot.transpose(-1, -2) @ trans
    return inv


def _homogenize(points: torch.Tensor) -> torch.Tensor:
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def _sample_masked_points(
    *values: torch.Tensor,
    mask: torch.Tensor,
    max_points: int,
) -> List[torch.Tensor]:
    mask_shape = tuple(mask.shape)
    flat_size = int(np.prod(mask_shape))
    flat_mask = mask.reshape(-1).bool()
    idx = torch.nonzero(flat_mask, as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        empty = []
        for value in values:
            trailing = tuple(value.shape[len(mask_shape):])
            empty.append(value.reshape(flat_size, *trailing)[:0] if trailing else value.reshape(-1)[:0])
        return empty
    if max_points > 0 and idx.numel() > max_points:
        sample_idx = torch.linspace(0, idx.numel() - 1, max_points, device=idx.device).round().long()
        idx = idx[sample_idx]
    sampled = []
    for value in values:
        trailing = tuple(value.shape[len(mask_shape):])
        if not trailing:
            flat = value.reshape(-1)
        else:
            flat = value.reshape(flat_size, *trailing)
        sampled.append(flat[idx])
    return sampled


def _fit_scale_shift(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred = pred.float()
    target = target.float()
    if pred.numel() == 0:
        return pred.new_tensor(1.0), pred.new_tensor(0.0)
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    denom = (pred_centered * pred_centered).mean().clamp_min(1e-8)
    scale = ((pred_centered * target_centered).mean() / denom).clamp(-100.0, 100.0)
    shift = target_mean - scale * pred_mean
    return scale, shift


def _depth_metrics_for_view(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    mask: torch.Tensor,
    max_points: int,
) -> Dict[str, float]:
    pred, target = _sample_masked_points(
        pred_depth,
        target_depth,
        mask=mask,
        max_points=max_points,
    )
    if pred.numel() == 0:
        return {}
    pred = pred.float().reshape(-1)
    target = target.float().reshape(-1)
    scale, shift = _fit_scale_shift(pred, target)
    pred = scale * pred + shift

    eps = 1e-5
    rel = torch.maximum(target / (pred + eps), pred / (target + eps))
    abs_err = (pred - target).abs()
    inv_err = 1.0 / (pred + 1e-8) - 1.0 / (target + 1e-8)

    def interval_mean(low: float, high: float) -> float:
        interval = (target > low) & (target <= high)
        if not interval.any():
            return 0.0
        return float(abs_err[interval].mean().item())

    return {
        "t11": float((rel < 1.1).float().mean().item()),
        "d05": float((rel < 1.25**0.5).float().mean().item()),
        "d1": float((rel < 1.25).float().mean().item()),
        "d2": float((rel < 1.25**2).float().mean().item()),
        "d3": float((rel < 1.25**3).float().mean().item()),
        "l1": float(abs_err.mean().item()),
        "rmse": float(torch.sqrt(((pred - target) ** 2).mean()).item()),
        "irmse": float(torch.sqrt((inv_err**2).mean()).item()),
        "imae": float(inv_err.abs().mean().item()),
        "abs_rel": float((abs_err / (target.abs() + eps)).mean().item()),
        "sq_rel": float((((pred - target) ** 2) / (target.abs() + eps)).mean().item()),
        "i00_02": interval_mean(0.0, 2.0),
        "i00_05": interval_mean(0.0, 5.0),
        "i00_10": interval_mean(0.0, 10.0),
    }


def _umeyama_align(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    src = src.float()
    dst = dst.float()
    if src.shape[0] < 3:
        return src
    src_mean = src.mean(dim=0, keepdim=True)
    dst_mean = dst.mean(dim=0, keepdim=True)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = src_centered.transpose(0, 1) @ dst_centered / max(int(src.shape[0]), 1)
    try:
        u, s, vh = torch.linalg.svd(cov, full_matrices=True)
    except RuntimeError:
        return src_centered + dst_mean
    rot = vh.transpose(-1, -2) @ u.transpose(-1, -2)
    if torch.det(rot) < 0:
        vh = vh.clone()
        vh[-1] *= -1
        rot = vh.transpose(-1, -2) @ u.transpose(-1, -2)
    var = (src_centered * src_centered).sum(dim=1).mean().clamp_min(1e-8)
    scale = s.sum() / var
    return scale * (src_centered @ rot.transpose(0, 1)) + dst_mean


def _xyz_metrics_for_view(
    pred_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    mask: torch.Tensor,
    max_points: int,
) -> Dict[str, float]:
    pred, target = _sample_masked_points(
        pred_xyz,
        target_xyz,
        mask=mask,
        max_points=max_points,
    )
    if pred.numel() == 0:
        return {}
    pred = _umeyama_align(pred, target)
    diff = pred - target.float()
    eps = 1e-5
    return {
        "l1": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt((diff**2).mean()).item()),
        "abs_rel": float((diff.abs() / (target.float().abs() + eps)).mean().item()),
    }


def _rotation_angle_deg(rot_pred: torch.Tensor, rot_gt: torch.Tensor) -> torch.Tensor:
    residual = rot_pred.transpose(-1, -2) @ rot_gt
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(dim=-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.rad2deg(torch.acos(cos))


def _translation_angle_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_norm = pred.norm(dim=-1, keepdim=True)
    target_norm = target.norm(dim=-1, keepdim=True)
    valid = (pred_norm[..., 0] > 1e-12) & (target_norm[..., 0] > 1e-12)
    pred_dir = pred / pred_norm.clamp_min(1e-12)
    target_dir = target / target_norm.clamp_min(1e-12)
    cos = (pred_dir * target_dir).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cos))
    angle = torch.minimum(angle, (180.0 - angle).abs())
    return torch.where(valid, angle, torch.full_like(angle, 1e6))


def _pose_auc(rot_err: torch.Tensor, trans_err: torch.Tensor, threshold: int) -> float:
    ths = torch.arange(1, threshold + 1, device=rot_err.device, dtype=rot_err.dtype)
    rra = (rot_err[:, None] < ths[None]).float().mean(dim=0)
    rta = (trans_err[:, None] < ths[None]).float().mean(dim=0)
    return float(torch.minimum(rra, rta).mean().item())


def _camera_metrics(gt_c2w: torch.Tensor, pred_c2w: torch.Tensor) -> Dict[str, float]:
    bsz, num_views = gt_c2w.shape[:2]
    if num_views < 2:
        return {}

    gt_w2c = _invert_se3(gt_c2w)
    pred_w2c = _invert_se3(pred_c2w)
    pair_idx = torch.combinations(torch.arange(num_views, device=gt_c2w.device), 2, with_replacement=False)
    i, j = pair_idx[:, 0], pair_idx[:, 1]

    gt_rel = gt_w2c[:, i] @ gt_c2w[:, j]
    pred_rel = pred_w2c[:, i] @ pred_c2w[:, j]
    rot_err = _rotation_angle_deg(pred_rel[..., :3, :3].reshape(-1, 3, 3), gt_rel[..., :3, :3].reshape(-1, 3, 3))
    trans_err = _translation_angle_deg(pred_rel[..., :3, 3].reshape(-1, 3), gt_rel[..., :3, 3].reshape(-1, 3))

    metrics: Dict[str, float] = {}
    for threshold in (1, 3, 5, 15):
        metrics[f"rotation_accuracy_{threshold:02d}"] = float((rot_err < threshold).float().mean().item())
        metrics[f"translation_accuracy_{threshold:02d}"] = float((trans_err < threshold).float().mean().item())
    for threshold in (1, 3, 5, 10, 20, 30):
        metrics[f"pose_auc_{threshold:02d}"] = _pose_auc(rot_err, trans_err, threshold)

    pred_norm = pred_w2c[:, 1:, :3, 3].norm(dim=-1).clamp_min(1e-12)
    gt_norm = gt_w2c[:, 1:, :3, 3].norm(dim=-1)
    metrics["translation_scale"] = float((gt_norm / pred_norm).mean().item()) if bsz and num_views > 1 else 0.0
    return metrics


def _prepare_pi3_geometry(pred: Mapping[str, torch.Tensor], batch: List[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
    gt_pts, masks = stack_gt_points_and_masks(batch)
    poses = torch.stack([view["camera_pose"] for view in batch], dim=1)
    bsz, num_views = gt_pts.shape[:2]

    w2c_target = _invert_se3(poses[:, 0])
    gt_global = (w2c_target[:, None, None, None] @ _homogenize(gt_pts)[..., None]).squeeze(-1)[..., :3]
    gt_poses = w2c_target[:, None] @ poses

    valid_batch = masks.sum(dim=(-1, -2, -3)) > 0
    gt_norm = torch.ones((bsz,), device=gt_pts.device, dtype=gt_pts.dtype)
    if valid_batch.any():
        valid_gt = gt_global[valid_batch].clone()
        valid_masks = masks[valid_batch]
        valid_gt[~valid_masks] = 0
        distances = valid_gt.reshape(valid_gt.shape[0], num_views, -1, 3).norm(dim=-1)
        gt_norm[valid_batch] = (
            distances.sum(dim=(-1, -2))
            / (valid_masks.float().sum(dim=(-1, -2, -3)) + 1e-8)
        ).clamp_min(1e-8)
        gt_global[valid_batch] = gt_global[valid_batch] / gt_norm[valid_batch, None, None, None, None]
        gt_poses[valid_batch, ..., :3, 3] /= gt_norm[valid_batch, None, None]

    gt_w2c = _invert_se3(gt_poses)
    gt_local = (gt_w2c[:, :, None, None] @ _homogenize(gt_global)[..., None]).squeeze(-1)[..., :3]

    pred_local = pred["local_points"].detach()
    pred_poses = pred.get("camera_poses")
    if pred_poses is None:
        pred_poses = _make_eye(bsz, num_views, pred_local.device, pred_local.dtype)
    else:
        pred_poses = pred_poses.detach()

    pred_norm = torch.ones((bsz,), device=pred_local.device, dtype=pred_local.dtype)
    valid_pred = masks.sum(dim=(-1, -2, -3)) > 0
    if valid_pred.any():
        masked = pred_local[valid_pred].clone()
        valid_masks = masks[valid_pred]
        masked[~valid_masks] = 0
        distances = masked.reshape(masked.shape[0], num_views, -1, 3).norm(dim=-1)
        pred_norm[valid_pred] = (
            distances.sum(dim=(-1, -2))
            / (valid_masks.float().sum(dim=(-1, -2, -3)) + 1e-8)
        ).clamp_min(1e-8)

    pred_local = pred_local / pred_norm[:, None, None, None, None]
    pred_poses = pred_poses.clone()
    pred_poses[..., :3, 3] /= pred_norm[:, None, None]
    pred_global = (pred_poses[:, :, None, None] @ _homogenize(pred_local)[..., None]).squeeze(-1)[..., :3]

    return {
        "gt_local": gt_local.detach(),
        "gt_global": gt_global.detach(),
        "gt_poses": gt_poses.detach(),
        "pred_local": pred_local.detach(),
        "pred_global": pred_global.detach(),
        "pred_poses": pred_poses.detach(),
        "mask": masks.detach(),
    }


class VggtStylePi3MetricAccumulator:
    def __init__(self, cfg: Any = None):
        self.cfg = cfg or {}
        self.enabled = _truthy(_cfg_get(self.cfg, "enabled", False))
        self.compute_cam = _truthy(_cfg_get(self.cfg, "compute_cam", True))
        self.compute_dpt = _truthy(_cfg_get(self.cfg, "compute_dpt", True))
        self.compute_xyz = _truthy(_cfg_get(self.cfg, "compute_xyz", True))
        self.max_points_per_view = int(_cfg_get(self.cfg, "max_points_per_view", 65536) or 0)
        self.values: Dict[str, List[float]] = defaultdict(list)
        self.sample_values: Dict[str, Dict[str, float]] = defaultdict(dict)

    @staticmethod
    def _first_collated_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            value = value.reshape(-1)[0]
            return value.item() if value.ndim == 0 else value
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            return value.reshape(-1)[0].item()
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            return value[0]
        return value

    @classmethod
    def _sample_id_from_batch(cls, batch: List[Mapping[str, Any]], batch_size: int) -> Optional[str]:
        if batch_size != 1 or not batch:
            return None
        first_view = batch[0]
        sample_id = cls._first_collated_value(first_view.get("sample_id"))
        if sample_id not in (None, ""):
            return str(sample_id)

        dataset = cls._first_collated_value(first_view.get("dataset"))
        label = cls._first_collated_value(first_view.get("label"))
        instance = cls._first_collated_value(first_view.get("instance"))
        if dataset in (None, "") or label in (None, "") or instance in (None, ""):
            return None
        return f"{dataset}/{label}/{instance}"

    @staticmethod
    def _normalize_sample_id(value: Any) -> Optional[str]:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            value = value.reshape(-1)[0].item()
        elif isinstance(value, np.ndarray):
            if value.size != 1:
                return None
            value = value.reshape(-1)[0].item()
        elif isinstance(value, (list, tuple)):
            if len(value) != 1:
                return None
            value = value[0]
        if value in (None, ""):
            return None
        return str(value)

    def update(self, metrics: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        sample_id = self._normalize_sample_id(metrics.get("_sample_id"))
        for key, value in metrics.items():
            key = str(key)
            if key.startswith("_"):
                continue
            scalar = _to_float(value)
            if scalar is not None:
                if sample_id is not None:
                    self.sample_values[key][sample_id] = scalar
                else:
                    self.values[key].append(scalar)

    def add_training_metrics(self, metrics: Dict[str, float], loss_outputs: Mapping[str, Any]) -> Dict[str, float]:
        for key, value in loss_outputs.items():
            scalar = _to_float(value)
            if scalar is not None:
                metrics[f"training:{key}"] = scalar
        return metrics

    def compute_batch_metrics(
        self,
        pred: Mapping[str, torch.Tensor],
        batch: List[Mapping[str, Any]],
    ) -> Dict[str, float]:
        if not self.enabled or "local_points" not in pred:
            return {}

        metrics: Dict[str, float] = {}
        with torch.no_grad():
            geom = _prepare_pi3_geometry(pred, batch)

            if self.compute_cam:
                for key, value in _camera_metrics(geom["gt_poses"], geom["pred_poses"]).items():
                    metrics[f"cam:{key}"] = value

            if self.compute_dpt:
                dpt_values: Dict[str, List[float]] = defaultdict(list)
                pred_depth = geom["pred_local"][..., 2]
                gt_depth = geom["gt_local"][..., 2]
                for b in range(pred_depth.shape[0]):
                    for v in range(pred_depth.shape[1]):
                        view_metrics = _depth_metrics_for_view(
                            pred_depth[b, v],
                            gt_depth[b, v],
                            geom["mask"][b, v],
                            self.max_points_per_view,
                        )
                        for key, value in view_metrics.items():
                            dpt_values[key].append(value)
                for key, values in dpt_values.items():
                    if values:
                        metrics[f"dpt:{key}"] = float(np.mean(values))

            if self.compute_xyz:
                xyz_values: Dict[str, List[float]] = defaultdict(list)
                for b in range(geom["pred_global"].shape[0]):
                    for v in range(geom["pred_global"].shape[1]):
                        view_metrics = _xyz_metrics_for_view(
                            geom["pred_global"][b, v],
                            geom["gt_global"][b, v],
                            geom["mask"][b, v],
                            self.max_points_per_view,
                        )
                        for key, value in view_metrics.items():
                            xyz_values[key].append(value)
                for key, values in xyz_values.items():
                    if values:
                        metrics[f"xyz:{key}"] = float(np.mean(values))

            sample_id = self._sample_id_from_batch(batch, int(geom["mask"].shape[0]))
            if sample_id is not None:
                metrics["_sample_id"] = sample_id

        return metrics

    def _distributed_values(self) -> Dict[str, List[float]]:
        def merge_payloads(payloads: Iterable[Mapping[str, Any]]) -> Dict[str, List[float]]:
            merged: Dict[str, List[float]] = defaultdict(list)
            sample_merged: Dict[str, Dict[str, float]] = defaultdict(dict)
            for payload in payloads:
                for key, values in payload.get("values", {}).items():
                    merged[key].extend(values)
                for key, values_by_sample in payload.get("sample_values", {}).items():
                    sample_merged[key].update(values_by_sample)
            for key, values_by_sample in sample_merged.items():
                merged[key].extend(values_by_sample.values())
            return merged

        local_payload = {
            "values": {key: list(values) for key, values in self.values.items()},
            "sample_values": {
                key: dict(values_by_sample)
                for key, values_by_sample in self.sample_values.items()
            },
        }
        if not (dist.is_available() and dist.is_initialized()):
            return merge_payloads([local_payload])
        gathered: List[Dict[str, Any]] = [dict() for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_payload)
        return merge_payloads(gathered)

    def summarize(self) -> Dict[str, float]:
        if not self.enabled:
            return {}
        values = self._distributed_values()
        summary: Dict[str, float] = {}
        for key, vals in values.items():
            arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
            if arr.size == 0:
                continue
            summary[f"{key}_mean"] = float(arr.mean())
            summary[f"{key}_std"] = float(arr.std())
        return summary
