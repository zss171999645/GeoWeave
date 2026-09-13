"""
Given images, output scalar metrics on CPU
Used for evaluation. For training, please check out loss_utils
"""

import os
import torch
import numpy as np
from pytorch3d.ops import knn_points
from scipy.spatial import cKDTree as KDTree
try:
    from evo.core.trajectory import PosePath3D
    from evo.core.metrics import PoseRelation, RPE
    from evo.core.units import Unit
    from evo.core.metrics import id_pairs_from_delta
    _HAS_EVO = True
    _EVO_IMPORT_ERROR = None
except Exception as _exc:
    PosePath3D = None
    PoseRelation = None
    RPE = None
    Unit = None
    id_pairs_from_delta = None
    _HAS_EVO = False
    _EVO_IMPORT_ERROR = _exc

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.loss_utils import mse as compute_mse
from easyvolcap.utils.loss_utils import lpips as compute_lpips
from skimage.metrics import structural_similarity as compare_ssim
from easyvolcap.utils.loss_utils import align_dpt_scale_shift_ransac, align_xyz_scale_shift_roe
from easyvolcap.utils.cam_utils import (
    decode_camera_params,
    camera_to_relative_degree,
    camera_to_relative_degree_centers,
    calculate_auc,
    calculate_auc_np,
    compute_translation_angle,
    compute_rotation_angle,
)
from easyvolcap.utils.saturnv_labelmap import DYNAMIC_ELEMENT_ID
from enum import Enum, auto

_EVO_WARNED = False

def _require_evo(func_name: str) -> bool:
    global _EVO_WARNED
    if _HAS_EVO:
        return True
    if not _EVO_WARNED:
        log(
            yellow(
                f"[metric_utils] evo not available, skip {func_name}; "
                f"install evo to enable camera trajectory metrics ({type(_EVO_IMPORT_ERROR).__name__})"
            )
        )
        _EVO_WARNED = True
    return False

def depth_interval_metric(depth_est: torch.Tensor, depth_gt: torch.Tensor, mask: torch.Tensor, interval: List[float]):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    error = torch.abs(depth_est - depth_gt)
    interval_error = error[(depth_gt > interval[0]) & (depth_gt <= interval[1])]
    if interval_error.shape[0] == 0:
        return torch.tensor(0, device=error.device, dtype=depth_gt.dtype)
    return torch.mean(interval_error)


def depth_absrel_metric(depth_est: torch.Tensor, depth_gt: torch.Tensor, mask: torch.Tensor):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    abs_rel_error = torch.abs(depth_est - depth_gt) / (torch.abs(depth_gt) + 1e-6)
    if abs_rel_error.shape[0] == 0:
        return torch.tensor(0, device=abs_rel_error.device, dtype=depth_gt.dtype)
    return torch.mean(abs_rel_error)


def threshold_interval_metric(depth_est: torch.Tensor, depth_gt: torch.Tensor, mask: torch.Tensor, thres, interval=[1e-6, 1e6]):
    assert isinstance(thres, (int, float))
    interval_mask = (depth_gt >= interval[0]) & (depth_gt <= interval[1])
    mask = mask & interval_mask
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt)
    err_mask = errors > thres
    if err_mask.shape[0] == 0:
        return torch.tensor(0, device=err_mask.device, dtype=depth_gt.dtype)
    return torch.mean(err_mask.float())


def target_depth_metric(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    batch: dotdict = None,
    fixscale: bool = True,
    batching: bool = True,
):
    # Align the predicted depth to the ground truth depth
    prediction = prediction.clone()
    target = target.clone()
    mask = mask.clone()

    if batching:
        prediction = prediction[0].unsqueeze(0) # Only use the first frame
        target = target[0].unsqueeze(0)
        mask = mask[0].unsqueeze(0)
        
        target[mask == 0] = 0.0
        prediction[mask == 0] = 0.0
        
        target = target * batch.scale
        if fixscale:
            prediction = prediction * batch.scale
        else:
            prediction = np.stack([
                align_dpt_scale_shift_ransac(p, t, m)
                    for p, t, m in zip(prediction, target, mask)
            ], axis=0)
    else:
        target[mask == 0] = 0.0
        prediction[mask == 0] = 0.0
        target = target * batch.scale
        if fixscale:
            prediction = prediction * batch.scale
        else:
            prediction = align_dpt_scale_shift_ransac(
                prediction, target, mask
            )

    prediction = torch.tensor(prediction, device=target.device)
    prediction = prediction.squeeze()
    target = target.squeeze()
    mask = mask.squeeze()

    di = 0.1
    scalar_outputs = dotdict()
    scalar_outputs = {
        "depth_interval0-2m_error": depth_interval_metric(prediction, target, mask, [0, 2]),
        "depth_interval0-5m_error": depth_interval_metric(prediction, target, mask, [0, 5]),
        "depth_interval5-10m_error": depth_interval_metric(prediction, target, mask, [5, 10]),
        "depth_interval10-20m_error": depth_interval_metric(prediction, target, mask, [10, 20]),
        "depth_interval20-40m_error": depth_interval_metric(prediction, target, mask, [20, 40]),
        "depth_interval40-60m_error": depth_interval_metric(prediction, target, mask, [40, 60]),
        
        "depth_error>thres0.1m_ratio": threshold_interval_metric(prediction, target, mask, di),
        "depth_error>thres0.2m_ratio": threshold_interval_metric(prediction, target, mask, di * 2),
        "depth_error>thres0.2m_ratio(2-20m)": threshold_interval_metric(prediction, target, mask, di * 2, [2, 20]),
        "depth_error>thres0.2m_ratio(2-30m)": threshold_interval_metric(prediction, target, mask, di * 2, [2, 30]),
        "depth_error>thres0.4m_ratio": threshold_interval_metric(prediction, target, mask, di * 4),
        "depth_error>thres0.8m_ratio": threshold_interval_metric(prediction, target, mask, di * 8),
        "depth_error>thres1.4m_ratio": threshold_interval_metric(prediction, target, mask, di * 14),
        "depth_error>thres2.0m_ratio": threshold_interval_metric(prediction, target, mask, di * 20),

        "absrel_error_ratio": depth_absrel_metric(prediction, target, mask),
    }     

    return scalar_outputs

    
def mvs_depth_dynamic_metric(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    batch: dotdict,
    fixscale: bool = True,
    batching: bool = True,
):
    modes = ["target", "refs", "refs_srcs"]
    all_outputs = {}

    for mode in modes:
        pred = prediction.clone()
        tgt = target.clone()
        msk = mask.clone()
        
        dynamic_msk = batch.dynamic_msk.clone()
        dynamic_msk = dynamic_msk[0]
        
        seg = batch.seg.clone()
        seg = torch.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
        seg = seg[0]

        if batching:
            if mode == "target":
                pred = pred[0].unsqueeze(0)
                tgt = tgt[0].unsqueeze(0)
                msk = msk[0].unsqueeze(0)
                dynamic_msk = dynamic_msk[0].unsqueeze(0)
                seg = seg[0].unsqueeze(0)
                
            elif mode == "refs":
                frame_idxs = batch.inds.squeeze().clone()
                ref_frame_id = frame_idxs[0]
                indices = torch.where(frame_idxs == ref_frame_id)[0]
                pred = pred[indices]
                tgt = tgt[indices]
                msk = msk[indices]
                dynamic_msk = dynamic_msk[indices]
                seg = seg[indices]

            elif mode == "refs_srcs":
                pass  # 全部保留

            else:
                raise ValueError(f"Unknown mode: {mode}")

            # mask 置零
            tgt[msk == 0] = 0.0
            pred[msk == 0] = 0.0

            tgt = tgt * batch.scale
            if fixscale:
                pred = pred * batch.scale
            else:
                pred = np.stack([
                    align_dpt_scale_shift_ransac(p, t, m)
                    for p, t, m in zip(pred, tgt, msk)
                ], axis=0)
        else:
            tgt[msk == 0] = 0.0
            pred[msk == 0] = 0.0
            tgt = tgt * batch.scale
            if fixscale:
                pred = pred * batch.scale
            else:
                pred = align_dpt_scale_shift_ransac(pred, tgt, msk)

        # 转 tensor
        pred = torch.tensor(pred, device=tgt.device).squeeze()
        tgt = tgt.squeeze()
        msk = msk.squeeze()
        dynamic_msk = dynamic_msk.squeeze()
        seg = seg.squeeze()

        di = 0.1
        prefix = f"{mode}_"
        all_outputs.update({
            f"{prefix}depth_interval0-0.5m_error": depth_interval_metric(pred, tgt, msk, [0, 0.5]),
            f"{prefix}depth_interval0.5-2m_error": depth_interval_metric(pred, tgt, msk, [0.5, 2]),
            f"{prefix}depth_interval0-2m_error": depth_interval_metric(pred, tgt, msk, [0, 2]),
            f"{prefix}depth_interval0-5m_error": depth_interval_metric(pred, tgt, msk, [0, 5]),
            f"{prefix}depth_interval5-10m_error": depth_interval_metric(pred, tgt, msk, [5, 10]),
            f"{prefix}depth_interval10-20m_error": depth_interval_metric(pred, tgt, msk, [10, 20]),
            f"{prefix}depth_interval20-40m_error": depth_interval_metric(pred, tgt, msk, [20, 40]),
            f"{prefix}depth_interval40-60m_error": depth_interval_metric(pred, tgt, msk, [40, 60]),

            f"{prefix}depth_error>thres0.1m_ratio": threshold_interval_metric(pred, tgt, msk, di),
            f"{prefix}depth_error>thres0.2m_ratio": threshold_interval_metric(pred, tgt, msk, di * 2),
            f"{prefix}depth_error>thres0.2m_ratio(0.5-20m)": threshold_interval_metric(pred, tgt, msk, di * 2, [0.5, 20]),
            f"{prefix}depth_error>thres0.2m_ratio(2-20m)": threshold_interval_metric(pred, tgt, msk, di * 2, [2, 20]),
            f"{prefix}depth_error>thres0.2m_ratio(2-30m)": threshold_interval_metric(pred, tgt, msk, di * 2, [2, 30]),
            f"{prefix}depth_error>thres0.4m_ratio": threshold_interval_metric(pred, tgt, msk, di * 4),
            f"{prefix}depth_error>thres0.8m_ratio": threshold_interval_metric(pred, tgt, msk, di * 8),
            f"{prefix}depth_error>thres1.4m_ratio": threshold_interval_metric(pred, tgt, msk, di * 14),
            f"{prefix}depth_error>thres2.0m_ratio": threshold_interval_metric(pred, tgt, msk, di * 20),

            f"{prefix}absrel_error_ratio": depth_absrel_metric(pred, tgt, msk),
        })


        if torch.all(torch.isnan(dynamic_msk)):
            moving_dynamic_mask_bchw = torch.zeros_like(dynamic_msk, dtype=torch.bool, device=dynamic_msk.device)
            static_dynamic_mask_bchw = torch.zeros_like(dynamic_msk, dtype=torch.bool, device=dynamic_msk.device)
            hybrid_dynamic_mask_bchw = torch.ones_like(dynamic_msk, dtype=torch.bool, device=dynamic_msk.device)
        else:
            moving_dynamic_mask_bchw = dynamic_msk > 0
            static_dynamic_mask_bchw = ~(dynamic_msk > 0)
            hybrid_dynamic_mask_bchw = torch.zeros_like(dynamic_msk, dtype=torch.bool, device=dynamic_msk.device)

        moving_dynamic_mask_raw_bchw = torch.isin(seg, torch.tensor(DYNAMIC_ELEMENT_ID, device=seg.device)) & (tgt > 0) & moving_dynamic_mask_bchw
        static_dynamic_mask_raw_bchw = torch.isin(seg, torch.tensor(DYNAMIC_ELEMENT_ID, device=seg.device)) & (tgt > 0) & static_dynamic_mask_bchw
        hybrid_dynamic_mask_raw_bchw = torch.isin(seg, torch.tensor(DYNAMIC_ELEMENT_ID, device=seg.device)) & (tgt > 0) & hybrid_dynamic_mask_bchw


        # static vehicle
        all_outputs.update({
            f"{prefix}static_dynamic_depth_interval0-0.5m_error": depth_interval_metric(pred, tgt, static_dynamic_mask_raw_bchw, [0, 0.5]),
            f"{prefix}static_dynamic_depth_interval0.5-2m_error": depth_interval_metric(pred, tgt, static_dynamic_mask_raw_bchw, [0.5, 2]),
            f"{prefix}static_dynamic_depth_interval2-5m_error": depth_interval_metric(pred, tgt, static_dynamic_mask_raw_bchw, [2, 5]),
            f"{prefix}static_dynamic_depth_interval5-10m_error": depth_interval_metric(pred, tgt, static_dynamic_mask_raw_bchw, [5, 10]),
            f"{prefix}static_dynamic_depth_interval10-20m_error": depth_interval_metric(pred, tgt, static_dynamic_mask_raw_bchw, [10, 20]),
            f"{prefix}static_dynamic_depth_interval20-40m_error": depth_interval_metric(pred, tgt, static_dynamic_mask_raw_bchw, [20, 40]),
            f"{prefix}static_dynamic_depth_interval40-60m_error": depth_interval_metric(pred, tgt, static_dynamic_mask_raw_bchw, [40, 60]),
            f"{prefix}static_dynamic_depth_interval60-100m_error": depth_interval_metric(pred, tgt, static_dynamic_mask_raw_bchw, [60, 100]),
            
            f"{prefix}static_dynamic_depth_error>thres0.1m_ratio": threshold_interval_metric(pred, tgt, static_dynamic_mask_raw_bchw, di),
            f"{prefix}static_dynamic_depth_error>thres0.2m_ratio": threshold_interval_metric(pred, tgt, static_dynamic_mask_raw_bchw, di * 2),
            f"{prefix}static_dynamic_depth_error>thres0.2m_ratio(0.5-20m)": threshold_interval_metric(pred, tgt, static_dynamic_mask_raw_bchw, di * 2, [0.5, 20]),
            f"{prefix}static_dynamic_depth_error>thres0.2m_ratio(2-20m)": threshold_interval_metric(pred, tgt, static_dynamic_mask_raw_bchw, di * 2, [2, 20]),
            f"{prefix}static_dynamic_depth_error>thres0.2m_ratio(2-30m)": threshold_interval_metric(pred, tgt, static_dynamic_mask_raw_bchw, di * 2, [2, 30]),
            f"{prefix}static_dynamic_depth_error>thres0.4m_ratio": threshold_interval_metric(pred, tgt, static_dynamic_mask_raw_bchw, di * 4),
            f"{prefix}static_dynamic_depth_error>thres0.8m_ratio": threshold_interval_metric(pred, tgt, static_dynamic_mask_raw_bchw, di * 8),
            f"{prefix}static_dynamic_depth_error>thres1.4m_ratio": threshold_interval_metric(pred, tgt, static_dynamic_mask_raw_bchw, di * 14),
            f"{prefix}static_dynamic_depth_error>thres2.0m_ratio": threshold_interval_metric(pred, tgt, static_dynamic_mask_raw_bchw, di * 20),
            
            f"{prefix}static_dynamic_absrel_error_ratio": depth_absrel_metric(pred, tgt, static_dynamic_mask_raw_bchw),
        })
        
        # moving vehicle
        all_outputs.update({
            f"{prefix}moving_dynamic_depth_interval0-0.5m_error": depth_interval_metric(pred, tgt, moving_dynamic_mask_raw_bchw, [0, 0.5]),
            f"{prefix}moving_dynamic_depth_interval0.5-2m_error": depth_interval_metric(pred, tgt, moving_dynamic_mask_raw_bchw, [0.5, 2]),
            f"{prefix}moving_dynamic_depth_interval2-5m_error": depth_interval_metric(pred, tgt, moving_dynamic_mask_raw_bchw, [2, 5]),
            f"{prefix}moving_dynamic_depth_interval5-10m_error": depth_interval_metric(pred, tgt, moving_dynamic_mask_raw_bchw, [5, 10]),
            f"{prefix}moving_dynamic_depth_interval10-20m_error": depth_interval_metric(pred, tgt, moving_dynamic_mask_raw_bchw, [10, 20]),
            f"{prefix}moving_dynamic_depth_interval20-40m_error": depth_interval_metric(pred, tgt, moving_dynamic_mask_raw_bchw, [20, 40]),
            f"{prefix}moving_dynamic_depth_interval40-60m_error": depth_interval_metric(pred, tgt, moving_dynamic_mask_raw_bchw, [40, 60]),
            f"{prefix}moving_dynamic_depth_interval60-100m_error": depth_interval_metric(pred, tgt, moving_dynamic_mask_raw_bchw, [60, 100]),
            
            f"{prefix}moving_dynamic_depth_error>thres0.1m_ratio": threshold_interval_metric(pred, tgt, moving_dynamic_mask_raw_bchw, di),
            f"{prefix}moving_dynamic_depth_error>thres0.2m_ratio": threshold_interval_metric(pred, tgt, moving_dynamic_mask_raw_bchw, di * 2),
            f"{prefix}moving_dynamic_depth_error>thres0.2m_ratio(0.5-20m)": threshold_interval_metric(pred, tgt, moving_dynamic_mask_raw_bchw, di * 2, [0.5, 20]),
            f"{prefix}moving_dynamic_depth_error>thres0.2m_ratio(2-20m)": threshold_interval_metric(pred, tgt, moving_dynamic_mask_raw_bchw, di * 2, [2, 20]),
            f"{prefix}moving_dynamic_depth_error>thres0.2m_ratio(2-30m)": threshold_interval_metric(pred, tgt, moving_dynamic_mask_raw_bchw, di * 2, [2, 30]),
            f"{prefix}moving_dynamic_depth_error>thres0.4m_ratio": threshold_interval_metric(pred, tgt, moving_dynamic_mask_raw_bchw, di * 4),
            f"{prefix}moving_dynamic_depth_error>thres0.8m_ratio": threshold_interval_metric(pred, tgt, moving_dynamic_mask_raw_bchw, di * 8),
            f"{prefix}moving_dynamic_depth_error>thres1.4m_ratio": threshold_interval_metric(pred, tgt, moving_dynamic_mask_raw_bchw, di * 14),
            f"{prefix}moving_dynamic_depth_error>thres2.0m_ratio": threshold_interval_metric(pred, tgt, moving_dynamic_mask_raw_bchw, di * 20),
            
            f"{prefix}moving_dynamic_absrel_error_ratio": depth_absrel_metric(pred, tgt, moving_dynamic_mask_raw_bchw),
        })
        
        # hybrid vehicle
        all_outputs.update({
            f"{prefix}hybrid_dynamic_depth_interval0-0.5m_error": depth_interval_metric(pred, tgt, hybrid_dynamic_mask_raw_bchw, [0, 0.5]),
            f"{prefix}hybrid_dynamic_depth_interval0.5-2m_error": depth_interval_metric(pred, tgt, hybrid_dynamic_mask_raw_bchw, [0.5, 2]),
            f"{prefix}hybrid_dynamic_depth_interval2-5m_error": depth_interval_metric(pred, tgt, hybrid_dynamic_mask_raw_bchw, [2, 5]),
            f"{prefix}hybrid_dynamic_depth_interval5-10m_error": depth_interval_metric(pred, tgt, hybrid_dynamic_mask_raw_bchw, [5, 10]),
            f"{prefix}hybrid_dynamic_depth_interval10-20m_error": depth_interval_metric(pred, tgt, hybrid_dynamic_mask_raw_bchw, [10, 20]),
            f"{prefix}hybrid_dynamic_depth_interval20-40m_error": depth_interval_metric(pred, tgt, hybrid_dynamic_mask_raw_bchw, [20, 40]),
            f"{prefix}hybrid_dynamic_depth_interval40-60m_error": depth_interval_metric(pred, tgt, hybrid_dynamic_mask_raw_bchw, [40, 60]),
            f"{prefix}hybrid_dynamic_depth_interval60-100m_error": depth_interval_metric(pred, tgt, hybrid_dynamic_mask_raw_bchw, [60, 100]),
            
            f"{prefix}hybrid_dynamic_depth_error>thres0.1m_ratio": threshold_interval_metric(pred, tgt, hybrid_dynamic_mask_raw_bchw, di),
            f"{prefix}hybrid_dynamic_depth_error>thres0.2m_ratio": threshold_interval_metric(pred, tgt, hybrid_dynamic_mask_raw_bchw, di * 2),
            f"{prefix}hybrid_dynamic_depth_error>thres0.2m_ratio(0.5-20m)": threshold_interval_metric(pred, tgt, hybrid_dynamic_mask_raw_bchw, di * 2, [0.5, 20]),
            f"{prefix}hybrid_dynamic_depth_error>thres0.2m_ratio(2-20m)": threshold_interval_metric(pred, tgt, hybrid_dynamic_mask_raw_bchw, di * 2, [2, 20]),
            f"{prefix}hybrid_dynamic_depth_error>thres0.2m_ratio(2-30m)": threshold_interval_metric(pred, tgt, hybrid_dynamic_mask_raw_bchw, di * 2, [2, 30]),
            f"{prefix}hybrid_dynamic_depth_error>thres0.4m_ratio": threshold_interval_metric(pred, tgt, hybrid_dynamic_mask_raw_bchw, di * 4),
            f"{prefix}hybrid_dynamic_depth_error>thres0.8m_ratio": threshold_interval_metric(pred, tgt, hybrid_dynamic_mask_raw_bchw, di * 8),
            f"{prefix}hybrid_dynamic_depth_error>thres1.4m_ratio": threshold_interval_metric(pred, tgt, hybrid_dynamic_mask_raw_bchw, di * 14),
            f"{prefix}hybrid_dynamic_depth_error>thres2.0m_ratio": threshold_interval_metric(pred, tgt, hybrid_dynamic_mask_raw_bchw, di * 20),
            
            f"{prefix}hybrid_dynamic_absrel_error_ratio": depth_absrel_metric(pred, tgt, hybrid_dynamic_mask_raw_bchw),
        })

    return all_outputs

@torch.no_grad()
def psnr(x: torch.Tensor, y: torch.Tensor):
    mse = compute_mse(x, y).mean()
    psnr = (1 / mse.clip(1e-10)).log() * 10 / np.log(10)
    return psnr.item()  # tensor to scalar


@torch.no_grad()
def ssim(x: torch.Tensor, y: torch.Tensor):
    return np.mean([
        compare_ssim(
            _x.detach().cpu().numpy(),
            _y.detach().cpu().numpy(),
            channel_axis=-1,
            data_range=2.0
        )
        for _x, _y in zip(x, y)
    ]).astype(float).item()


@torch.no_grad()
def lpips(x: torch.Tensor, y: torch.Tensor):
    if x.ndim == 3: x = x.unsqueeze(0)
    if y.ndim == 3: y = y.unsqueeze(0)
    x = x.permute(0, 3, 1, 2)
    y = y.permute(0, 3, 1, 2)
    return compute_lpips(x, y, net='vgg').item()


@torch.no_grad()
def chamfer_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):
    chamfer_distance = []

    # Compute the chamfer distance
    for i in range(x.shape[0]):
        dab = knn_points(
            x[i:i+1][mask[i:i+1]][None],
            y[i:i+1][mask[i:i+1]][None],
        )[0].mean()
        dba = knn_points(
            y[i:i+1][mask[i:i+1]][None],
            x[i:i+1][mask[i:i+1]][None],
        )[0].mean()
        chamfer_distance.append(dab + dba)

    return np.mean(chamfer_distance)


@torch.no_grad()
def distance(x: torch.Tensor, y: torch.Tensor):
    dist = (x - y).norm(dim=-1)
    return dist


@torch.no_grad()
def distance_accuracy(
    x: torch.Tensor,
    y: torch.Tensor,
    thresh: float = 0.2
):
    dist = (x - y).norm(dim=-1)
    acc = (dist < thresh).float()
    return acc


def interval_threshold(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.1,
    interval: List[float] = [1e-6, 1e6],
):
    m = (y >= interval[0]) & (y <= interval[1])
    if mask is not None: m = m & mask
    x = x[m]
    y = y[m]
    e = torch.abs(x - y)
    e = e > threshold
    if e.shape[0] == 0:
        return torch.tensor(0, device=e.device, dtype=y.dtype)
    return torch.mean(e.float())


def _finite_cam_metric_value(v):
    """将相机指标转为有限 float，避免 _filter_nonfinite_metrics 整帧丢弃（NaN/Inf 的 acc/auc）。"""
    try:
        if isinstance(v, torch.Tensor):
            t = torch.nan_to_num(v.detach().float(), nan=0.0, posinf=1.0, neginf=0.0)
            x = float(t.mean().item() if t.numel() > 0 else 0.0)
        else:
            x = float(np.asarray(v, dtype=np.float64))
            x = float(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0))
        if not np.isfinite(x):
            x = 0.0
        return x
    except Exception:
        return 0.0


def _sanitize_cam_metric_dict(metric: dotdict) -> dotdict:
    for k in list(metric.keys()):
        metric[k] = _finite_cam_metric_value(metric[k])
    return metric


@torch.no_grad()
def camera_accuracy_auc(
    x: torch.Tensor,
    y: torch.Tensor,
    batch: dotdict = None,
    acc_thresh: List[int] = [1, 3, 5, 15],
    auc_thresh: List[int] = [1, 3, 5, 10, 20, 30]
):
    # Decode the camera parameters
    x_w2c, x_ixt = decode_camera_params(
        x, batch.meta.H[0].item(), batch.meta.W[0].item()
    )  # (B, S, 3, 4), (B, S, 3, 3)
    y_w2c, y_ixt = decode_camera_params(
        y, batch.meta.H[0].item(), batch.meta.W[0].item()
    )  # (B, S, 3, 4), (B, S, 3, 3)

    # Debug toggle: some upstream codebases use c2w convention for pose matrices.
    # This switch helps verify whether a w2c/c2w mismatch is the source of AUC gap.
    # NOTE: affine_inverse supports (..., 3, 4) poses.
    if os.environ.get("EVC_CAM_AUC_GT_INVERT", "0") == "1":
        from easyvolcap.utils.math_utils import affine_inverse
        x_w2c = affine_inverse(x_w2c)
    if os.environ.get("EVC_CAM_AUC_PRED_INVERT", "0") == "1":
        from easyvolcap.utils.math_utils import affine_inverse
        y_w2c = affine_inverse(y_w2c)

    # Optional debug dump for paper-pose AUC mismatch investigation.
    # Enabled via env var to avoid spamming during long eval runs.
    if os.environ.get("EVC_DEBUG_CAM_AUC", "0") == "1":
        global _EVC_DEBUG_CAM_AUC_PRINTED
        try:
            printed = bool(_EVC_DEBUG_CAM_AUC_PRINTED)
        except Exception:
            printed = False
        if not printed:
            _EVC_DEBUG_CAM_AUC_PRINTED = True
            with torch.no_grad():
                # Absolute translation norms (w2c[..., :3, 3]).
                x_T = x_w2c[..., :3, 3].detach()
                y_T = y_w2c[..., :3, 3].detach()
                x_Tn = torch.linalg.norm(x_T, dim=-1)  # (B, S)
                y_Tn = torch.linalg.norm(y_T, dim=-1)  # (B, S)

                # Relative translation norms, matching camera_to_relative_degree.
                try:
                    from easyvolcap.utils.cam_utils import get_pairs
                    from easyvolcap.utils.math_utils import affine_padding, affine_inverse
                    B, S = x_w2c.shape[:2]
                    i, j = get_pairs(S)
                    i = i.to(x_w2c.device, non_blocking=True)
                    j = j.to(x_w2c.device, non_blocking=True)
                    P = int(i.numel())

                    x_i = x_w2c[:, i].reshape(B * P, 3, 4)
                    x_j_inv = affine_inverse(x_w2c[:, j].reshape(B * P, 3, 4))
                    rel_x = affine_padding(x_i).bmm(affine_padding(x_j_inv)).reshape(B, P, 4, 4)

                    y_i = y_w2c[:, i].reshape(B * P, 3, 4)
                    y_j_inv = affine_inverse(y_w2c[:, j].reshape(B * P, 3, 4))
                    rel_y = affine_padding(y_i).bmm(affine_padding(y_j_inv)).reshape(B, P, 4, 4)

                    rel_x_tn = torch.linalg.norm(rel_x[..., :3, 3], dim=-1)  # (B, P)
                    rel_y_tn = torch.linalg.norm(rel_y[..., :3, 3], dim=-1)  # (B, P)
                    rel_x_zero = (rel_x_tn < 1e-6).float().mean().item()
                    rel_y_zero = (rel_y_tn < 1e-6).float().mean().item()
                except Exception as exc:
                    rel_x_tn = rel_y_tn = None
                    rel_x_zero = rel_y_zero = None
                    exc_str = f"{type(exc).__name__}: {exc}"
                else:
                    exc_str = ""

                def _stat(t: torch.Tensor) -> str:
                    if t is None:
                        return "None"
                    safe = torch.nan_to_num(t.float(), nan=0.0, posinf=0.0, neginf=0.0)
                    return (
                        f"mean={safe.mean().item():.6g} "
                        f"min={safe.min().item():.6g} "
                        f"p1={torch.quantile(safe.flatten(), 0.01).item():.6g} "
                        f"p50={torch.quantile(safe.flatten(), 0.50).item():.6g} "
                        f"p99={torch.quantile(safe.flatten(), 0.99).item():.6g} "
                        f"max={safe.max().item():.6g}"
                    )

                log(yellow("[EVC_DEBUG_CAM_AUC] abs_T_norm gt  :"), blue(_stat(x_Tn)))
                log(yellow("[EVC_DEBUG_CAM_AUC] abs_T_norm pred:"), blue(_stat(y_Tn)))
                if rel_x_tn is not None:
                    log(yellow("[EVC_DEBUG_CAM_AUC] rel_t_norm gt  :"), blue(_stat(rel_x_tn)))
                    log(yellow("[EVC_DEBUG_CAM_AUC] rel_t_norm pred:"), blue(_stat(rel_y_tn)))
                    log(yellow("[EVC_DEBUG_CAM_AUC] rel_t_norm <1e-6 ratio gt/pred:"), blue(f"{rel_x_zero:.6g}/{rel_y_zero:.6g}"))
                else:
                    log(yellow("[EVC_DEBUG_CAM_AUC] rel_t_norm stats unavailable:"), red(exc_str))

    # Compute the relative rotation and translation in degree.
    # Default mode is the legacy VGGT implementation. For paper reproduction debugging,
    # set `EVC_CAM_AUC_MODE=centers` to compare translation directions via camera centers
    # in the canonical world frame (first camera), which avoids an extra left-multiply by R_i.
    cam_auc_mode = os.environ.get("EVC_CAM_AUC_MODE", "").strip().lower()
    if cam_auc_mode in {"centers", "paper", "paper_centers", "world_centers"}:
        r_rel, t_rel = camera_to_relative_degree_centers(x_w2c, y_w2c)
    else:
        r_rel, t_rel = camera_to_relative_degree(x_w2c, y_w2c)

    if os.environ.get("EVC_DEBUG_CAM_AUC", "0") == "1":
        try:
            log(yellow("[EVC_DEBUG_CAM_AUC] cam_auc_mode:"), blue(cam_auc_mode or "legacy"))
            log(yellow("[EVC_DEBUG_CAM_AUC] r_rel deg:"), blue(f"p50={torch.quantile(r_rel.flatten().float(), 0.5).item():.3g} p90={torch.quantile(r_rel.flatten().float(), 0.9).item():.3g} max={r_rel.max().item():.3g}"))
            log(yellow("[EVC_DEBUG_CAM_AUC] t_rel deg:"), blue(f"p50={torch.quantile(t_rel.flatten().float(), 0.5).item():.3g} p90={torch.quantile(t_rel.flatten().float(), 0.9).item():.3g} max={t_rel.max().item():.3g}"))
        except Exception:
            pass

    # Record the metrics
    metric = dotdict()

    # Compute the accuracy
    for thresh in acc_thresh:
        metric[f'rotation_accuracy_{thresh:02d}'] = (
            r_rel < thresh
        ).float().mean()
        metric[f'translation_accuracy_{thresh:02d}'] = (
            t_rel < thresh
        ).float().mean()

    # # Compute the histogram of the maximum error
    # histogram = calculate_auc(r_rel, t_rel)

    # # Compute the camera parameter AUC
    # for thresh in auc_thresh:
    #     metric[f'pose_auc_{thresh:02d}'] = torch.cumsum(
    #         histogram[..., :thresh], dim=-1
    #     ).mean()

    # Compute the camera parameter AUC
    # VGGT paper definition: AUC is the area under the accuracy-threshold curve of
    # the minimum values between RRA and RTA across varying thresholds.
    # We keep a legacy mode for backwards comparison.
    auc_combine = os.environ.get("EVC_CAM_AUC_COMBINE", "min").strip().lower()
    for thresh in auc_thresh:
        metric[f'pose_auc_{thresh:02d}'] = calculate_auc_np(
            r_rel.cpu().numpy(),
            t_rel.cpu().numpy(),
            thresh,
            combine=auc_combine,
        )

    # Compute the relative translation scale（预测平移范数过小会导致 inf，进而整帧 metric 被 _filter_nonfinite_metrics 丢弃）
    pred_t_norm = torch.norm(y_w2c[:, 1:, :3, 3], dim=-1).clamp(min=1e-12)
    gt_t_norm = torch.norm(x_w2c[:, 1:, :3, 3], dim=-1)
    scale = (gt_t_norm / pred_t_norm).mean()
    metric['translation_scale'] = float(torch.nan_to_num(scale, nan=0.0, posinf=1e6, neginf=0.0))

    return _sanitize_cam_metric_dict(metric)


@torch.no_grad()
def camera_accuracy_auc_vggt_eval(
    x: torch.Tensor,
    y: torch.Tensor,
    batch: dotdict = None,
    acc_thresh: List[int] = [1, 3, 5, 15],
    auc_thresh: List[int] = [1, 3, 5, 10, 20, 30],
):
    """VGGT evaluation-branch pose AUC (max-error histogram) implementation.

    Reference:
    - facebookresearch/vggt, branch `evaluation`, file `evaluation/test_co3d.py`
      (`calculate_auc_np` + `se3_to_relative_pose_error` path)
    """
    # Decode the camera parameters
    x_w2c, _ = decode_camera_params(
        x, batch.meta.H[0].item(), batch.meta.W[0].item()
    )  # (B, S, 3, 4)
    y_w2c, _ = decode_camera_params(
        y, batch.meta.H[0].item(), batch.meta.W[0].item()
    )  # (B, S, 3, 4)

    # Keep all unordered pairs C(S,2), exactly as the official eval script.
    r_rel, t_rel = camera_to_relative_degree(x_w2c, y_w2c)

    metric = dotdict()
    for thresh in acc_thresh:
        metric[f'rotation_accuracy_{thresh:02d}'] = (r_rel < thresh).float().mean()
        metric[f'translation_accuracy_{thresh:02d}'] = (t_rel < thresh).float().mean()

    # Official branch AUC: histogram over max(rotation_error, translation_error).
    for thresh in auc_thresh:
        metric[f'pose_auc_{thresh:02d}'] = calculate_auc_np(
            r_rel.cpu().numpy(),
            t_rel.cpu().numpy(),
            thresh,
            combine="max",
        )

    # Keep this diagnostic field consistent with CAM_ACC_AUC.
    pred_norm = torch.norm(y_w2c[:, 1:, :3, 3], dim=-1)
    # Avoid inf/NaN in translation_scale when predicted translations are degenerate.
    pred_norm = pred_norm.clamp(min=1e-12)
    scale = (torch.norm(x_w2c[:, 1:, :3, 3], dim=-1) / pred_norm).mean()
    metric['translation_scale'] = float(torch.nan_to_num(scale, nan=0.0, posinf=1e6, neginf=0.0))
    return _sanitize_cam_metric_dict(metric)


def align_sim3(gt_dict, pred_dict):
    """
    用Sim(3)变换（旋转+平移+尺度）对齐pred_dict到gt_dict。
    返回：尺度scale和对齐后的pred_dict（不修改输入）
    """
    assert gt_dict.keys() == pred_dict.keys(), "时间戳必须对齐"

    keys = sorted(gt_dict.keys())
    if len(keys) < 3:
        return 1.0, {k: pred_dict[k].copy() for k in keys}

    gt_xyz = np.array([gt_dict[t][:3, 3] for t in keys], dtype=np.float64)
    pred_xyz = np.array([pred_dict[t][:3, 3] for t in keys], dtype=np.float64)
    valid = np.isfinite(gt_xyz).all(axis=1) & np.isfinite(pred_xyz).all(axis=1)
    if valid.sum() < 3:
        return 1.0, {k: pred_dict[k].copy() for k in keys}
    gt_xyz = gt_xyz[valid]
    pred_xyz = pred_xyz[valid]

    # 计算质心
    centroid_gt = np.mean(gt_xyz, axis=0)
    centroid_pred = np.mean(pred_xyz, axis=0)

    # 中心化
    gt_centered = gt_xyz - centroid_gt
    pred_centered = pred_xyz - centroid_pred

    # 估计尺度：只使用过滤后的有效点，避免 NaN/Inf 样本污染
    n_valid = gt_centered.shape[0]
    gt_norm = np.sqrt(np.sum(gt_centered ** 2) / n_valid)
    pred_norm = np.sqrt(np.sum(pred_centered ** 2) / n_valid)
    if pred_norm < 1e-12:
        return 1.0, {k: pred_dict[k].copy() for k in keys}
    scale = gt_norm / pred_norm

    # 缩放pred_centered
    pred_scaled = pred_centered * scale

    # 计算旋转矩阵（SVD）
    H = pred_scaled.T @ gt_centered
    try:
        U, _, Vt = np.linalg.svd(H)
        R_mat = Vt.T @ U.T
        if np.linalg.det(R_mat) < 0:
            Vt[-1, :] *= -1
            R_mat = Vt.T @ U.T
    except np.linalg.LinAlgError:
        return 1.0, {k: pred_dict[k].copy() for k in keys}

    # 计算平移向量
    t_vec = centroid_gt - scale * (R_mat @ centroid_pred)

    # 应用Sim(3)变换到姿态矩阵
    aligned_pred_dict = {}
    for t in keys:
        pose = pred_dict[t].copy()
        if not np.isfinite(pose).all():
            aligned_pred_dict[t] = pose
            continue
        # 旋转旋转矩阵部分
        pose[:3, :3] = R_mat @ pose[:3, :3]
        # 变换平移部分
        pose[:3, 3] = scale * (R_mat @ pose[:3, 3]) + t_vec
        aligned_pred_dict[t] = pose

    return scale, aligned_pred_dict

def align_se3(gt_dict, pred_dict):
    """
    用Sim(3)变换（旋转+平移+尺度）对齐pred_dict到gt_dict。
    返回：尺度scale和对齐后的pred_dict（不修改输入）
    """
    assert gt_dict.keys() == pred_dict.keys(), "时间戳必须对齐"

    keys = sorted(gt_dict.keys())
    gt_xyz = np.array([gt_dict[t][:3, 3] for t in keys])
    pred_xyz = np.array([pred_dict[t][:3, 3] for t in keys])

    # 计算质心
    centroid_gt = np.mean(gt_xyz, axis=0)
    centroid_pred = np.mean(pred_xyz, axis=0)

    # 中心化
    gt_centered = gt_xyz - centroid_gt
    pred_centered = pred_xyz - centroid_pred

    # 缩放pred_centered
    pred_scaled = pred_centered

    # 计算旋转矩阵（SVD）
    H = pred_scaled.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T

    # 计算平移向量
    t_vec = centroid_gt - (R_mat @ centroid_pred)

    # 应用Sim(3)变换到姿态矩阵
    aligned_pred_dict = {}
    for t in keys:
        pose = pred_dict[t].copy()
        # 旋转旋转矩阵部分
        pose[:3, :3] = R_mat @ pose[:3, :3]
        # 变换平移部分
        pose[:3, 3] = (R_mat @ pose[:3, 3]) + t_vec
        aligned_pred_dict[t] = pose

    return 1.0, aligned_pred_dict

def compute_rpe(gt_dict, pred_dict):
    if not _require_evo("compute_rpe"):
        return 0.0, 0.0
    traj_gt = PosePath3D(poses_se3=[gt_dict[i] for i in sorted(gt_dict.keys())])
    traj_pred = PosePath3D(poses_se3=[pred_dict[i] for i in sorted(pred_dict.keys())])

    # Translational error
    rpe_trans_metric = RPE(
        pose_relation=PoseRelation.translation_part,
        delta=1, delta_unit=Unit.frames, pairs_from_reference=True
    )
    rpe_trans_metric.process_data((traj_gt, traj_pred))
    
    # 获取 GT pair 索引 (i, j)
    id_pairs = id_pairs_from_delta(
        traj_gt.poses_se3 if rpe_trans_metric.pairs_from_reference else traj_pred.poses_se3,
        delta=rpe_trans_metric.delta,
        delta_unit=rpe_trans_metric.delta_unit,
        rel_tol=rpe_trans_metric.rel_delta_tol, 
        all_pairs=rpe_trans_metric.all_pairs
    )
    # 用 GT 的位置信息计算每对的实际距离
    positions_gt = traj_gt.positions_xyz
    gt_dists = np.array([
        np.linalg.norm(positions_gt[j] - positions_gt[i])
        for i, j in id_pairs
    ])
    errors = np.array(rpe_trans_metric.error)
    trans_err_per_meter = errors / gt_dists
    
    # Rotational error in degrees
    rpe_rot_metric = RPE(
        pose_relation=PoseRelation.rotation_angle_deg,
        delta=1, delta_unit=Unit.frames, pairs_from_reference=True
    )
    rpe_rot_metric.process_data((traj_gt, traj_pred))
    rot_errors = np.array(rpe_rot_metric.error)
    rot_error_per_meter = rot_errors / gt_dists  # 与 trans 用同一 id_pairs
    
    mean_trans_err = np.mean(trans_err_per_meter)
    mean_rot_err = np.mean(rot_error_per_meter)
    
    return mean_trans_err, mean_rot_err


def _camera_maps_to_sorted_c2w(
    x: torch.Tensor,
    y: torch.Tensor,
    batch: dotdict = None,
):
    """
    Convert GT/pred camera maps to temporally sorted c2w trajectories.

    GeometryEvaluator passes:
      - x: gt camera map
      - y: predicted camera map

    For multiview batches in this repo, selected frames are often packed as
    `[target] + sources`, which is not guaranteed to be temporal order.
    Trajectory metrics must therefore re-sort by the actual frame indices.
    """

    def invert_poses_3x4(pose: torch.Tensor) -> torch.Tensor:
        if pose.shape[-2:] != (3, 4):
            raise ValueError(f"Expected (..., 3, 4), got {pose.shape}")
        prefix = pose.shape[:-2]
        ones = torch.zeros(*prefix, 1, 4, dtype=pose.dtype, device=pose.device)
        ones[..., 0, 3] = 1.0
        pose_4x4 = torch.cat([pose, ones], dim=-2)
        return torch.linalg.inv(pose_4x4)

    x_w2c, _ = decode_camera_params(x, batch.meta.H[0].item(), batch.meta.W[0].item())
    y_w2c, _ = decode_camera_params(y, batch.meta.H[0].item(), batch.meta.W[0].item())

    gt_c2w = invert_poses_3x4(x_w2c)[0]
    pred_c2w = invert_poses_3x4(y_w2c)[0]

    world_scale = float(batch.scale.item()) if hasattr(batch, "scale") else 1.0
    gt_c2w[:, :3, 3] = gt_c2w[:, :3, 3] * world_scale
    pred_c2w[:, :3, 3] = pred_c2w[:, :3, 3] * world_scale

    order = None
    for key in ("inds",):
        candidate = None
        if hasattr(batch, key):
            candidate = getattr(batch, key)
        elif hasattr(batch, "meta") and hasattr(batch.meta, key):
            candidate = getattr(batch.meta, key)
        if candidate is None:
            continue
        if isinstance(candidate, torch.Tensor):
            tensor = candidate.detach().cpu()
        else:
            tensor = torch.as_tensor(candidate)
        tensor = tensor.reshape(-1)
        if tensor.numel() == gt_c2w.shape[0]:
            order = torch.argsort(tensor).to(gt_c2w.device)
            break

    if order is None:
        order = torch.arange(gt_c2w.shape[0], device=gt_c2w.device)

    gt_c2w = gt_c2w[order]
    pred_c2w = pred_c2w[order]
    return gt_c2w, pred_c2w


def _trajectory_dict_from_c2w(
    gt_c2w: torch.Tensor,
    pred_c2w: torch.Tensor,
):
    valid_pairs = []
    for i, (g, p) in enumerate(zip(gt_c2w, pred_c2w)):
        g_np = g.detach().cpu().numpy()
        p_np = p.detach().cpu().numpy()
        if np.isfinite(g_np).all() and np.isfinite(p_np).all():
            valid_pairs.append((i, g_np, p_np))

    gt_dict = {j: g for j, (_, g, _) in enumerate(valid_pairs)}
    pred_dict = {j: p for j, (_, _, p) in enumerate(valid_pairs)}
    return gt_dict, pred_dict


def _pose_rpe_rmse(traj_gt: PosePath3D, traj_pred: PosePath3D):
    try:
        rpe_trans_metric = RPE(
            pose_relation=PoseRelation.translation_part,
            delta=1,
            delta_unit=Unit.frames,
            all_pairs=True,
            pairs_from_reference=True,
        )
        rpe_trans_metric.process_data((traj_gt, traj_pred))
        rpe_trans_err = np.array(rpe_trans_metric.error, dtype=np.float64)
        rpe_trans_rmse = float(np.sqrt(np.mean(rpe_trans_err**2))) if rpe_trans_err.size else 0.0

        rpe_rot_metric = RPE(
            pose_relation=PoseRelation.rotation_angle_deg,
            delta=1,
            delta_unit=Unit.frames,
            all_pairs=True,
            pairs_from_reference=True,
        )
        rpe_rot_metric.process_data((traj_gt, traj_pred))
        rpe_rot_err = np.array(rpe_rot_metric.error, dtype=np.float64)
        rpe_rot_rmse = float(np.sqrt(np.mean(rpe_rot_err**2))) if rpe_rot_err.size else 0.0
    except Exception:
        rpe_trans_rmse = 0.0
        rpe_rot_rmse = 0.0
    return rpe_trans_rmse, rpe_rot_rmse


def _rotation_angle_deg_from_pose_error(pose_error: np.ndarray) -> float:
    rel_r = pose_error[:3, :3]
    tr = float(np.clip((np.trace(rel_r) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(tr)))


def _trajectory_rpe_rmse(gt_dict, pred_dict):
    keys = sorted(gt_dict.keys())
    if len(keys) < 2:
        return 0.0, 0.0

    trans_err = []
    rot_err = []
    for prev_key, curr_key in zip(keys[:-1], keys[1:]):
        gt_rel = np.linalg.inv(gt_dict[prev_key]) @ gt_dict[curr_key]
        pred_rel = np.linalg.inv(pred_dict[prev_key]) @ pred_dict[curr_key]
        pose_error = np.linalg.inv(pred_rel) @ gt_rel
        trans_err.append(float(np.linalg.norm(pose_error[:3, 3])))
        rot_err.append(float(_rotation_angle_deg_from_pose_error(pose_error)))

    trans_err = np.asarray(trans_err, dtype=np.float64)
    rot_err = np.asarray(rot_err, dtype=np.float64)
    trans_rmse = float(np.sqrt(np.mean(trans_err**2))) if trans_err.size else 0.0
    rot_rmse = float(np.sqrt(np.mean(rot_err**2))) if rot_err.size else 0.0
    return trans_rmse, rot_rmse


@torch.no_grad()
def camera_evo(
    x: torch.Tensor,
    y: torch.Tensor,
    batch: dotdict = None,
):
    if not _require_evo("camera_evo"):
        metric = dotdict()
        metric["RPE_trans"] = torch.tensor(0.0)
        metric["RPE_rot"] = torch.tensor(0.0)
        return metric


    metric = dotdict()
    # Decode the camera 
    x_w2c, x_ixt = decode_camera_params(
        x, batch.meta.H[0].item(), batch.meta.W[0].item()
    )  # (B, S, 3, 4), (B, S, 3, 3)
    y_w2c, y_ixt = decode_camera_params(
        y, batch.meta.H[0].item(), batch.meta.W[0].item()
    )  # (B, S, 3, 4), (B, S, 3, 3)_w

    def invert_poses_3x4(pose: torch.Tensor) -> torch.Tensor:
        """
        将 (B, S, 3, 4) 的相机位姿矩阵变为 (B, S, 4, 4)，并求逆
        """
        if pose.shape[-2:] != (3, 4):
            raise ValueError(f"Expected (..., 3, 4), got {pose.shape}")
        
        # 构造齐次矩阵的最后一行 [0, 0, 0, 1]
        B = pose.shape[:-2]
        ones = torch.zeros(*B, 1, 4, dtype=pose.dtype, device=pose.device)
        ones[..., 0, 3] = 1.0

        # 拼接成 4x4
        pose_4x4 = torch.cat([pose, ones], dim=-2)  # (..., 4, 4)

        # 求逆
        pose_inv = torch.linalg.inv(pose_4x4)

        return pose_inv

    x_c2w = invert_poses_3x4(x_w2c)
    y_c2w = invert_poses_3x4(y_w2c)
    x_c2w = x_c2w[0] # (B, 4, 4)
    y_c2w = y_c2w[0] # (B, 4, 4)
    
    # trans乘scale
    x_c2w[:,:3,3] = x_c2w[:,:3,3] * batch.scale.item()
    y_c2w[:,:3,3] = y_c2w[:,:3,3] * batch.scale.item()
    gt_dict = {i: mat.cpu().numpy() for i, mat in enumerate(y_c2w)}  # GT
    pred_dict = {i: mat.cpu().numpy() for i, mat in enumerate(x_c2w)}  # Pred
    
    scale, traj_aligned = align_se3(gt_dict, pred_dict)
    try:
        rpe_trans, rpe_rot = compute_rpe(gt_dict, traj_aligned)
    except Exception as e:
        print(e)
        rpe_trans, rpe_rot = 0.0, 0.0
    # Run evo_rpe
    # print(f"rpe_trans: {rpe_trans}, rpe_rot: {rpe_rot}")
    metric["RPE_trans"] = torch.tensor(rpe_trans)
    metric["RPE_rot"] = torch.tensor(rpe_rot)
    return metric


@torch.no_grad()
def camera_evo_pi3(
    x: torch.Tensor,
    y: torch.Tensor,
    batch: dotdict = None,
):
    """
    PI3/VGGT-evaluation compatible RelPose Distance metric.

    Notes:
    - x is GT camera map, y is prediction camera map (GeometryEvaluator order).
    - Use Sim(3) alignment once on whole trajectory, then report:
      - ATE_trans: translation RMSE
      - ATE_angle: rotation RMSE (deg), convenience field for sheet export
      - RPE_trans: translation RPE RMSE (delta=1, all_pairs=True)
      - RPE_rot: rotation RPE RMSE in deg (delta=1, all_pairs=True)
    """
    if not _HAS_EVO:
        raise RuntimeError(
            "camera_evo_pi3 requires evo package. "
            "Please run eval in the VGGT docker env with evo installed."
        )

    # Decode camera parameters (w2c), then convert to c2w.
    x_w2c, _ = decode_camera_params(x, batch.meta.H[0].item(), batch.meta.W[0].item())
    y_w2c, _ = decode_camera_params(y, batch.meta.H[0].item(), batch.meta.W[0].item())

    def invert_poses_3x4(pose: torch.Tensor) -> torch.Tensor:
        if pose.shape[-2:] != (3, 4):
            raise ValueError(f"Expected (..., 3, 4), got {pose.shape}")
        B = pose.shape[:-2]
        ones = torch.zeros(*B, 1, 4, dtype=pose.dtype, device=pose.device)
        ones[..., 0, 3] = 1.0
        pose_4x4 = torch.cat([pose, ones], dim=-2)
        return torch.linalg.inv(pose_4x4)

    # GeometryEvaluator passes x=gt, y=pred.
    gt_c2w = invert_poses_3x4(x_w2c)[0]
    pred_c2w = invert_poses_3x4(y_w2c)[0]

    # Recover metric scale in world coordinates.
    world_scale = float(batch.scale.item()) if hasattr(batch, "scale") else 1.0
    gt_c2w[:, :3, 3] = gt_c2w[:, :3, 3] * world_scale
    pred_c2w[:, :3, 3] = pred_c2w[:, :3, 3] * world_scale

    valid_pairs = []
    for i, (g, p) in enumerate(zip(gt_c2w, pred_c2w)):
        g_np = g.detach().cpu().numpy()
        p_np = p.detach().cpu().numpy()
        if np.isfinite(g_np).all() and np.isfinite(p_np).all():
            valid_pairs.append((i, g_np, p_np))
    if len(valid_pairs) < 2:
        metric = dotdict()
        metric["RPE_trans"] = torch.tensor(0.0)
        metric["RPE_rot"] = torch.tensor(0.0)
        metric["ATE_trans"] = torch.tensor(0.0)
        metric["ATE_angle"] = torch.tensor(0.0)
        metric["translation_scale"] = torch.tensor(1.0)
        return metric

    # Remap to compact indices for robust evo pair generation.
    gt_dict = {j: g for j, (_, g, _) in enumerate(valid_pairs)}
    pred_dict = {j: p for j, (_, _, p) in enumerate(valid_pairs)}

    # PI3 relpose distance uses scale-aligned trajectories.
    sim3_scale, pred_aligned = align_sim3(gt_dict, pred_dict)

    keys = sorted(gt_dict.keys())
    if len(keys) < 2:
        metric = dotdict()
        metric["RPE_trans"] = torch.tensor(0.0)
        metric["RPE_rot"] = torch.tensor(0.0)
        metric["ATE_trans"] = torch.tensor(0.0)
        metric["ATE_angle"] = torch.tensor(0.0)
        metric["translation_scale"] = torch.tensor(sim3_scale)
        return metric

    # ATE RMSE
    trans_err = np.empty(len(keys), dtype=np.float64)
    rot_err = np.empty(len(keys), dtype=np.float64)
    for i, k in enumerate(keys):
        gt_pose = gt_dict[k]
        pr_pose = pred_aligned[k]
        trans_err[i] = float(np.linalg.norm(pr_pose[:3, 3] - gt_pose[:3, 3]))
        rel_R = pr_pose[:3, :3] @ gt_pose[:3, :3].T
        tr = float(np.clip((np.trace(rel_R) - 1.0) / 2.0, -1.0, 1.0))
        rot_err[i] = float(np.degrees(np.arccos(tr)))
    ate_trans_rmse = float(np.sqrt(np.mean(trans_err**2)))
    ate_rot_rmse = float(np.sqrt(np.mean(rot_err**2)))

    # RPE RMSE with evo, no per-meter normalization.
    traj_gt = PosePath3D(poses_se3=[gt_dict[i] for i in keys])
    traj_pred = PosePath3D(poses_se3=[pred_aligned[i] for i in keys])

    try:
        rpe_trans_metric = RPE(
            pose_relation=PoseRelation.translation_part,
            delta=1,
            delta_unit=Unit.frames,
            all_pairs=True,
            pairs_from_reference=True,
        )
        rpe_trans_metric.process_data((traj_gt, traj_pred))
        rpe_trans_err = np.array(rpe_trans_metric.error, dtype=np.float64)
        rpe_trans_rmse = float(np.sqrt(np.mean(rpe_trans_err**2))) if rpe_trans_err.size else 0.0

        rpe_rot_metric = RPE(
            pose_relation=PoseRelation.rotation_angle_deg,
            delta=1,
            delta_unit=Unit.frames,
            all_pairs=True,
            pairs_from_reference=True,
        )
        rpe_rot_metric.process_data((traj_gt, traj_pred))
        rpe_rot_err = np.array(rpe_rot_metric.error, dtype=np.float64)
        rpe_rot_rmse = float(np.sqrt(np.mean(rpe_rot_err**2))) if rpe_rot_err.size else 0.0
    except Exception:
        rpe_trans_rmse = 0.0
        rpe_rot_rmse = 0.0

    metric = dotdict()
    metric["RPE_trans"] = torch.tensor(rpe_trans_rmse)
    metric["RPE_rot"] = torch.tensor(rpe_rot_rmse)
    metric["ATE_trans"] = torch.tensor(ate_trans_rmse)
    metric["ATE_angle"] = torch.tensor(ate_rot_rmse)
    metric["translation_scale"] = torch.tensor(sim3_scale)
    return metric


@torch.no_grad()
def camera_traj_evo(
    x: torch.Tensor,
    y: torch.Tensor,
    batch: dotdict = None,
):
    """
    Sequence trajectory metric for driving-style evaluation.

    - Re-sort selected views by their real frame indices.
    - Align the whole predicted trajectory to GT with Sim(3).
    - Report ATE/RPE RMSE.
    """
    gt_c2w, pred_c2w = _camera_maps_to_sorted_c2w(x, y, batch)
    gt_dict, pred_dict = _trajectory_dict_from_c2w(gt_c2w, pred_c2w)
    keys = sorted(gt_dict.keys())

    metric = dotdict()
    if len(keys) < 2:
        metric["RPE_trans"] = torch.tensor(0.0)
        metric["RPE_rot"] = torch.tensor(0.0)
        metric["ATE_trans"] = torch.tensor(0.0)
        metric["ATE_angle"] = torch.tensor(0.0)
        metric["translation_scale"] = torch.tensor(1.0)
        metric["num_frames"] = torch.tensor(float(len(keys)))
        return metric

    sim3_scale, pred_aligned = align_sim3(gt_dict, pred_dict)

    trans_err = np.empty(len(keys), dtype=np.float64)
    rot_err = np.empty(len(keys), dtype=np.float64)
    for i, k in enumerate(keys):
        gt_pose = gt_dict[k]
        pr_pose = pred_aligned[k]
        trans_err[i] = float(np.linalg.norm(pr_pose[:3, 3] - gt_pose[:3, 3]))
        rel_r = pr_pose[:3, :3] @ gt_pose[:3, :3].T
        tr = float(np.clip((np.trace(rel_r) - 1.0) / 2.0, -1.0, 1.0))
        rot_err[i] = float(np.degrees(np.arccos(tr)))

    rpe_trans_rmse, rpe_rot_rmse = _trajectory_rpe_rmse(gt_dict, pred_aligned)

    metric["RPE_trans"] = torch.tensor(rpe_trans_rmse)
    metric["RPE_rot"] = torch.tensor(rpe_rot_rmse)
    metric["ATE_trans"] = torch.tensor(float(np.sqrt(np.mean(trans_err**2))))
    metric["ATE_angle"] = torch.tensor(float(np.sqrt(np.mean(rot_err**2))))
    metric["translation_scale"] = torch.tensor(sim3_scale)
    metric["num_frames"] = torch.tensor(float(len(keys)))
    return metric


@torch.no_grad()
def camera_kitti_odometry(
    x: torch.Tensor,
    y: torch.Tensor,
    batch: dotdict = None,
):
    """
    KITTI odometry-style drift metric on a temporally sorted trajectory window.

    The metric follows the official KITTI odometry definition:
      - translational drift (%)
      - rotational drift (deg / m)

    For windowed inference, only segment lengths supported by the current
    trajectory chunk contribute to the average.
    """
    gt_c2w, pred_c2w = _camera_maps_to_sorted_c2w(x, y, batch)
    gt_dict, pred_dict = _trajectory_dict_from_c2w(gt_c2w, pred_c2w)
    keys = sorted(gt_dict.keys())

    metric = dotdict()
    if len(keys) < 2:
        metric["KITTI_t_err_pct"] = torch.tensor(0.0)
        metric["KITTI_r_err_deg_per_m"] = torch.tensor(0.0)
        metric["KITTI_num_segments"] = torch.tensor(0.0)
        metric["translation_scale"] = torch.tensor(1.0)
        metric["num_frames"] = torch.tensor(float(len(keys)))
        return metric

    align_mode = os.environ.get("EVC_KITTI_ODOM_ALIGN", "sim3").strip().lower()
    align_scale = 1.0
    if align_mode == "sim3":
        align_scale, pred_dict = align_sim3(gt_dict, pred_dict)
    elif align_mode == "se3":
        align_scale, pred_dict = align_se3(gt_dict, pred_dict)
    elif align_mode in {"none", ""}:
        pass
    else:
        raise ValueError(f"Unsupported EVC_KITTI_ODOM_ALIGN={align_mode}")

    gt = np.stack([gt_dict[k] for k in keys], axis=0)
    pred = np.stack([pred_dict[k] for k in keys], axis=0)

    gt0_inv = np.linalg.inv(gt[0])
    pred0_inv = np.linalg.inv(pred[0])
    gt = gt0_inv[None] @ gt
    pred = pred0_inv[None] @ pred

    dists = np.zeros((len(gt),), dtype=np.float64)
    for i in range(1, len(gt)):
        dists[i] = dists[i - 1] + np.linalg.norm(gt[i, :3, 3] - gt[i - 1, :3, 3])

    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    first_frame_step = 10
    trans_errors = []
    rot_errors = []

    for first in range(0, len(gt), first_frame_step):
        start_dist = dists[first]
        for seg_len in lengths:
            last = -1
            for idx in range(first + 1, len(gt)):
                if dists[idx] > start_dist + seg_len:
                    last = idx
                    break
            if last < 0:
                continue

            pose_delta_gt = np.linalg.inv(gt[first]) @ gt[last]
            pose_delta_pred = np.linalg.inv(pred[first]) @ pred[last]
            pose_error = np.linalg.inv(pose_delta_pred) @ pose_delta_gt

            trans_errors.append(float(np.linalg.norm(pose_error[:3, 3]) / seg_len))
            rot_errors.append(float(_rotation_angle_deg_from_pose_error(pose_error) / seg_len))

    if trans_errors:
        metric["KITTI_t_err_pct"] = torch.tensor(float(np.mean(trans_errors) * 100.0))
        metric["KITTI_r_err_deg_per_m"] = torch.tensor(float(np.mean(rot_errors)))
        metric["KITTI_num_segments"] = torch.tensor(float(len(trans_errors)))
    else:
        metric["KITTI_t_err_pct"] = torch.tensor(0.0)
        metric["KITTI_r_err_deg_per_m"] = torch.tensor(0.0)
        metric["KITTI_num_segments"] = torch.tensor(0.0)
    metric["translation_scale"] = torch.tensor(float(align_scale))
    metric["num_frames"] = torch.tensor(float(len(keys)))
    return metric

def depth_metric(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    batch: dotdict = None,
    fixscale: bool = False,
    eps: float = 1e-5,
    batching: bool = True,
):
    # Align the predicted depth to the ground truth depth
    if batching:
        prediction = np.stack([
            align_dpt_scale_shift_ransac(p, t, m)
                for p, t, m in zip(prediction, target, mask)
        ], axis=0)
    else:
        prediction = align_dpt_scale_shift_ransac(
            prediction, target, mask
        )

    # Convert everything to numpy and normalize shapes.
    # NOTE: when batching=True, `prediction` can already be a numpy array with
    # an extra batch axis (e.g. [1, H, W] or [1, N]), while target/mask are
    # squeezed tensors. We explicitly squeeze+reshape here to avoid boolean
    # indexing shape mismatch in `prediction[mask]`.
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    else:
        prediction = np.asarray(prediction)
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    else:
        target = np.asarray(target)
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    else:
        mask = np.asarray(mask)

    prediction = np.squeeze(prediction)
    target = np.squeeze(target)
    mask = (np.squeeze(mask) > 0)

    if prediction.shape != target.shape or prediction.shape != mask.shape:
        if prediction.size == target.size == mask.size:
            prediction = prediction.reshape(-1)
            target = target.reshape(-1)
            mask = mask.reshape(-1)
        else:
            log(
                yellow(
                    "[depth_metric] shape mismatch: "
                    f"pred={prediction.shape} target={target.shape} mask={mask.shape}; "
                    "fallback to flattened min-size intersection."
                )
            )
            n = min(prediction.size, target.size, mask.size)
            if n <= 0:
                return dotdict(
                    t11=0.0, d05=0.0, d1=0.0, d2=0.0, d3=0.0,
                    l1=0.0, rmse=0.0, irmse=0.0, imae=0.0,
                    abs_rel=0.0, sq_rel=0.0, i00_02=0.0, i00_05=0.0, i00_10=0.0,
                )
            prediction = prediction.reshape(-1)[:n]
            target = target.reshape(-1)[:n]
            mask = mask.reshape(-1)[:n]

    # Using the valid points only
    x = prediction[mask]
    y = target[mask]

    if x.size == 0 or y.size == 0:
        return dotdict(
            t11=0.0, d05=0.0, d1=0.0, d2=0.0, d3=0.0,
            l1=0.0, rmse=0.0, irmse=0.0, imae=0.0,
            abs_rel=0.0, sq_rel=0.0, i00_02=0.0, i00_05=0.0, i00_10=0.0,
        )

    # Compute the error threshold related metrics
    e = np.maximum(y / (x + eps), x / (y + eps))
    t11 = (e < 1.1).mean()
    d05 = (e < 1.25 ** 0.5).mean()
    d1  = (e < 1.25).mean()
    d2  = (e < 1.25 ** 2).mean()
    d3  = (e < 1.25 ** 3).mean()

    # Compute the L1 error
    l1 = np.mean(np.abs(y - x))

    # Compute the RMSE
    rmse = (y - x) ** 2
    rmse = np.sqrt(rmse.mean())

    # Compute the inverse RMSE and MAE
    diff_inv = 1 / (x + 1e-8) - 1 / (y + 1e-8)
    irmse = np.sqrt((diff_inv ** 2).mean())
    imae = np.mean(np.abs(diff_inv))

    # Compute the absolute relative and squared relative error
    abs_rel = np.mean(np.abs(y - x) / (y + eps))
    sq_rel = np.mean(((y - x) ** 2) / (y + eps))

    # Compute the error interval related metrics
    e = np.abs(y - x)
    i00_02 = e[(y > 0) & (y <=  2)].mean()
    i00_05 = e[(y > 0) & (y <=  5)].mean()
    i00_10 = e[(y > 0) & (y <= 10)].mean()

    return dotdict(
        t11=t11,
        d05=d05,
        d1=d1,
        d2=d2,
        d3=d3,
        l1=l1,
        rmse=rmse,
        irmse=irmse,
        imae=imae,
        abs_rel=abs_rel,
        sq_rel=sq_rel,
        i00_02=i00_02,
        i00_05=i00_05,
        i00_10=i00_10,
    )


def compute_accuracy(
    target: torch.Tensor,
    prediction: torch.Tensor,
    workers: int = 24,
):
    """ A numpy implementation of the accuracy metric """
    kdtree = KDTree(target)
    distance, idx = kdtree.query(prediction, workers=workers)
    return np.mean(distance)


def compute_completion(
    target: torch.Tensor,
    prediction: torch.Tensor,
    workers: int = 24,
):
    """ A numpy implementation of the completion metric """
    kdtree = KDTree(prediction)
    distance, idx = kdtree.query(target, workers=workers)
    return np.mean(distance)


def point_accuracy_completion_overall(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    batch: dotdict = None,
    eps: float = 1e-5,
    batching: bool = True,
):
    # Align the predicted point to the ground truth point
    # Using the ROE optimal alignment method here
    if batching:
        prediction = np.stack([
            align_xyz_umeyama(p, t, m)
                for p, t, m in zip(prediction, target, mask)
        ], axis=0)
    else:
        prediction = align_xyz_umeyama(
            prediction, target, mask
        )

    # Convert to numpy while preserving scene dimension.
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if mask is None:
        mask = np.ones(prediction.shape[:-1], dtype=bool)
    else:
        mask = np.asarray(mask)

    # Keep batching layout as (S, P, 3)/(S, P) to avoid squeeze()-induced
    # axis collapse when S==1.
    if batching:
        if prediction.ndim == 2 and prediction.shape[-1] == 3:
            prediction = prediction[None, ...]
        if target.ndim == 2 and target.shape[-1] == 3:
            target = target[None, ...]
        if mask.ndim == 1:
            mask = mask[None, ...]
        if mask.ndim > 1 and mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)

    # Compute the accuracy and completion
    accuracy = 0.0
    completion = 0.0
    
    if batching:
        cnt = 0
        for p, t, m in zip(prediction, target, mask):
            # Normalize mask to point-wise shape to avoid numpy boolean indexing
            # flattening (e.g. m: (P, 1) on p: (P, 3)).
            m = np.asarray(m).astype(bool)
            if m.shape != p.shape[:-1]:
                try:
                    m = np.reshape(m, p.shape[:-1])
                except Exception:
                    continue
            if m.sum() == 0:
                continue
            p_sel = p[m]
            t_sel = t[m]
            if getattr(p_sel, "ndim", 0) != 2 or getattr(t_sel, "ndim", 0) != 2:
                continue
            if p_sel.shape[0] == 0 or t_sel.shape[0] == 0:
                continue
            accuracy += compute_accuracy(t_sel, p_sel)
            completion += compute_completion(t_sel, p_sel)
            cnt += 1
        if cnt > 0:
            accuracy /= cnt
            completion /= cnt
        overall = (accuracy + completion) / 2
    else:
        accuracy = accuracy(prediction[mask], target[mask])
        completion = completion(prediction[mask], target[mask])
        overall = (accuracy + completion) / 2

    return dotdict(
        accuracy=accuracy,
        completion=completion,
        overall=overall,
    )


def point_accuracy_completion_overall_noalign(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    batch: dotdict = None,
    eps: float = 1e-5,
    batching: bool = True,
):
    # Same as point_accuracy_completion_overall, but without Umeyama alignment.
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if mask is None:
        mask = np.ones(prediction.shape[:-1], dtype=bool)
    else:
        mask = np.asarray(mask)

    if batching:
        if prediction.ndim == 2 and prediction.shape[-1] == 3:
            prediction = prediction[None, ...]
        if target.ndim == 2 and target.shape[-1] == 3:
            target = target[None, ...]
        if mask.ndim == 1:
            mask = mask[None, ...]
        if mask.ndim > 1 and mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)

    accuracy = 0.0
    completion = 0.0

    if batching:
        cnt = 0
        for p, t, m in zip(prediction, target, mask):
            m = np.asarray(m).astype(bool)
            if m.shape != p.shape[:-1]:
                try:
                    m = np.reshape(m, p.shape[:-1])
                except Exception:
                    continue
            if m.sum() == 0:
                continue
            p_sel = p[m]
            t_sel = t[m]
            if getattr(p_sel, "ndim", 0) != 2 or getattr(t_sel, "ndim", 0) != 2:
                continue
            if p_sel.shape[0] == 0 or t_sel.shape[0] == 0:
                continue
            accuracy += compute_accuracy(t_sel, p_sel)
            completion += compute_completion(t_sel, p_sel)
            cnt += 1
        if cnt > 0:
            accuracy /= cnt
            completion /= cnt
        overall = (accuracy + completion) / 2
    else:
        m = np.asarray(mask).astype(bool)
        if m.ndim > 1 and m.shape[-1] == 1:
            m = np.squeeze(m, axis=-1)
        if m.shape != prediction.shape[:-1]:
            m = np.reshape(m, prediction.shape[:-1])
        p_sel = prediction[m]
        t_sel = target[m]
        if p_sel.ndim != 2 or t_sel.ndim != 2 or p_sel.shape[0] == 0 or t_sel.shape[0] == 0:
            return dotdict(accuracy=0.0, completion=0.0, overall=0.0)
        accuracy = compute_accuracy(t_sel, p_sel)
        completion = compute_completion(t_sel, p_sel)
        overall = (accuracy + completion) / 2

    return dotdict(
        accuracy=accuracy,
        completion=completion,
        overall=overall,
    )


def point_scene_accuracy_completion_overall(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    batch: dotdict = None,
    eps: float = 1e-5,
    batching: bool = True,
):
    # Align once at scene level, then evaluate on the aggregated point cloud.
    prediction = align_xyz_umeyama(prediction, target, mask)

    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy().squeeze()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy().squeeze()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy().squeeze()
    elif mask is None:
        mask = np.ones_like(target[..., 0], dtype=bool).squeeze()

    if prediction.shape[:mask.ndim] != mask.shape:
        try:
            mask = np.reshape(mask, prediction.shape[:-1])
        except Exception:
            mask = np.reshape(mask, (-1,))

    p_sel = prediction[mask]
    t_sel = target[mask]
    if getattr(p_sel, "ndim", 0) != 2 or getattr(t_sel, "ndim", 0) != 2:
        return dotdict(accuracy=0.0, completion=0.0, overall=0.0)
    if p_sel.shape[0] == 0 or t_sel.shape[0] == 0:
        return dotdict(accuracy=0.0, completion=0.0, overall=0.0)

    max_points = int(os.environ.get("EVC_XYZ_SCENE_MAX_POINTS", "0") or 0)
    if max_points > 0 and p_sel.shape[0] > max_points:
        rng = np.random.default_rng(20260318)
        idx = rng.choice(p_sel.shape[0], size=max_points, replace=False)
        p_sel = p_sel[idx]
        t_sel = t_sel[idx]

    accuracy = compute_accuracy(t_sel, p_sel)
    completion = compute_completion(t_sel, p_sel)
    overall = (accuracy + completion) / 2
    return dotdict(
        accuracy=accuracy,
        completion=completion,
        overall=overall,
    )


def point_scene_accuracy_completion_overall_noalign(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    batch: dotdict = None,
    eps: float = 1e-5,
    batching: bool = True,
):
    # Scene-level evaluation without Umeyama alignment.
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy().squeeze()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy().squeeze()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy().squeeze()
    elif mask is None:
        mask = np.ones_like(target[..., 0], dtype=bool).squeeze()

    if prediction.shape[:mask.ndim] != mask.shape:
        try:
            mask = np.reshape(mask, prediction.shape[:-1])
        except Exception:
            mask = np.reshape(mask, (-1,))

    p_sel = prediction[mask]
    t_sel = target[mask]
    if getattr(p_sel, "ndim", 0) != 2 or getattr(t_sel, "ndim", 0) != 2:
        return dotdict(accuracy=0.0, completion=0.0, overall=0.0)
    if p_sel.shape[0] == 0 or t_sel.shape[0] == 0:
        return dotdict(accuracy=0.0, completion=0.0, overall=0.0)

    max_points = int(os.environ.get("EVC_XYZ_SCENE_MAX_POINTS", "0") or 0)
    if max_points > 0 and p_sel.shape[0] > max_points:
        rng = np.random.default_rng(20260318)
        idx = rng.choice(p_sel.shape[0], size=max_points, replace=False)
        p_sel = p_sel[idx]
        t_sel = t_sel[idx]

    accuracy = compute_accuracy(t_sel, p_sel)
    completion = compute_completion(t_sel, p_sel)
    overall = (accuracy + completion) / 2
    return dotdict(
        accuracy=accuracy,
        completion=completion,
        overall=overall,
    )


def point_metric(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    batch: dotdict = None,
    eps: float = 1e-5,
    batching: bool = True,
):
    # Align the predicted point to the ground truth point
    # Using the ROE optimal alignment method here
    if batching:
        prediction = np.stack([
            align_xyz_umeyama(p, t, m)
                for p, t, m in zip(prediction, target, mask)
        ], axis=0)
    else:
        prediction = align_xyz_umeyama(
            prediction, target, mask
        )

    # Convert the prediction to numpy array
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy().squeeze()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy().squeeze()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy().squeeze()

    # Compute the accuracy and completion
    l1 = 0.0
    rmse = 0.0
    abs_rel = 0.0

    if batching:
        cnt = 0
        for p, t, m in zip(prediction, target, mask):
            if m.sum() == 0:
                continue
            l1 += np.mean(np.abs(p[m] - t[m]))
            rmse += np.sqrt(np.mean((p[m] - t[m]) ** 2))
            abs_rel += np.mean(np.abs((p[m] - t[m]) / (t[m] + eps)))
            cnt += 1
        if cnt != 0:
            l1 /= cnt
            rmse /= cnt
            abs_rel /= cnt
    else:
        l1 = np.mean(np.abs(prediction[mask] - target[mask]))
        rmse = np.sqrt(np.mean((prediction[mask] - target[mask]) ** 2))
        abs_rel = np.mean(np.abs((prediction[mask] - target[mask]) / (target[mask] + eps)))

    return dotdict(
        l1=l1,
        rmse=rmse,
        abs_rel=abs_rel,
    )


def align_xyz_umeyama(
    src_xyz: torch.Tensor,
    tar_xyz: torch.Tensor,
    msk: Optional[torch.Tensor] = None,
):
    """Umeyama alignment for arbitrary [..., 3] point layouts.

    The legacy implementation assumed shape (N, 3). DTU/ETH3D scene-level evals
    pass dense XYZ maps shaped (H, W, 3), so flatten the spatial axes, align in
    point space, then reshape back.
    """
    if isinstance(src_xyz, torch.Tensor):
        src_xyz = src_xyz.detach().cpu().numpy().squeeze()
    if isinstance(tar_xyz, torch.Tensor):
        tar_xyz = tar_xyz.detach().cpu().numpy().squeeze()

    if msk is not None and isinstance(msk, torch.Tensor):
        msk = msk.detach().cpu().numpy().squeeze()
    elif msk is None:
        msk = np.ones_like(tar_xyz[..., 0], dtype=bool).squeeze()

    src_shape = src_xyz.shape
    tar_shape = tar_xyz.shape
    if src_shape[-1] != 3 or tar_shape[-1] != 3:
        return src_xyz

    src_flat = np.reshape(src_xyz, (-1, 3))
    tar_flat = np.reshape(tar_xyz, (-1, 3))
    msk_flat = np.reshape(msk, (-1,)).astype(bool)

    if msk_flat.sum() == 0:
        return src_xyz

    x = src_flat[msk_flat]
    y = tar_flat[msk_flat]
    if x.shape[0] == 0 or y.shape[0] == 0:
        return src_xyz

    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x = x - x_mean
    y = y - y_mean

    cov = (x.T @ y) / x.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    V = Vt.T
    S = np.eye(3)
    if np.linalg.det(V @ U.T) < 0:
        S[2, 2] = -1
    R = V @ S @ U.T

    denom = (np.sum(x ** 2) / x.shape[0])
    if denom <= 0:
        return src_xyz
    scale = np.sum(D * np.diag(S)) / denom
    t = y_mean - scale * R @ x_mean

    res_flat = scale * (src_flat @ R.T) + t
    return np.reshape(res_flat, src_shape)


@torch.no_grad()
def camera_ate(
    x: torch.Tensor,
    y: torch.Tensor,
    batch: dotdict = None,
):
    # Decode the camera parameters
    x_w2c, x_ixt = decode_camera_params(
        x, batch.meta.H[0].item(), batch.meta.W[0].item()
    )  # (B, S, 3, 4), (B, S, 3, 3)
    y_w2c, y_ixt = decode_camera_params(
        y, batch.meta.H[0].item(), batch.meta.W[0].item()
    )  # (B, S, 3, 4), (B, S, 3, 3)

    # Record the metrics
    metric = dotdict()

    B = x.shape[0]
    rotation = []
    translation = []
    for b in range(B):
        x_mat = x_w2c[b]
        y_mat = y_w2c[b]
        # Compute the difference in rotation and translation
        rotation.append(compute_rotation_angle(
            x_mat[..., :3, :3], y_mat[..., :3, :3]
        ))
        translation.append(torch.norm(x_mat[..., :3, 3] - y_mat[..., :3, 3], 2, dim=-1))

    # Stack the results
    rotation = torch.stack(rotation, dim=0)  # (B, P)
    translation = torch.stack(translation, dim=0)  # (B, P)

    # Keep naming consistent with metric semantics:
    # - ATE_trans: translational error
    # - ATE_angle: rotational error
    metric['ATE_trans'] = translation.mean()
    metric['ATE_angle'] = rotation.mean()

    return metric


class Metrics(Enum):
    PSNR = psnr
    SSIM = ssim
    LPIPS = lpips
    CD = chamfer_distance
    ND = distance
    DAC = distance_accuracy
    CAM_ACC_AUC = camera_accuracy_auc
    CAM_ACC_AUC_VGGT_EVAL = camera_accuracy_auc_vggt_eval
    CAM_EVO = camera_evo
    CAM_EVO_PI3 = camera_evo_pi3
    CAM_TRAJ_EVO = camera_traj_evo
    CAM_KITTI_ODOM = camera_kitti_odometry
    CAM_ATE = camera_ate
    DPT = depth_metric
    DPT_TARGET = target_depth_metric
    MVS_DPT_DYNAMIC = mvs_depth_dynamic_metric
    XYZ = point_metric
    XYZ_ACO = point_accuracy_completion_overall
    XYZ_ACO_NA = point_accuracy_completion_overall_noalign
    XYZ_SACO = point_scene_accuracy_completion_overall
    XYZ_SACO_NA = point_scene_accuracy_completion_overall_noalign
