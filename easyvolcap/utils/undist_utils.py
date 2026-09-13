"""
Use pycolmap camera models to match COLMAP-style image undistortion.
"""

import cv2
import torch
import pycolmap
import numpy as np
import torch.nn.functional as F
from functools import lru_cache

from easyvolcap.utils.colmap_utils import CAMERA_MODEL_NAMES


def _normalize_camera_model(camera_model: str | int | None, D: np.ndarray) -> str:
    if camera_model is None:
        dsize = int(np.asarray(D).size)
        if dsize == 0:
            return 'PINHOLE'
        if dsize in (1, 2):
            return 'RADIAL'
        if dsize == 4:
            return 'OPENCV'
        if dsize == 5:
            return 'FULL_OPENCV'
        if dsize == 8:
            return 'THIN_PRISM_FISHEYE'
        raise ValueError(f'Cannot infer COLMAP camera model from distortion size {dsize}')
    if isinstance(camera_model, int):
        return CAMERA_MODEL_NAMES[camera_model].model_name
    camera_model = camera_model.upper()
    if camera_model not in CAMERA_MODEL_NAMES:
        raise KeyError(f'Unsupported COLMAP camera model: {camera_model}')
    return camera_model


def _compose_camera_params(K: np.ndarray, D: np.ndarray, camera_model: str) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    zeros = np.zeros(8, dtype=np.float64)
    zeros[:min(len(D), 8)] = D[:8]
    if camera_model == 'SIMPLE_PINHOLE':
        return np.array([0.5 * (fx + fy), cx, cy], dtype=np.float64)
    if camera_model == 'PINHOLE':
        return np.array([fx, fy, cx, cy], dtype=np.float64)
    if camera_model == 'SIMPLE_RADIAL':
        return np.array([0.5 * (fx + fy), cx, cy, zeros[0]], dtype=np.float64)
    if camera_model == 'RADIAL':
        return np.array([0.5 * (fx + fy), cx, cy, zeros[0], zeros[1]], dtype=np.float64)
    if camera_model == 'OPENCV':
        return np.array([fx, fy, cx, cy, zeros[0], zeros[1], zeros[2], zeros[3]], dtype=np.float64)
    if camera_model == 'FULL_OPENCV':
        return np.array([fx, fy, cx, cy, zeros[0], zeros[1], zeros[2], zeros[3], zeros[4], zeros[5], zeros[6], zeros[7]], dtype=np.float64)
    if camera_model == 'THIN_PRISM_FISHEYE':
        return np.array([fx, fy, cx, cy, zeros[0], zeros[1], zeros[2], zeros[3], zeros[4], zeros[5], zeros[6], zeros[7]], dtype=np.float64)
    raise NotImplementedError(f'Camera model not implemented for undistortion: {camera_model}')


def _make_colmap_camera(H: int, W: int, K: np.ndarray, D: np.ndarray, camera_model: str) -> pycolmap.Camera:
    model = CAMERA_MODEL_NAMES[camera_model]
    camera = pycolmap.Camera.create(0, model.model_id, 0, W, H)
    camera.params = _compose_camera_params(K, D, camera_model)
    return camera


@lru_cache(maxsize=256)
def compute_mapping(H: int, W: int, fx: float, fy: float, cx: float, cy: float,
                    camera_model: str, camera_params: tuple[float, ...], blank_pixels: bool):
    src_camera = pycolmap.Camera.create(0, CAMERA_MODEL_NAMES[camera_model].model_id, 0, W, H)
    src_camera.params = np.asarray(camera_params, dtype=np.float64)

    undistorted_camera = pycolmap.Camera.create(0, CAMERA_MODEL_NAMES['PINHOLE'].model_id, 0, W, H)
    undistorted_camera.params = np.array([fx, fy, cx, cy], dtype=np.float64)

    ys = np.arange(H, dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)
    left_border = np.stack([np.full_like(ys, 0.5), ys + 0.5], axis=-1)
    right_border = np.stack([np.full_like(ys, W - 0.5), ys + 0.5], axis=-1)
    top_border = np.stack([xs + 0.5, np.full_like(xs, 0.5)], axis=-1)
    bottom_border = np.stack([xs + 0.5, np.full_like(xs, H - 0.5)], axis=-1)

    left_ud = undistorted_camera.img_from_cam(src_camera.cam_from_img(left_border))
    right_ud = undistorted_camera.img_from_cam(src_camera.cam_from_img(right_border))
    top_ud = undistorted_camera.img_from_cam(src_camera.cam_from_img(top_border))
    bottom_ud = undistorted_camera.img_from_cam(src_camera.cam_from_img(bottom_border))

    left_min_x, left_max_x = left_ud[:, 0].min(), left_ud[:, 0].max()
    right_min_x, right_max_x = right_ud[:, 0].min(), right_ud[:, 0].max()
    top_min_y, top_max_y = top_ud[:, 1].min(), top_ud[:, 1].max()
    bottom_min_y, bottom_max_y = bottom_ud[:, 1].min(), bottom_ud[:, 1].max()

    min_scale_x = min(cx / (cx - left_min_x), (W - 0.5 - cx) / (right_max_x - cx))
    min_scale_y = min(cy / (cy - top_min_y), (H - 0.5 - cy) / (bottom_max_y - cy))
    max_scale_x = max(cx / (cx - left_max_x), (W - 0.5 - cx) / (right_min_x - cx))
    max_scale_y = max(cy / (cy - top_max_y), (H - 0.5 - cy) / (bottom_min_y - cy))

    blank_weight = 1.0 if blank_pixels else 0.0
    scale_x = 1.0 / (min_scale_x * blank_weight + max_scale_x * (1.0 - blank_weight))
    scale_y = 1.0 / (min_scale_y * blank_weight + max_scale_y * (1.0 - blank_weight))

    Wt = int(max(1.0, scale_x * W))
    Ht = int(max(1.0, scale_y * H))
    cxt = Wt / W * cx
    cyt = Ht / H * cy

    target_camera = pycolmap.Camera.create(0, CAMERA_MODEL_NAMES['PINHOLE'].model_id, 0, Wt, Ht)
    target_camera.params = np.array([fx, fy, cxt, cyt], dtype=np.float64)

    x, y = np.meshgrid(np.arange(Wt, dtype=np.float64) + 0.5, np.arange(Ht, dtype=np.float64) + 0.5, indexing='xy')
    target_pixels = np.stack([x.reshape(-1), y.reshape(-1)], axis=-1)
    src = src_camera.img_from_cam(target_camera.cam_from_img(target_pixels)).reshape(Ht, Wt, 2)

    Kt = np.array([
        [fx, 0.0, cxt],
        [0.0, fy, cyt],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return src.astype(np.float32), Kt


def colmap_undistort_numpy(img: np.ndarray, K: np.ndarray, D: np.ndarray,
                           blank_pixels: bool = False, device: str = 'cpu',
                           camera_model: str | int | None = None,
                           interpolation: int = cv2.INTER_LINEAR):
    dst, Kt = colmap_undistort(
        torch.from_numpy(img).to(device, non_blocking=True),
        torch.from_numpy(K).to(device, non_blocking=True),
        torch.from_numpy(D).to(device, non_blocking=True),
        blank_pixels=blank_pixels,
        camera_model=camera_model,
        interpolation=interpolation,
    )
    return dst.cpu().numpy(), Kt.cpu().numpy()


def colmap_undistort(img: torch.Tensor, K: torch.Tensor, D: torch.Tensor,
                     blank_pixels: bool = False,
                     camera_model: str | int | None = None,
                     interpolation: int = cv2.INTER_LINEAR):
    is_single_channel = img.ndim == 2
    if is_single_channel:
        H, W = img.shape
    else:
        H, W = img.shape[-3:-1]
    K_np = K.detach().cpu().numpy()
    D_np = D.detach().cpu().numpy()
    camera_model = _normalize_camera_model(camera_model, D_np)
    camera_params = tuple(_compose_camera_params(K_np, D_np, camera_model).tolist())
    fx, fy, cx, cy = K_np[0, 0].item(), K_np[1, 1].item(), K_np[0, 2].item(), K_np[1, 2].item()

    src, Kt = compute_mapping(H, W, fx, fy, cx, cy, camera_model, camera_params, blank_pixels)

    if img.device.type == 'cuda':
        src_t = torch.from_numpy(src).to(img.device, non_blocking=True)
        src_t = src_t / torch.as_tensor([W, H], dtype=torch.float32, device=img.device) * 2 - 1
        mode = 'nearest' if interpolation == cv2.INTER_NEAREST else 'bilinear'
        if is_single_channel:
            img_t = img.float()[None, None]
            dst = F.grid_sample(img_t, src_t[None], mode=mode, align_corners=False)[0, 0]
        else:
            dst = F.grid_sample(img.float().permute(2, 0, 1)[None], src_t[None], mode=mode, align_corners=False)[0].permute(1, 2, 0)
    else:
        dst = cv2.remap(img.detach().cpu().numpy(), src[..., 0], src[..., 1], interpolation)
        dst = torch.from_numpy(dst)
    return dst, torch.from_numpy(Kt).to(img.device, non_blocking=True)
