#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate PI3-style multi-view reconstruction metrics (Sim3 + ICP + Acc/Comp/NC).

Supported datasets:
- DTU (raw official layout or EVC layout; seq-id-map kf5)
- ETH3D (seq-id-map kf5)
- 7-Scenes (sparse/dense seq-id-map)
- NRGBD (sparse/dense seq-id-map)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import torchvision.transforms as tvf
from PIL import Image

from easyvolcap.engine import Config
from easyvolcap.models.official_vggt_model import OfficialVGGTModel
from easyvolcap.official_vggt.utils.load_fn import load_and_preprocess_images as load_and_preprocess_vggt_images
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from aidi.scripts.baselines.eval_config_utils import load_resolved_config

TO_TENSOR = tvf.ToTensor()
DEFAULT_PI3_MODEL = "yyfz233/Pi3"
DEFAULT_PI3_ROOT = "tmp/external_refs/pi3-official2"
DEFAULT_DTU_ROOT = "tmp/dtu_test_mvsnet_release"
DEFAULT_ETH3D_ROOT = "tmp/eth3d_pi3_style_root"
DEFAULT_DTU_SHARED_ROOT = (
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/eval_datasets/dtu_test_mvsnet_release_full"
)
DEFAULT_ETH3D_SHARED_ROOT = (
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/eval_datasets/eth3d_pi3_style_root"
)
DEFAULT_ETH3D_SHARED_SEQMAP = (
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/eval_datasets/eth3d_pi3_style_seqmap.json"
)
DEFAULT_7SCENES_ROOT = "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/tao02.xie/datasets/7scenes/test"
DEFAULT_7SCENES_ROOT_ALT = "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/zinan.lv/dataset_point/7Scenes"
DEFAULT_NRGBD_ROOT = "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/zinan.lv/dataset_point/NRGBD"
DEFAULT_VGGT_OFFICIAL_ROOT = (
    "/horizon-bucket/saturn_v_dev/01_users/tao02.xie/projects/GFM/gfm/data/trained_model/vggt/official/VGGT-1B"
)
DEFAULT_VGGT_PT34_CKPT = (
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/trained_model/vggt/official/"
    "finetune_5090_sparse_20260226_0139_resume_finetune_5090_sparse_topk2048-layers9_19-resumept30-fp16-pointhead/34.pt"
)
DEFAULT_DA3_REPO_CANDIDATES = (
    REPO_ROOT / "third_party" / "Depth-Anything-3",
    REPO_ROOT / "tmp" / "depthanything3_fetch" / "Depth-Anything-3",
)
DEFAULT_DA3_MODEL_CANDIDATES = (
    REPO_ROOT / "tmp" / "pretrained_official" / "depth-anything__DA3-SMALL",
    Path(
        "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/pretrained/depthanything3/depth-anything__DA3-SMALL"
    ),
)


def first_existing_dir(candidates: Tuple[Path, ...]) -> str:
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return str(candidates[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PI3-aligned mv_recon metrics on DTU/ETH3D/7-Scenes/NRGBD in EVC or raw format."
    )
    parser.add_argument("--dataset", type=str, default="dtu", choices=["dtu", "eth3d", "7scenes", "nrgbd"])
    parser.add_argument(
        "--protocol",
        type=str,
        default="auto",
        choices=["auto", "sparse", "dense", "both"],
        help="7scenes/nrgbd support sparse/dense/both. DTU/ETH3D use auto.",
    )
    parser.add_argument(
        "--model-family",
        type=str,
        default="vggt",
        choices=["pi3", "vggt", "da3", "vggt_omega"],
        help="Model family for inference.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="",
        help="Dataset root. Empty => default by --dataset.",
    )
    parser.add_argument(
        "--pi3-root",
        type=str,
        default=DEFAULT_PI3_ROOT,
        help="Path to PI3 official repo clone.",
    )
    parser.add_argument(
        "--seq-map",
        type=str,
        default="",
        help="Override seq-id-map path.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Checkpoint path or model id. For PI3: local ckpt or HF id.",
    )
    parser.add_argument(
        "--vggt-model-tag",
        type=str,
        default="official",
        choices=["official", "pt34", "custom"],
        help="VGGT checkpoint mode when --model-family=vggt. pt34 is kept for legacy compatibility.",
    )
    parser.add_argument(
        "--vggt-config",
        type=str,
        default="",
        help="Optional VGGT config path; defaults by vggt-model-tag.",
    )
    parser.add_argument(
        "--vggt-checkpoint",
        type=str,
        default="",
        help="Custom full VGGT checkpoint path. If set, custom loading mode is used.",
    )
    parser.add_argument(
        "--vggt-official-ckpt-root",
        type=str,
        default=DEFAULT_VGGT_OFFICIAL_ROOT,
        help="Root with official VGGT component checkpoints.",
    )
    parser.add_argument(
        "--vggt-pt34-ckpt",
        type=str,
        default=DEFAULT_VGGT_PT34_CKPT,
        help="Full pt34 OfficialVGGTModel checkpoint path.",
    )
    parser.add_argument(
        "--da3-repo",
        type=str,
        default=first_existing_dir(DEFAULT_DA3_REPO_CANDIDATES),
        help="Depth Anything 3 official repository clone.",
    )
    parser.add_argument(
        "--da3-model",
        type=str,
        default=first_existing_dir(DEFAULT_DA3_MODEL_CANDIDATES),
        help="Depth Anything 3 local model snapshot. No random-init fallback is used.",
    )
    parser.add_argument("--da3-process-res", type=int, default=504)
    parser.add_argument(
        "--da3-process-res-method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "upper_bound_crop", "lower_bound_resize", "lower_bound_crop"],
    )
    parser.add_argument("--vggt-omega-repo", type=str, default="")
    parser.add_argument("--vggt-omega-checkpoint", type=str, default="")
    parser.add_argument("--vggt-omega-resolution", type=int, default=512)
    parser.add_argument("--vggt-omega-mode", choices=["balanced", "max_size"], default="balanced")
    parser.add_argument(
        "--point-source",
        type=str,
        default="depth_pose",
        choices=["native", "depth_pose"],
        help="Point source. native=point head output; depth_pose=unproject from depth+pose.",
    )
    parser.add_argument(
        "--vggt-input-style",
        type=str,
        default="official_crop",
        choices=["official_crop", "pi3_resize"],
        help="VGGT image preprocessing for point-map eval. pi3_resize matches Pi3/CUT3R resize semantics.",
    )
    parser.add_argument("--load-img-size", type=int, default=518)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument(
        "--dtu-unit-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to DTU GT depth/camera translation before GT pointmap unprojection.",
    )
    parser.add_argument(
        "--dtu-mask-erode",
        type=int,
        default=10,
        help="DTU mask erosion kernel size; <=0 disables erosion.",
    )
    parser.add_argument(
        "--dtu-center-crop-height",
        type=int,
        default=0,
        help="Optional center-crop height applied to DTU RGB/depth/mask before resize; 0 disables.",
    )
    parser.add_argument(
        "--dtu-data-format",
        type=str,
        default="auto",
        choices=["auto", "evc", "raw"],
        help="DTU data layout. raw follows Pi3/CUT3R official DTU loader; evc follows current repo layout.",
    )
    parser.add_argument(
        "--eth3d-extri-c2w",
        action="store_true",
        help="Treat ETH3D extrinsics in intri/extri.yml as camera-to-world (c2w).",
    )
    parser.add_argument(
        "--icp-threshold",
        type=float,
        default=0.1,
        help="ICP threshold for non-DTU datasets.",
    )
    parser.add_argument(
        "--icp-threshold-dtu",
        type=float,
        default=100.0,
        help="ICP threshold for DTU (follow PI3 eval.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory. Empty => tmp/pi3_mv_recon_<dataset>_<timestamp>.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="Debug helper. >0 limits the number of seq-id-map entries evaluated per protocol.",
    )
    parser.add_argument(
        "--eval-frame-indices",
        type=str,
        default="",
        help=(
            "Optional comma-separated input indices used for metrics after inference. "
            "All seq-map frames are still fed to the model; only selected indices are scored."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def parse_eval_frame_indices(value: str) -> List[int]:
    raw = str(value or "").strip()
    if not raw:
        return []
    indices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if any(index < 0 for index in indices):
        raise ValueError(f"eval-frame-indices must be non-negative: {indices}")
    return indices


def subset_arrays_for_metric(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    valid_mask: np.ndarray,
    images: torch.Tensor,
    eval_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    if not eval_indices:
        return pred_pts, gt_pts, valid_mask, images
    if max(eval_indices) >= int(pred_pts.shape[0]):
        raise IndexError(
            f"eval-frame-indices include {max(eval_indices)}, but prediction has {pred_pts.shape[0]} frames"
        )
    index = np.asarray(eval_indices, dtype=np.int64)
    return pred_pts[index], gt_pts[index], valid_mask[index], images[index]


def default_dataset_root(dataset: str) -> str:
    if dataset == "dtu":
        for candidate in [DEFAULT_DTU_ROOT, DEFAULT_DTU_SHARED_ROOT]:
            if Path(candidate).is_dir():
                return candidate
        return DEFAULT_DTU_ROOT
    if dataset == "eth3d":
        for candidate in [DEFAULT_ETH3D_ROOT, DEFAULT_ETH3D_SHARED_ROOT]:
            if Path(candidate).is_dir():
                return candidate
        return DEFAULT_ETH3D_ROOT
    if dataset == "7scenes":
        for candidate in [DEFAULT_7SCENES_ROOT, DEFAULT_7SCENES_ROOT_ALT]:
            if Path(candidate).is_dir():
                return candidate
        return DEFAULT_7SCENES_ROOT
    if dataset == "nrgbd":
        return DEFAULT_NRGBD_ROOT
    raise ValueError(dataset)


def resolve_protocols(dataset: str, protocol: str) -> List[str]:
    if dataset == "7scenes":
        if protocol == "auto":
            return ["7scenes-sparse", "7scenes-dense"]
        if protocol == "sparse":
            return ["7scenes-sparse"]
        if protocol == "dense":
            return ["7scenes-dense"]
        if protocol == "both":
            return ["7scenes-sparse", "7scenes-dense"]
        raise ValueError(f"Unsupported protocol={protocol} for dataset={dataset}")
    if dataset == "nrgbd":
        if protocol == "auto":
            return ["nrgbd-sparse", "nrgbd-dense"]
        if protocol == "sparse":
            return ["nrgbd-sparse"]
        if protocol == "dense":
            return ["nrgbd-dense"]
        if protocol == "both":
            return ["nrgbd-sparse", "nrgbd-dense"]
        raise ValueError(f"Unsupported protocol={protocol} for dataset={dataset}")
    if protocol not in {"auto", "both"}:
        raise ValueError(f"dataset={dataset} only supports --protocol auto/both")
    if dataset == "dtu":
        return ["DTU"]
    if dataset == "eth3d":
        return ["ETH3D"]
    raise ValueError(dataset)


def resolve_seq_map(pi3_root: Path, dataset: str, protocol: str, seq_map_override: str = "") -> Path:
    if seq_map_override:
        return Path(seq_map_override).resolve()
    local_seq_root = REPO_ROOT / "aidi" / "assets" / "pi3_seq_id_maps"
    seq_root = pi3_root / "datasets" / "seq-id-maps"
    if protocol == "7scenes-sparse":
        local_path = local_seq_root / "7scenes_mv-recon_seq-id-map-kf200.json"
        if local_path.is_file():
            return local_path.resolve()
        return (seq_root / "7scenes_mv-recon_seq-id-map-kf200.json").resolve()
    if protocol == "7scenes-dense":
        local_path = local_seq_root / "7scenes_mv-recon_seq-id-map-kf40.json"
        if local_path.is_file():
            return local_path.resolve()
        return (seq_root / "7scenes_mv-recon_seq-id-map-kf40.json").resolve()
    if protocol == "nrgbd-sparse":
        local_path = local_seq_root / "NRGBD_mv-recon_seq-id-map-kf500.json"
        if local_path.is_file():
            return local_path.resolve()
        return (seq_root / "NRGBD_mv-recon_seq-id-map-kf500.json").resolve()
    if protocol == "nrgbd-dense":
        local_path = local_seq_root / "NRGBD_mv-recon_seq-id-map-kf100.json"
        if local_path.is_file():
            return local_path.resolve()
        return (seq_root / "NRGBD_mv-recon_seq-id-map-kf100.json").resolve()
    if protocol == "DTU":
        return (local_seq_root / "DTU_mv-recon_seq-id-map-kf5.json").resolve()
    if protocol == "ETH3D":
        shared_seq = Path(DEFAULT_ETH3D_SHARED_SEQMAP)
        if shared_seq.is_file():
            return shared_seq.resolve()
        return (local_seq_root / "ETH3D_mv-recon_seq-id-map-kf5.json").resolve()
    raise ValueError(f"Unsupported protocol={protocol} for dataset={dataset}")


def default_pi3_ckpt() -> str:
    env_ckpt = os.environ.get("PI3_CKPT", "").strip()
    if env_ckpt:
        return env_ckpt

    user_name = os.environ.get("USER", "feng01.zhou")
    candidates = [
        Path(
            f"/horizon-bucket/saturn_v_dev/01_users/{user_name}/projects/meshx/baseline/pretrained/pi3/yyfz233_Pi3_model.safetensors"
        ),
        Path("weights/pi3/model.safetensors"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return DEFAULT_PI3_MODEL


def import_local_pi3_metric_utils():
    from aidi.scripts.baselines.pi3_metric_utils import accuracy, completion, umeyama

    return umeyama, accuracy, completion


def import_pi3_model(pi3_root: Path):
    pi3_root = pi3_root.resolve()
    if not pi3_root.is_dir():
        raise FileNotFoundError(f"PI3 root not found: {pi3_root}")
    sys.path.insert(0, str(pi3_root))
    from pi3.models.pi3 import Pi3

    return Pi3


def resize_image(image: Image.Image, output_resolution: Tuple[int, int]) -> Image.Image:
    max_resize_scale = max(output_resolution[0] / image.size[0], output_resolution[1] / image.size[1])
    resample = Image.Resampling.LANCZOS if max_resize_scale < 1 else Image.Resampling.BICUBIC
    return image.resize(output_resolution, resample=resample)


def resize_image_depth_and_intrinsic(
    image: Image.Image,
    depth_map: np.ndarray,
    intrinsic: np.ndarray,
    output_width: int,
    pixel_center: bool = True,
) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    if len(depth_map.shape) != 2:
        raise ValueError(f"Depth map must be 2D, got {depth_map.shape}")
    input_resolution = np.array(depth_map.shape[::-1], dtype=np.float32)  # (W,H)
    output_resolution = np.array(
        [output_width, round(input_resolution[1] * (output_width / input_resolution[0]) / 14) * 14]
    )
    image = resize_image(image, tuple(output_resolution))
    depth_map = cv2.resize(depth_map, tuple(output_resolution.astype(int)), interpolation=cv2.INTER_NEAREST)

    intrinsic = np.copy(intrinsic)
    if pixel_center:
        intrinsic[0, 2] += 0.5
        intrinsic[1, 2] += 0.5
    resize_scale = np.max(output_resolution / input_resolution)
    intrinsic[:2, :] = intrinsic[:2, :] * resize_scale
    if pixel_center:
        intrinsic[0, 2] -= 0.5
        intrinsic[1, 2] -= 0.5
    return image, depth_map, intrinsic


def load_images_for_pi3(filelist: List[str], new_width: int, device: str, verbose: bool):
    sources = [Image.open(img_path).convert("RGB") for img_path in filelist]
    if not sources:
        raise RuntimeError("No images loaded")

    w0, h0 = sources[0].size
    target_w = int(new_width)
    target_h = int(round(h0 * (new_width / w0) / 14) * 14)
    if verbose:
        print(f"[pi3-load] resized all inputs to ({target_w}, {target_h})", flush=True)

    imgs = []
    for img in sources:
        resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        imgs.append(TO_TENSOR(resized))
    tensor = torch.stack(imgs, dim=0).to(device)  # (N,3,H,W)

    patch_h, patch_w = tensor.shape[-2] // 14, tensor.shape[-1] // 14
    tensor = F.interpolate(
        tensor,
        (patch_h * 14, patch_w * 14),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).unsqueeze(0)
    return tensor


def load_images_for_vggt(
    filelist: List[str],
    load_img_size: int,
    device: str,
    verbose: bool,
    input_style: str = "official_crop",
):
    if input_style == "official_crop":
        tensor = load_and_preprocess_vggt_images(filelist, mode="crop", target_size=load_img_size).to(device)
        if verbose:
            print(f"[vggt-load] official_crop loader output shape={tuple(tensor.shape)}", flush=True)
        return tensor.unsqueeze(0)
    if input_style == "pi3_resize":
        tensor = load_images_for_pi3(filelist=filelist, new_width=load_img_size, device=device, verbose=verbose)
        if verbose:
            print(f"[vggt-load] pi3_resize loader output shape={tuple(tensor.shape)}", flush=True)
        return tensor
    raise ValueError(f"Unknown vggt input_style: {input_style}")


def extract_pi3_points_and_poses(
    pred,
    *,
    data_size: Tuple[int, int],
    point_source: str = "native",
) -> Dict[str, np.ndarray]:
    camera_poses = pred.get("camera_poses", None)
    if camera_poses is None:
        raise RuntimeError("PI3 output missing camera_poses")
    camera_poses = camera_poses[0]

    if point_source == "native":
        global_points = pred["points"][0]
    elif point_source == "depth_pose":
        local_points = pred.get("local_points", None)
        if local_points is None:
            raise RuntimeError("PI3 depth_pose mode requires local_points")
        local_points = local_points[0]
        ones = torch.ones_like(local_points[..., :1])
        local_h = torch.cat([local_points, ones], dim=-1)
        global_points = torch.einsum("nij,nhwj->nhwi", camera_poses, local_h)[..., :3]
    else:
        raise ValueError(f"Unknown point_source: {point_source}")

    global_points = F.interpolate(
        global_points.permute(0, 3, 1, 2),
        data_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).permute(0, 2, 3, 1)
    return {
        "points": global_points.detach().float().cpu().numpy(),
        "camera_poses": camera_poses.detach().float().cpu().numpy(),
    }


def infer_pi3_mv_predictions(
    filelist: List[str],
    model,
    load_img_size: int,
    device: str,
    verbose: bool,
    data_size: Tuple[int, int],
    point_source: str = "native",
    vggt_input_style: str = "official_crop",
):
    imgs = load_images_for_pi3(filelist=filelist, new_width=load_img_size, device=device, verbose=verbose)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast(device, dtype=dtype):
            pred = model(imgs)

    return extract_pi3_points_and_poses(pred, data_size=data_size, point_source=point_source)


def infer_pi3_mv_pointclouds(
    filelist: List[str],
    model,
    load_img_size: int,
    device: str,
    verbose: bool,
    data_size: Tuple[int, int],
    point_source: str = "native",
    vggt_input_style: str = "official_crop",
):
    return infer_pi3_mv_predictions(
        filelist=filelist,
        model=model,
        load_img_size=load_img_size,
        device=device,
        verbose=verbose,
        data_size=data_size,
        point_source=point_source,
        vggt_input_style=vggt_input_style,
    )["points"]


def infer_vggt_mv_pointclouds(
    filelist: List[str],
    model,
    load_img_size: int,
    device: str,
    verbose: bool,
    data_size: Tuple[int, int],
    point_source: str = "native",
    vggt_input_style: str = "official_crop",
):
    imgs = load_images_for_vggt(
        filelist=filelist,
        load_img_size=load_img_size,
        device=device,
        verbose=verbose,
        input_style=vggt_input_style,
    )
    h, w = imgs.shape[-2], imgs.shape[-1]
    batch = dotdict(
        images=imgs,
        meta=dotdict(
            iter=torch.tensor(0, device=imgs.device),
            H=torch.tensor([h], device=imgs.device),
            W=torch.tensor([w], device=imgs.device),
        ),
    )
    use_amp = str(device).startswith("cuda")
    dtype = None
    if use_amp:
        major, _ = torch.cuda.get_device_capability(device=torch.device(device))
        dtype = torch.bfloat16 if major >= 8 else torch.float16

    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                output = model(batch)
        else:
            output = model(batch)

    if point_source == "native":
        if not hasattr(output, "xyz_map"):
            raise RuntimeError("VGGT output missing xyz_map")
        xyz_flat = output.xyz_map[0]  # (N, HW, 3)
        n, hw, c = xyz_flat.shape
        if c != 3:
            raise RuntimeError(f"Unexpected xyz_map shape: {tuple(xyz_flat.shape)}")
        if hw != h * w:
            raise RuntimeError(f"Cannot reshape xyz_map with hw={hw} into ({h},{w})")
        xyz = xyz_flat.reshape(n, h, w, 3)
    elif point_source == "depth_pose":
        if (not hasattr(output, "dpt_map")) or (not hasattr(output, "cam_map")):
            raise RuntimeError("VGGT depth_pose mode requires dpt_map and cam_map")
        depth_flat = output.dpt_map[0]  # (N, HW, 1)
        n, hw, c = depth_flat.shape
        if c != 1:
            raise RuntimeError(f"Unexpected dpt_map shape: {tuple(depth_flat.shape)}")
        if hw != h * w:
            raise RuntimeError(f"Cannot reshape dpt_map with hw={hw} into ({h},{w})")
        depth = depth_flat.reshape(n, h, w, 1)
        pose_enc = output.cam_map[0]  # (N, 9)
        if pose_enc.ndim != 2 or pose_enc.shape[-1] != 9:
            raise RuntimeError(f"Unexpected cam_map shape for pose decoding: {tuple(output.cam_map.shape)}")

        pose_enc = pose_enc.unsqueeze(0)  # (1, N, 9)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=(h, w), pose_encoding_type="absT_quaR_FoV", build_intrinsics=True
        )
        depth_np = depth.detach().float().cpu().numpy()
        xyz_np = unproject_depth_map_to_point_map(
            depth_map=depth_np,
            extrinsics_cam=extrinsics[0].detach().float().cpu().numpy(),
            intrinsics_cam=intrinsics[0].detach().float().cpu().numpy(),
        )  # (N,h,w,3)
        xyz = torch.from_numpy(xyz_np).to(device=depth.device, dtype=depth.dtype)
    else:
        raise ValueError(f"Unknown point_source: {point_source}")

    xyz = F.interpolate(
        xyz.permute(0, 3, 1, 2),
        data_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).permute(0, 2, 3, 1)
    return xyz.detach().float().cpu().numpy()


def infer_vggt_omega_mv_pointclouds(
    filelist: List[str],
    model,
    load_img_size: int,
    device: str,
    verbose: bool,
    data_size: Tuple[int, int],
    point_source: str = "depth_pose",
    vggt_input_style: str = "official_crop",
):
    del load_img_size, verbose, vggt_input_style
    if point_source != "depth_pose":
        raise ValueError("VGGT-Omega mv-recon supports point_source=depth_pose only")
    from aidi.scripts.vggt.vggt_omega_eval_utils import depth_to_world_points, predict_vggt_omega

    repo_path = os.environ.get("VGGT_OMEGA_REPO", "").strip()
    image_resolution = int(os.environ.get("VGGT_OMEGA_RESOLUTION", "512"))
    mode = os.environ.get("VGGT_OMEGA_MODE", "balanced").strip() or "balanced"
    if not repo_path:
        raise ValueError("VGGT_OMEGA_REPO environment variable is required for VGGT-Omega inference")

    predictions, extrinsics, intrinsics = predict_vggt_omega(
        image_files=filelist,
        model=model,
        repo_path=repo_path,
        image_resolution=image_resolution,
        mode=mode,
        device=device,
    )
    world_points = depth_to_world_points(predictions["depth"], extrinsics, intrinsics)
    xyz = torch.from_numpy(world_points).to(device=device)
    xyz = F.interpolate(
        xyz.permute(0, 3, 1, 2),
        data_size,
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    return xyz.detach().float().cpu().numpy()


def _install_da3_optional_dependency_stubs() -> None:
    try:
        import moviepy.editor  # noqa: F401
    except ModuleNotFoundError:
        moviepy_module = types.ModuleType("moviepy")
        editor_module = types.ModuleType("moviepy.editor")

        class _UnsupportedImageSequenceClip:
            def __init__(self, *_args, **_kwargs) -> None:
                raise RuntimeError("moviepy is required for gs_video export.")

        editor_module.ImageSequenceClip = _UnsupportedImageSequenceClip
        moviepy_module.editor = editor_module
        sys.modules.setdefault("moviepy", moviepy_module)
        sys.modules.setdefault("moviepy.editor", editor_module)

    try:
        import pycolmap  # noqa: F401
    except ModuleNotFoundError:
        sys.modules.setdefault("pycolmap", types.ModuleType("pycolmap"))


def _deprioritize_stale_workspace_paths() -> None:
    sys.path[:] = [item for item in sys.path if "/workspace/meshx_5090" not in item]
    einops_module = sys.modules.get("einops")
    einops_file = str(getattr(einops_module, "__file__", ""))
    if "/workspace/meshx_5090" in einops_file:
        sys.modules.pop("einops", None)


def _to_numpy_float32(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def load_da3_model(args: argparse.Namespace, device: torch.device):
    da3_repo = Path(args.da3_repo).expanduser().resolve()
    src_path = da3_repo / "src"
    if not src_path.is_dir():
        raise FileNotFoundError(f"Depth Anything 3 src path not found: {src_path}")
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    _deprioritize_stale_workspace_paths()
    _install_da3_optional_dependency_stubs()

    from depth_anything_3.api import DepthAnything3

    model_path = Path(args.da3_model).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Depth Anything 3 model snapshot not found: {model_path}")
    model = DepthAnything3.from_pretrained(str(model_path)).to(device).eval()
    model._meshx_da3_process_res = int(args.da3_process_res)
    model._meshx_da3_process_res_method = str(args.da3_process_res_method)
    return model, str(model_path)


def infer_da3_mv_pointclouds(
    filelist: List[str],
    model,
    load_img_size: int,
    device: str,
    verbose: bool,
    data_size: Tuple[int, int],
    point_source: str = "depth_pose",
    vggt_input_style: str = "official_crop",
):
    del load_img_size, device, point_source, vggt_input_style
    if verbose:
        print(
            f"[da3-load] process_res={model._meshx_da3_process_res} "
            f"method={model._meshx_da3_process_res_method} views={len(filelist)}",
            flush=True,
        )
    with torch.inference_mode():
        prediction = model.inference(
            [str(path) for path in filelist],
            process_res=int(model._meshx_da3_process_res),
            process_res_method=str(model._meshx_da3_process_res_method),
            ref_view_strategy="first",
            use_ray_pose=False,
        )
    depth = _to_numpy_float32(prediction.depth)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise RuntimeError(f"Unexpected DA3 depth shape: {depth.shape}")
    extrinsics = _to_numpy_float32(prediction.extrinsics)[:, :3, :4]
    intrinsics = _to_numpy_float32(prediction.intrinsics)
    xyz_np = unproject_depth_map_to_point_map(
        depth_map=depth[..., None],
        extrinsics_cam=extrinsics,
        intrinsics_cam=intrinsics,
        invert_extrinsics=True,
    )
    xyz = torch.from_numpy(xyz_np).float()
    xyz = F.interpolate(
        xyz.permute(0, 3, 1, 2),
        data_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).permute(0, 2, 3, 1)
    return xyz.numpy()


def closed_form_inverse_se3(se3: np.ndarray) -> np.ndarray:
    if se3.shape[-2:] not in {(4, 4), (3, 4)}:
        raise ValueError(f"se3 must be Nx4x4 or Nx3x4, got {se3.shape}")
    R = se3[:, :3, :3]
    T = se3[:, :3, 3:]
    R_t = np.transpose(R, (0, 2, 1))
    top_right = -np.matmul(R_t, T)
    inv = np.tile(np.eye(4, dtype=se3.dtype), (len(R), 1, 1))
    inv[:, :3, :3] = R_t
    inv[:, :3, 3:] = top_right
    return inv


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    h, w = depth_map.shape
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map
    return np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray,
    extrinsics_cam: np.ndarray,
    intrinsics_cam: np.ndarray,
    invert_extrinsics: bool = True,
) -> np.ndarray:
    world_points = []
    for frame_idx in range(depth_map.shape[0]):
        depth = depth_map[frame_idx].squeeze(-1)
        intrinsic = intrinsics_cam[frame_idx]
        extrinsic = extrinsics_cam[frame_idx]
        cam_coords = depth_to_cam_coords_points(depth, intrinsic)

        if invert_extrinsics:
            cam_to_world = closed_form_inverse_se3(extrinsic[None])[0]
        else:
            cam_to_world = np.eye(4, dtype=extrinsic.dtype)
            cam_to_world[:3, :4] = extrinsic
        R_c2w = cam_to_world[:3, :3]
        t_c2w = cam_to_world[:3, 3]
        world = np.dot(cam_coords, R_c2w.T) + t_c2w
        world_points.append(world)
    return np.stack(world_points, axis=0)


def load_pi3_model(Pi3, ckpt: str, device: torch.device):
    from aidi.scripts.baselines.pi3_checkpoint_loader import is_native_pi3_requested, load_pi3_model_for_eval

    if Pi3 is None or os.environ.get("PI3_MODEL_IMPL", "").strip() or os.environ.get("PI3_CONFIG", "").strip() or is_native_pi3_requested(ckpt):
        return load_pi3_model_for_eval(
            ckpt,
            device,
            official_pi3_cls=Pi3,
            native_root=os.environ.get("PI3_NATIVE_ROOT", ""),
            model_impl=os.environ.get("PI3_MODEL_IMPL", ""),
            config_path=os.environ.get("PI3_CONFIG", ""),
        )

    ckpt_path = Path(ckpt)
    if ckpt_path.is_file():
        model = Pi3().to(device).eval()
        if ckpt_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(ckpt_path))
        else:
            state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        return model, str(ckpt_path)
    return Pi3.from_pretrained(ckpt).to(device).eval(), ckpt


def load_vggt_model(args: argparse.Namespace, device: torch.device) -> Tuple[OfficialVGGTModel, str]:
    use_custom_ckpt = bool(args.vggt_checkpoint) or args.vggt_model_tag in {"pt34", "custom"}

    if use_custom_ckpt:
        if args.vggt_checkpoint and not args.vggt_config:
            raise ValueError("--vggt-config is required when --vggt-checkpoint is set.")
        cfg_path = args.vggt_config or "aidi/configs/vggt/finetune_5090_sparse_topk2048_l9_19_p34_record.yaml"
    else:
        cfg_path = args.vggt_config or "configs/exps/vggt/vggt_official_eval_paper.yaml"

    cfg = load_resolved_config(cfg_path)
    model_cfg = dotdict(cfg.model_cfg)
    model_cfg.pop("type", None)
    model_cfg.pretrained_path = ""

    if not use_custom_ckpt:
        ckpt_root = Path(args.vggt_official_ckpt_root)
        model_cfg.agg_ckpt = str(ckpt_root / "aggregator.pt")
        model_cfg.cam_ckpt = str(ckpt_root / "camera.pt")
        model_cfg.xyz_ckpt = str(ckpt_root / "point.pt")
        model_cfg.dpt_ckpt = str(ckpt_root / "depth.pt")
        model_cfg.tra_ckpt = str(ckpt_root / "track.pt")
        model = OfficialVGGTModel(**model_cfg).to(device=device).eval()
        loaded_ckpt = str(ckpt_root)
    else:
        model = OfficialVGGTModel(**model_cfg).to(device=device).eval()
        ckpt_path = Path(args.vggt_checkpoint or args.vggt_pt34_ckpt)
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        state_dict, _ = OfficialVGGTModel._remap_special_token_keys(state_dict)
        model.load_state_dict(state_dict, strict=False)
        loaded_ckpt = str(ckpt_path)

    if args.point_source == "depth_pose" and hasattr(model, "vggt") and hasattr(model.vggt, "point_head"):
        model.vggt.point_head = None
    return model, loaded_ckpt


def load_vggt_omega_model(args: argparse.Namespace, device: torch.device):
    if not args.vggt_omega_repo:
        raise ValueError("--vggt-omega-repo is required when --model-family=vggt_omega")
    if not args.vggt_omega_checkpoint:
        raise ValueError("--vggt-omega-checkpoint is required when --model-family=vggt_omega")
    from aidi.scripts.vggt.vggt_omega_eval_utils import load_vggt_omega_model as _load

    return _load(
        repo_path=args.vggt_omega_repo,
        checkpoint_path=args.vggt_omega_checkpoint,
        device=device,
    )


def discover_sequences(dataset_root: Path, dataset: str = "") -> Dict[str, Path]:
    seq_dirs = {}
    for path in sorted(dataset_root.iterdir()):
        if not path.is_dir():
            continue
        seq_dirs[path.name] = path
    if dataset == "7scenes":
        for scene_dir in sorted(dataset_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            for sub in sorted(scene_dir.iterdir()):
                if not sub.is_dir():
                    continue
                nested_key = f"{scene_dir.name}_{sub.name}"
                if nested_key not in seq_dirs:
                    seq_dirs[nested_key] = sub
    return seq_dirs


def seq_key_to_dir_name(dataset: str, seq_key: str) -> str:
    if dataset == "7scenes":
        return seq_key.replace("/", "_")
    return seq_key


def normalize_seq_map_entry(dataset: str, seq_key: str, raw_entry) -> Tuple[str, List[int]]:
    """Support standard `scene -> [ids]` and diagnostic `key -> {scene, ids}` seq maps."""

    if isinstance(raw_entry, dict):
        if "ids" not in raw_entry:
            raise KeyError(f"Diagnostic seq-map entry {seq_key!r} is missing required key 'ids'")
        scene_key = str(raw_entry.get("scene") or seq_key)
        return seq_key_to_dir_name(dataset, scene_key), [int(item) for item in raw_entry["ids"]]
    return seq_key_to_dir_name(dataset, seq_key), [int(item) for item in raw_entry]


def load_sequence_cameras(dataset: str, seq_dir: Path):
    if dataset == "eth3d" and (seq_dir / "custom_undistorted_cam").is_dir():
        return None, None
    if dataset == "dtu" and (seq_dir / "cams").is_dir():
        return None, None
    if dataset == "dtu":
        intri = seq_dir / "cameras" / "00" / "intri.yml"
        extri = seq_dir / "cameras" / "00" / "extri.yml"
    else:
        intri = seq_dir / "intri.yml"
        extri = seq_dir / "extri.yml"
    if (not intri.is_file()) or (not extri.is_file()):
        raise FileNotFoundError(f"Missing camera files: {intri}, {extri}")
    cams = read_camera(str(intri), str(extri), use_dict=False)
    cam_names = sorted([k for k in cams.keys() if str(k).isdigit()], key=lambda x: int(x))
    return cams, cam_names


def _resolve_first_file(candidates: List[Path]) -> Path:
    for p in candidates:
        if p.is_file():
            return p
    return Path()


def _load_depth(depth_path: Path) -> np.ndarray:
    if depth_path.suffix.lower() == ".npy":
        depth = np.load(str(depth_path))
    else:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Failed reading depth: {depth_path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return depth


def _load_eth3d_official_depth(depth_path: Path, target_hw: Tuple[int, int]) -> np.ndarray:
    depth = np.fromfile(str(depth_path), dtype=np.float32)
    expected = int(target_hw[0] * target_hw[1])
    if depth.size != expected:
        raise RuntimeError(
            f"ETH3D official depth size mismatch for {depth_path}: got {depth.size}, expected {expected}"
        )
    depth = depth.reshape(target_hw[0], target_hw[1])
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return depth


def _load_mask(mask_path: Path, target_hw: Tuple[int, int], dtu_mask_erode: int) -> np.ndarray:
    if not mask_path.is_file():
        return np.ones(target_hw, dtype=np.float32)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return np.ones(target_hw, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    mask = cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0.5).astype(np.float32)
    if dtu_mask_erode > 0:
        kernel = np.ones((dtu_mask_erode, dtu_mask_erode), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    return mask


def _center_crop_dtu(
    rgb: Image.Image,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    crop_h: int,
):
    if crop_h <= 0:
        return rgb, depth, mask, intrinsic
    src_h, src_w = depth.shape[:2]
    if crop_h >= src_h:
        return rgb, depth, mask, intrinsic
    top = max((src_h - crop_h) // 2, 0)
    bottom = top + crop_h
    depth = depth[top:bottom]
    mask = mask[top:bottom]
    rgb = rgb.crop((0, top, src_w, bottom))
    intrinsic = intrinsic.copy()
    intrinsic[1, 2] -= float(top)
    return rgb, depth, mask, intrinsic


def _list_eth3d_official_images(seq_dir: Path) -> List[Path]:
    image_root = seq_dir / "images" / "custom_undistorted"
    image_list = sorted(
        [p for p in image_root.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not image_list:
        raise FileNotFoundError(f"No ETH3D official images found under {image_root}")
    return image_list


def _resolve_frame_files(dataset: str, seq_dir: Path, frame_id: int) -> Tuple[Path, Path, Path]:
    key = f"{frame_id:06d}"
    if dataset == "dtu":
        img_path = _resolve_first_file(
            [
                seq_dir / "images" / "00" / f"{key}.jpg",
                seq_dir / "images" / "00" / f"{key}.JPG",
                seq_dir / "images" / "00" / f"{key}.png",
                seq_dir / "images" / "00" / f"{key}.jpeg",
            ]
        )
        depth_path = _resolve_first_file(
            [
                seq_dir / "depths" / "00" / f"{key}.exr",
                seq_dir / "depths" / "00" / f"{key}.npy",
            ]
        )
        mask_path = _resolve_first_file(
            [
                seq_dir / "masks" / "00" / f"{key}.png",
                seq_dir / "masks" / "00" / f"{key}.jpg",
                seq_dir / "binary_masks" / f"{key}.png",
            ]
        )
    else:
        img_path = _resolve_first_file(
            [
                seq_dir / "images" / key / f"{key}.jpg",
                seq_dir / "images" / key / f"{key}.JPG",
                seq_dir / "images" / key / f"{key}.png",
                seq_dir / "images" / f"{key}.jpg",
                seq_dir / "images" / f"{key}.png",
            ]
        )
        if not img_path.is_file():
            img_path = _resolve_first_file(
                sorted((seq_dir / "images" / key).glob("*.jpg"))
                + sorted((seq_dir / "images" / key).glob("*.JPG"))
                + sorted((seq_dir / "images" / key).glob("*.png"))
                + sorted((seq_dir / "images" / key).glob("*.jpeg"))
            )
        depth_path = _resolve_first_file(
            [
                seq_dir / "depths" / key / f"{key}.exr",
                seq_dir / "depths" / key / f"{key}.npy",
                seq_dir / "depths" / f"{key}.exr",
                seq_dir / "depths" / f"{key}.npy",
            ]
        )
        if not depth_path.is_file():
            depth_path = _resolve_first_file(
                sorted((seq_dir / "depths" / key).glob("*.exr"))
                + sorted((seq_dir / "depths" / key).glob("*.npy"))
            )
        mask_path = _resolve_first_file(
            [
                seq_dir / "masks" / key / f"{key}.png",
                seq_dir / "masks" / key / f"{key}.jpg",
                seq_dir / "masks" / f"{key}.png",
            ]
        )
        if not mask_path.is_file():
            mask_path = _resolve_first_file(
                sorted((seq_dir / "masks" / key).glob("*.png"))
                + sorted((seq_dir / "masks" / key).glob("*.jpg"))
            )

    if not img_path.is_file():
        raise FileNotFoundError(f"Missing image for frame {frame_id} under {seq_dir}")
    if not depth_path.is_file():
        raise FileNotFoundError(f"Missing depth for frame {frame_id} under {seq_dir}")
    return img_path, depth_path, mask_path


def _is_7scenes_raw(seq_dir: Path) -> bool:
    for path in seq_dir.iterdir():
        if path.is_file() and re.match(r"^frame-(\d+)\.color\.(png|jpg|jpeg)$", path.name, flags=re.IGNORECASE):
            return True
    return False


def _list_7scenes_raw_frame_ids(seq_dir: Path) -> List[int]:
    ids: List[int] = []
    for path in seq_dir.iterdir():
        if not path.is_file():
            continue
        match = re.match(r"^frame-(\d+)\.color\.(png|jpg|jpeg)$", path.name, flags=re.IGNORECASE)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def _is_nrgbd_raw(seq_dir: Path) -> bool:
    return (seq_dir / "poses.txt").is_file() and (seq_dir / "images").is_dir() and (seq_dir / "depth").is_dir()


def _list_nrgbd_raw_frame_ids(seq_dir: Path) -> List[int]:
    ids: List[int] = []
    img_dir = seq_dir / "images"
    if not img_dir.is_dir():
        return ids
    for path in img_dir.iterdir():
        if not path.is_file():
            continue
        match = re.match(r"^img(\d+)\.(png|jpg|jpeg)$", path.name, flags=re.IGNORECASE)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def _load_nrgbd_raw_extrinsics_w2c(seq_dir: Path) -> np.ndarray:
    pose_file = seq_dir / "poses.txt"
    rows = [line.strip() for line in pose_file.read_text().splitlines() if line.strip()]
    if len(rows) % 4 != 0:
        raise ValueError(f"NRGBD poses.txt has {len(rows)} non-empty lines, not divisible by 4: {pose_file}")
    mats = []
    for index in range(0, len(rows), 4):
        mats.append([[float(x) for x in rows[index + offset].split()] for offset in range(4)])
    poses = np.asarray(mats, dtype=np.float32)
    poses[:, :, 1:3] *= -1.0
    extrinsics = closed_form_inverse_se3(poses)[:, :3, :]
    return extrinsics


def _load_depth_nrgbd_raw(depth_path: Path) -> np.ndarray:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed reading depth: {depth_path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0) / 1000.0
    depth[depth > 10.0] = 0.0
    depth[depth < 1e-3] = 0.0
    return depth


def _nrgbd_raw_intrinsic() -> np.ndarray:
    fx = fy = 554.2562584220408
    cx, cy = 320.0, 240.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _resolve_7scenes_raw_frame_files(seq_dir: Path, frame_id: int) -> Tuple[Path, Path, Path]:
    key = f"{frame_id:06d}"
    img_path = _resolve_first_file(
        [
            seq_dir / f"frame-{key}.color.png",
            seq_dir / f"frame-{key}.color.jpg",
            seq_dir / f"frame-{key}.color.jpeg",
        ]
    )
    depth_path = _resolve_first_file(
        [
            seq_dir / f"frame-{key}.depth.proj.png",
            seq_dir / f"frame-{key}.depth.png",
            seq_dir / f"frame-{key}.depth.exr",
            seq_dir / f"frame-{key}.depth.npy",
        ]
    )
    pose_path = _resolve_first_file([seq_dir / f"frame-{key}.pose.txt"])
    if not img_path.is_file():
        raise FileNotFoundError(f"Missing raw-7scenes image for frame {frame_id} under {seq_dir}")
    if not depth_path.is_file():
        raise FileNotFoundError(f"Missing raw-7scenes depth for frame {frame_id} under {seq_dir}")
    if not pose_path.is_file():
        raise FileNotFoundError(f"Missing raw-7scenes pose for frame {frame_id} under {seq_dir}")
    return img_path, depth_path, pose_path


def _load_7scenes_raw_intrinsic(seq_dir: Path) -> np.ndarray:
    del seq_dir
    fx = fy = 525.0
    cx, cy = 320.0, 240.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _load_7scenes_raw_pose_w2c(pose_path: Path) -> np.ndarray:
    values = [float(x) for x in pose_path.read_text().strip().replace(",", " ").split() if x]
    if len(values) != 16:
        raise ValueError(f"Unrecognized pose format in {pose_path} (expected 16 numbers, got {len(values)})")
    c2w = np.asarray(values, dtype=np.float32).reshape(4, 4)
    return np.linalg.inv(c2w)[:3, :4].astype(np.float32)


def _load_depth_7scenes_raw(depth_path: Path) -> np.ndarray:
    if depth_path.suffix.lower() == ".png":
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Failed reading depth: {depth_path}")
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.dtype == np.uint16:
            depth = depth.copy()
            depth[depth == 65535] = 0
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth > 10.0] = 0.0
        depth[depth < 1e-3] = 0.0
        return depth
    return _load_depth(depth_path)


def load_cam_mvsnet(words: str, interval_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    cam = np.zeros((2, 4, 4), dtype=np.float32)
    words = words.split()
    for i in range(4):
        for j in range(4):
            cam[0, i, j] = float(words[4 * i + j + 1])
    for i in range(3):
        for j in range(3):
            cam[1, i, j] = float(words[3 * i + j + 18])

    if len(words) == 29:
        cam[1, 3, 0] = float(words[27])
        cam[1, 3, 1] = float(words[28]) * interval_scale
        cam[1, 3, 2] = 192.0
        cam[1, 3, 3] = cam[1, 3, 0] + cam[1, 3, 1] * cam[1, 3, 2]
    elif len(words) == 30:
        cam[1, 3, 0] = float(words[27])
        cam[1, 3, 1] = float(words[28]) * interval_scale
        cam[1, 3, 2] = float(words[29])
        cam[1, 3, 3] = cam[1, 3, 0] + cam[1, 3, 1] * cam[1, 3, 2]
    elif len(words) == 31:
        cam[1, 3, 0] = float(words[27])
        cam[1, 3, 1] = float(words[28]) * interval_scale
        cam[1, 3, 2] = float(words[29])
        cam[1, 3, 3] = float(words[30])

    return cam[1].copy(), cam[0].copy()


def infer_dtu_data_format(seq_dir: Path, preferred: str) -> str:
    if preferred in {"evc", "raw"}:
        return preferred
    if (seq_dir / "cams").is_dir() and (seq_dir / "binary_masks").is_dir():
        return "raw"
    if (seq_dir / "cameras" / "00").is_dir():
        return "evc"
    raise FileNotFoundError(f"Cannot infer DTU data layout under {seq_dir}")


def _list_dtu_raw_image_ids(seq_dir: Path) -> List[int]:
    image_root = seq_dir / "images"
    ids = []
    for path in sorted(image_root.glob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path.stem.isdigit():
            ids.append(int(path.stem))
    return ids


def load_sample_dtu_raw(
    seq_dir: Path,
    ids: List[int],
    load_img_size: int,
    dtu_mask_erode: int,
):
    filelist: List[str] = []
    images: List[torch.Tensor] = []
    depths: List[np.ndarray] = []
    extrinsics = np.zeros((len(ids), 3, 4), dtype=np.float32)
    intrinsics = np.zeros((len(ids), 3, 3), dtype=np.float32)

    image_root = seq_dir / "images"
    depth_root = seq_dir / "depths"
    mask_root = seq_dir / "binary_masks"
    cam_root = seq_dir / "cams"

    for i, frame_id in enumerate(ids):
        stem = f"{frame_id:08d}"
        img_path = _resolve_first_file(
            [
                image_root / f"{stem}.jpg",
                image_root / f"{stem}.JPG",
                image_root / f"{stem}.png",
                image_root / f"{stem}.jpeg",
            ]
        )
        depth_path = depth_root / f"{stem}.npy"
        mask_path = mask_root / f"{stem}.png"
        cam_path = cam_root / f"{stem}_cam.txt"
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing raw DTU image: {img_path}")
        if not depth_path.is_file():
            raise FileNotFoundError(f"Missing raw DTU depth: {depth_path}")
        if not cam_path.is_file():
            raise FileNotFoundError(f"Missing raw DTU camera: {cam_path}")

        rgb = Image.open(img_path).convert("RGB")
        depth = np.load(str(depth_path))
        rgb = resize_image(rgb, (depth.shape[1], depth.shape[0]))
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        mask = _load_mask(mask_path, depth.shape, dtu_mask_erode=dtu_mask_erode)
        depth = depth * mask

        cur_intrinsics, extrinsic = load_cam_mvsnet(cam_path.read_text())
        intrinsic = cur_intrinsics[:3, :3]
        rgb, depth, intrinsic = resize_image_depth_and_intrinsic(
            image=rgb,
            depth_map=depth,
            intrinsic=intrinsic,
            output_width=load_img_size,
        )

        filelist.append(str(img_path))
        images.append(TO_TENSOR(rgb))
        depths.append(depth)
        intrinsics[i] = intrinsic
        extrinsics[i] = extrinsic[:3, :]

    images_tensor = torch.stack(images, dim=0)
    depths_np = np.stack(depths, axis=0)
    valid_mask = depths_np > 1e-4
    gt_pts = unproject_depth_map_to_point_map(
        depth_map=depths_np[..., None],
        extrinsics_cam=extrinsics,
        intrinsics_cam=intrinsics,
        invert_extrinsics=True,
    )
    return filelist, images_tensor, gt_pts, valid_mask


def load_sample(
    dataset: str,
    seq_dir: Path,
    cam_dict,
    ids: List[int],
    load_img_size: int,
    max_depth: float,
    min_depth: float,
    dtu_mask_erode: int,
    dtu_unit_scale: float,
    dtu_center_crop_height: int,
    dtu_data_format: str,
    eth3d_extri_c2w: bool,
):
    if dataset == "eth3d" and (seq_dir / "custom_undistorted_cam").is_dir():
        return load_sample_eth3d_official(
            seq_dir=seq_dir,
            ids=ids,
            load_img_size=load_img_size,
        )
    if dataset == "dtu" and dtu_data_format == "raw":
        return load_sample_dtu_raw(
            seq_dir=seq_dir,
            ids=ids,
            load_img_size=load_img_size,
            dtu_mask_erode=dtu_mask_erode,
        )
    if dataset == "7scenes" and _is_7scenes_raw(seq_dir):
        filelist: List[str] = []
        images: List[torch.Tensor] = []
        depths: List[np.ndarray] = []
        extrinsics = np.zeros((len(ids), 3, 4), dtype=np.float32)
        intrinsics = np.zeros((len(ids), 3, 3), dtype=np.float32)

        intrinsic0 = _load_7scenes_raw_intrinsic(seq_dir)
        for index, frame_id in enumerate(ids):
            img_path, depth_path, pose_path = _resolve_7scenes_raw_frame_files(seq_dir, frame_id)
            rgb = Image.open(img_path).convert("RGB")
            depth = _load_depth_7scenes_raw(depth_path)
            depth[(depth > max_depth) | (depth < min_depth)] = 0.0

            extrinsic = _load_7scenes_raw_pose_w2c(pose_path)
            intrinsic = intrinsic0.copy()

            rgb = resize_image(rgb, (depth.shape[1], depth.shape[0]))
            rgb, depth, intrinsic = resize_image_depth_and_intrinsic(
                image=rgb,
                depth_map=depth,
                intrinsic=intrinsic,
                output_width=load_img_size,
            )

            filelist.append(str(img_path))
            images.append(TO_TENSOR(rgb))
            depths.append(depth)
            intrinsics[index] = intrinsic
            extrinsics[index] = extrinsic

        images_tensor = torch.stack(images, dim=0)
        depths_np = np.stack(depths, axis=0)
        valid_mask = depths_np > 1e-4
        gt_pts = unproject_depth_map_to_point_map(
            depth_map=depths_np[..., None],
            extrinsics_cam=extrinsics,
            intrinsics_cam=intrinsics,
            invert_extrinsics=True,
        )
        return filelist, images_tensor, gt_pts, valid_mask
    if dataset == "nrgbd" and _is_nrgbd_raw(seq_dir):
        filelist: List[str] = []
        images: List[torch.Tensor] = []
        depths: List[np.ndarray] = []
        extrinsics_all = _load_nrgbd_raw_extrinsics_w2c(seq_dir)
        intrinsic0 = _nrgbd_raw_intrinsic()

        extrinsics = np.zeros((len(ids), 3, 4), dtype=np.float32)
        intrinsics = np.zeros((len(ids), 3, 3), dtype=np.float32)

        for index, frame_id in enumerate(ids):
            img_path = seq_dir / "images" / f"img{frame_id}.png"
            if not img_path.is_file():
                img_path = _resolve_first_file(
                    [
                        seq_dir / "images" / f"img{frame_id}.jpg",
                        seq_dir / "images" / f"img{frame_id}.jpeg",
                        seq_dir / "images" / f"img{frame_id}.png",
                    ]
                )
            depth_path = seq_dir / "depth" / f"depth{frame_id}.png"
            if not depth_path.is_file():
                raise FileNotFoundError(f"Missing NRGBD depth for frame {frame_id} under {seq_dir}")
            if not img_path.is_file():
                raise FileNotFoundError(f"Missing NRGBD image for frame {frame_id} under {seq_dir}")
            if frame_id >= extrinsics_all.shape[0]:
                raise IndexError(
                    f"NRGBD pose index {frame_id} out of range for {seq_dir} (n={extrinsics_all.shape[0]})"
                )

            rgb = Image.open(img_path).convert("RGB")
            depth = _load_depth_nrgbd_raw(depth_path)
            extrinsic = extrinsics_all[frame_id].astype(np.float32)
            intrinsic = intrinsic0.copy()

            rgb = resize_image(rgb, (depth.shape[1], depth.shape[0]))
            rgb, depth, intrinsic = resize_image_depth_and_intrinsic(
                image=rgb,
                depth_map=depth,
                intrinsic=intrinsic,
                output_width=load_img_size,
            )

            filelist.append(str(img_path))
            images.append(TO_TENSOR(rgb))
            depths.append(depth)
            intrinsics[index] = intrinsic
            extrinsics[index] = extrinsic

        images_tensor = torch.stack(images, dim=0)
        depths_np = np.stack(depths, axis=0)
        valid_mask = depths_np > 1e-4
        gt_pts = unproject_depth_map_to_point_map(
            depth_map=depths_np[..., None],
            extrinsics_cam=extrinsics,
            intrinsics_cam=intrinsics,
            invert_extrinsics=True,
        )
        return filelist, images_tensor, gt_pts, valid_mask

    filelist: List[str] = []
    images: List[torch.Tensor] = []
    depths: List[np.ndarray] = []
    extrinsics = np.zeros((len(ids), 3, 4), dtype=np.float32)
    intrinsics = np.zeros((len(ids), 3, 3), dtype=np.float32)

    for i, frame_id in enumerate(ids):
        key = f"{frame_id:06d}"
        if key not in cam_dict:
            raise KeyError(f"Missing camera key {key} in {seq_dir}")
        img_path, depth_path, mask_path = _resolve_frame_files(dataset, seq_dir, frame_id)

        rgb = Image.open(img_path).convert("RGB")
        depth = _load_depth(depth_path)
        if dataset != "eth3d":
            depth[(depth > max_depth) | (depth < min_depth)] = 0.0

        if dataset == "dtu":
            mask = _load_mask(mask_path, depth.shape, dtu_mask_erode=dtu_mask_erode)
            depth = depth * mask

        cam = cam_dict[key]
        K = np.asarray(cam.K, dtype=np.float32).copy()
        RT = np.asarray(cam.RT, dtype=np.float32).copy()

        if dataset == "dtu" and dtu_center_crop_height > 0:
            rgb, depth, mask, K = _center_crop_dtu(
                rgb=rgb,
                depth=depth,
                mask=mask,
                intrinsic=K,
                crop_h=dtu_center_crop_height,
            )

        rgb = resize_image(rgb, (depth.shape[1], depth.shape[0]))
        rgb, depth, K = resize_image_depth_and_intrinsic(
            image=rgb,
            depth_map=depth,
            intrinsic=K,
            output_width=load_img_size,
        )

        if dataset == "dtu" and dtu_unit_scale != 1.0:
            depth = depth * float(dtu_unit_scale)
            RT[:, 3] = RT[:, 3] * float(dtu_unit_scale)

        filelist.append(str(img_path))
        images.append(TO_TENSOR(rgb))
        depths.append(depth)
        intrinsics[i] = K
        extrinsics[i] = RT

    images_tensor = torch.stack(images, dim=0)  # (N,3,H,W)
    depths_np = np.stack(depths, axis=0)  # (N,H,W)
    valid_mask = depths_np > 1e-4
    gt_pts = unproject_depth_map_to_point_map(
        depth_map=depths_np[..., None],
        extrinsics_cam=extrinsics,
        intrinsics_cam=intrinsics,
        invert_extrinsics=not (dataset == "eth3d" and eth3d_extri_c2w),
    )
    return filelist, images_tensor, gt_pts, valid_mask


def load_sample_eth3d_official(
    seq_dir: Path,
    ids: List[int],
    load_img_size: int,
):
    image_list = _list_eth3d_official_images(seq_dir)
    filelist: List[str] = []
    images: List[torch.Tensor] = []
    depths: List[np.ndarray] = []
    extrinsics = np.zeros((len(ids), 3, 4), dtype=np.float32)
    intrinsics = np.zeros((len(ids), 3, 3), dtype=np.float32)

    for i, idx in enumerate(ids):
        if idx >= len(image_list):
            raise IndexError(f"ETH3D image index {idx} out of range for {seq_dir} (n={len(image_list)})")
        img_path = image_list[idx]
        img_name = img_path.name
        depth_path = seq_dir / "ground_truth_depth" / "custom_undistorted" / img_name
        cam_path = seq_dir / "custom_undistorted_cam" / img_name.replace(".JPG", ".npz").replace(".jpg", ".npz")
        if not depth_path.is_file():
            raise FileNotFoundError(f"Missing ETH3D official depth: {depth_path}")
        if not cam_path.is_file():
            raise FileNotFoundError(f"Missing ETH3D official camera: {cam_path}")

        rgb = Image.open(img_path).convert("RGB")
        width, height = rgb.size
        depth = _load_eth3d_official_depth(depth_path=depth_path, target_hw=(height, width))

        cam = np.load(str(cam_path))
        intrinsic = cam["intrinsics"].astype(np.float32)
        extrinsic = cam["extrinsics"][:3, :].astype(np.float32)

        rgb, depth, intrinsic = resize_image_depth_and_intrinsic(
            image=rgb,
            depth_map=depth,
            intrinsic=intrinsic,
            output_width=load_img_size,
        )

        filelist.append(str(img_path))
        images.append(TO_TENSOR(rgb))
        depths.append(depth)
        intrinsics[i] = intrinsic
        extrinsics[i] = extrinsic

    images_tensor = torch.stack(images, dim=0)
    depths_np = np.stack(depths, axis=0)
    valid_mask = depths_np > 1e-4
    gt_pts = unproject_depth_map_to_point_map(
        depth_map=depths_np[..., None],
        extrinsics_cam=extrinsics,
        intrinsics_cam=intrinsics,
        invert_extrinsics=True,
    )
    return filelist, images_tensor, gt_pts, valid_mask


def write_csv_row(path: Path, row: Dict[str, float]) -> None:
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def eval_protocol(
    dataset: str,
    protocol: str,
    seq_map_path: Path,
    seq_dirs: Dict[str, Path],
    model,
    infer_cfg: SimpleNamespace,
    load_img_size: int,
    max_depth: float,
    min_depth: float,
    dtu_mask_erode: int,
    dtu_unit_scale: float,
    dtu_center_crop_height: int,
    dtu_data_format: str,
    eth3d_extri_c2w: bool,
    output_dir: Path,
    infer_mv_pointclouds,
    umeyama,
    accuracy,
    completion,
    icp_threshold: float,
    eval_frame_indices: List[int] | None = None,
    max_sequences: int = 0,
    verbose: bool = False,
):
    with seq_map_path.open("r") as f:
        seq_map = json.load(f)

    protocol_dir = output_dir / protocol
    protocol_dir.mkdir(parents=True, exist_ok=True)
    per_seq_csv = protocol_dir / "_all_samples.csv"
    if per_seq_csv.exists():
        per_seq_csv.unlink()

    agg = {
        "Acc-mean": 0.0,
        "Acc-med": 0.0,
        "Comp-mean": 0.0,
        "Comp-med": 0.0,
        "NC-mean": 0.0,
        "NC-med": 0.0,
        "NC1-mean": 0.0,
        "NC1-med": 0.0,
        "NC2-mean": 0.0,
        "NC2-med": 0.0,
    }

    used_seqs = 0
    skipped: List[Tuple[str, str]] = []
    cam_cache: Dict[str, Tuple[Dict, List[str]]] = {}

    for seq_idx, (seq_key, raw_entry) in enumerate(seq_map.items(), start=1):
        if max_sequences > 0 and seq_idx > max_sequences:
            break
        try:
            seq_name, raw_ids = normalize_seq_map_entry(dataset, seq_key, raw_entry)
        except Exception as err:
            skipped.append((seq_key, f"bad_seq_map_entry:{type(err).__name__}:{err}"))
            continue
        if seq_name not in seq_dirs:
            skipped.append((seq_key, "missing_sequence"))
            continue

        seq_dir = seq_dirs[seq_name]
        cur_dtu_format = dtu_data_format
        if dataset == "dtu":
            cur_dtu_format = infer_dtu_data_format(seq_dir, dtu_data_format)
        is_raw_7scenes = dataset == "7scenes" and _is_7scenes_raw(seq_dir)
        is_raw_nrgbd = dataset == "nrgbd" and _is_nrgbd_raw(seq_dir)
        if not is_raw_7scenes and not is_raw_nrgbd:
            if seq_name not in cam_cache:
                cam_cache[seq_name] = load_sequence_cameras(dataset, seq_dir)
            cam_dict, cam_names = cam_cache[seq_name]
        else:
            cam_dict, cam_names = None, None
        if dataset == "dtu" and cur_dtu_format == "raw":
            available_ids = set(_list_dtu_raw_image_ids(seq_dir))
            ids = [int(i) for i in raw_ids if int(i) in available_ids]
            max_frames = len(available_ids)
        elif dataset == "eth3d" and (seq_dir / "custom_undistorted_cam").is_dir():
            max_frames = len(_list_eth3d_official_images(seq_dir))
            ids = [int(i) for i in raw_ids if int(i) < max_frames]
        elif is_raw_7scenes:
            available = _list_7scenes_raw_frame_ids(seq_dir)
            available_ids = set(available)
            ids = [int(i) for i in raw_ids if int(i) in available_ids]
            max_frames = len(available)
        elif is_raw_nrgbd:
            available = _list_nrgbd_raw_frame_ids(seq_dir)
            available_ids = set(available)
            ids = [int(i) for i in raw_ids if int(i) in available_ids]
            max_frames = len(available)
        else:
            max_frames = len(cam_names)
            ids = [int(i) for i in raw_ids if int(i) < max_frames]
        if len(ids) == 0:
            skipped.append((seq_key, "empty_after_range_filter"))
            continue

        t0 = time.time()
        try:
            filelist, images, gt_pts, valid_mask = load_sample(
                dataset=dataset,
                seq_dir=seq_dir,
                cam_dict=cam_dict,
                ids=ids,
                load_img_size=load_img_size,
                max_depth=max_depth,
                min_depth=min_depth,
                dtu_mask_erode=dtu_mask_erode,
                dtu_unit_scale=dtu_unit_scale,
                dtu_center_crop_height=dtu_center_crop_height,
                dtu_data_format=cur_dtu_format,
                eth3d_extri_c2w=eth3d_extri_c2w,
            )
        except Exception as err:
            skipped.append((seq_key, f"load_error:{type(err).__name__}:{err}"))
            continue

        data_h, data_w = images.shape[-2:]
        try:
            pred_pts = infer_mv_pointclouds(
                filelist=filelist,
                model=model,
                load_img_size=infer_cfg.load_img_size,
                device=infer_cfg.device,
                verbose=infer_cfg.verbose,
                data_size=(data_h, data_w),
                point_source=infer_cfg.point_source,
                vggt_input_style=infer_cfg.vggt_input_style,
            )
        except Exception as err:
            skipped.append((seq_key, f"infer_error:{type(err).__name__}:{err}"))
            continue

        if pred_pts.shape != gt_pts.shape:
            skipped.append((seq_key, "shape_mismatch"))
            continue
        try:
            pred_pts, gt_pts, valid_mask, images = subset_arrays_for_metric(
                pred_pts=pred_pts,
                gt_pts=gt_pts,
                valid_mask=valid_mask,
                images=images,
                eval_indices=list(eval_frame_indices or []),
            )
        except Exception as err:
            skipped.append((seq_key, f"eval_subset_error:{type(err).__name__}:{err}"))
            continue
        if not np.any(valid_mask):
            skipped.append((seq_key, "empty_valid_mask"))
            continue

        colors = images.permute(0, 2, 3, 1)[valid_mask].cpu().numpy().reshape(-1, 3)

        c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
        pred_pts = c * np.einsum("nhwj,ij->nhwi", pred_pts, R) + t.T
        pred_pts = pred_pts[valid_mask].reshape(-1, 3)
        gt_pts = gt_pts[valid_mask].reshape(-1, 3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pred_pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
        pcd_gt.colors = o3d.utility.Vector3dVector(colors)

        reg = o3d.pipelines.registration.registration_icp(
            pcd,
            pcd_gt,
            icp_threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        pcd = pcd.transform(reg.transformation)
        pcd.estimate_normals()
        pcd_gt.estimate_normals()

        pred_normal = np.asarray(pcd.normals)
        gt_normal = np.asarray(pcd_gt.normals)
        acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd.points, gt_normal, pred_normal)
        comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points, gt_normal, pred_normal)
        nc_mean = (nc1 + nc2) / 2.0
        nc_med = (nc1_med + nc2_med) / 2.0

        used_seqs += 1
        agg["Acc-mean"] += acc
        agg["Acc-med"] += acc_med
        agg["Comp-mean"] += comp
        agg["Comp-med"] += comp_med
        agg["NC-mean"] += nc_mean
        agg["NC-med"] += nc_med
        agg["NC1-mean"] += nc1
        agg["NC1-med"] += nc1_med
        agg["NC2-mean"] += nc2
        agg["NC2-med"] += nc2_med

        write_csv_row(
            per_seq_csv,
            {
                "seq": seq_key,
                "num_input_views": len(ids),
                "num_metric_views": int(len(eval_frame_indices or []) or len(ids)),
                "Acc-mean": acc,
                "Acc-med": acc_med,
                "Comp-mean": comp,
                "Comp-med": comp_med,
                "NC-mean": nc_mean,
                "NC-med": nc_med,
                "NC1-mean": nc1,
                "NC1-med": nc1_med,
                "NC2-mean": nc2,
                "NC2-med": nc2_med,
                "elapsed_sec": time.time() - t0,
            },
        )
        if verbose:
            print(
                f"[{protocol}] {seq_key}: Acc={acc:.4f} Comp={comp:.4f} NC={nc_mean:.4f} (views={len(ids)})",
                flush=True,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if used_seqs == 0:
        metric = {k: float("nan") for k in agg}
    else:
        metric = {k: float(v / used_seqs) for k, v in agg.items()}

    metric["num_sequences"] = int(used_seqs)
    metric["num_skipped"] = int(len(skipped))

    metric_csv = output_dir / f"{protocol}-metric.csv"
    with metric_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric.keys()))
        writer.writeheader()
        writer.writerow(metric)
    return metric, skipped, int(len(seq_map))


def main() -> None:
    args = parse_args()
    eval_frame_indices = parse_eval_frame_indices(args.eval_frame_indices)
    dataset_root = Path(args.dataset_root.strip() or default_dataset_root(args.dataset)).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    pi3_root = Path(args.pi3_root).resolve()
    if args.model_family == "pi3" and not args.ckpt:
        args.ckpt = default_pi3_ckpt()

    protocols = resolve_protocols(args.dataset, args.protocol)
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        if args.model_family == "vggt":
            model_suffix = f"vggt_{args.vggt_model_tag}"
        elif args.model_family == "da3":
            model_suffix = "da3"
        elif args.model_family == "vggt_omega":
            model_suffix = "vggt_omega"
        else:
            model_suffix = "pi3"
        output_dir = (Path("tmp") / f"pi3_mv_recon_{args.dataset}_{model_suffix}_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    umeyama, accuracy, completion = import_local_pi3_metric_utils()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)
    if args.model_family == "pi3":
        from aidi.scripts.baselines.pi3_checkpoint_loader import is_native_pi3_requested

        Pi3 = None if is_native_pi3_requested(args.ckpt) else import_pi3_model(pi3_root)
        model, loaded_ckpt = load_pi3_model(Pi3, args.ckpt, device)
        infer_fn = infer_pi3_mv_pointclouds
    elif args.model_family == "vggt":
        model, loaded_ckpt = load_vggt_model(args, device)
        infer_fn = infer_vggt_mv_pointclouds
    elif args.model_family == "vggt_omega":
        os.environ["VGGT_OMEGA_REPO"] = str(Path(args.vggt_omega_repo).expanduser().resolve())
        os.environ["VGGT_OMEGA_RESOLUTION"] = str(int(args.vggt_omega_resolution))
        os.environ["VGGT_OMEGA_MODE"] = str(args.vggt_omega_mode)
        model, loaded_ckpt = load_vggt_omega_model(args, device)
        infer_fn = infer_vggt_omega_mv_pointclouds
    else:
        model, loaded_ckpt = load_da3_model(args, device)
        infer_fn = infer_da3_mv_pointclouds

    infer_cfg = SimpleNamespace(
        load_img_size=args.load_img_size,
        device=args.device,
        verbose=args.verbose,
        point_source=args.point_source,
        vggt_input_style=args.vggt_input_style,
    )

    seq_dirs = discover_sequences(dataset_root, dataset=args.dataset)
    if not seq_dirs:
        raise RuntimeError(f"No scene folders found under {dataset_root}")

    summary = {
        "dataset": args.dataset,
        "model_family": args.model_family,
        "vggt_model_tag": args.vggt_model_tag if args.model_family == "vggt" else "",
        "dataset_root": str(dataset_root),
        "pi3_root": str(pi3_root),
        "ckpt": loaded_ckpt,
        "da3_repo": str(Path(args.da3_repo).resolve()) if args.model_family == "da3" else "",
        "da3_process_res": int(args.da3_process_res) if args.model_family == "da3" else 0,
        "da3_process_res_method": str(args.da3_process_res_method) if args.model_family == "da3" else "",
        "vggt_omega_repo": str(Path(args.vggt_omega_repo).expanduser().resolve()) if args.model_family == "vggt_omega" else "",
        "vggt_omega_resolution": int(args.vggt_omega_resolution) if args.model_family == "vggt_omega" else 0,
        "vggt_omega_mode": str(args.vggt_omega_mode) if args.model_family == "vggt_omega" else "",
        "device": args.device,
        "point_source": args.point_source,
        "vggt_input_style": args.vggt_input_style,
        "load_img_size": args.load_img_size,
        "dtu_unit_scale": args.dtu_unit_scale,
        "dtu_center_crop_height": args.dtu_center_crop_height,
        "dtu_data_format": args.dtu_data_format,
        "eval_frame_indices": [int(item) for item in eval_frame_indices],
        "protocols": {},
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for protocol in protocols:
        seq_map = resolve_seq_map(pi3_root, args.dataset, protocol, args.seq_map)
        if not seq_map.is_file():
            raise FileNotFoundError(f"seq-id-map not found: {seq_map}")

        icp_threshold = args.icp_threshold_dtu if args.dataset == "dtu" else args.icp_threshold
        metric, skipped, seq_total = eval_protocol(
            dataset=args.dataset,
            protocol=protocol,
            seq_map_path=seq_map,
            seq_dirs=seq_dirs,
            model=model,
            infer_cfg=infer_cfg,
            load_img_size=args.load_img_size,
            max_depth=args.max_depth,
            min_depth=args.min_depth,
            dtu_mask_erode=args.dtu_mask_erode,
            dtu_unit_scale=args.dtu_unit_scale,
            dtu_center_crop_height=args.dtu_center_crop_height,
            dtu_data_format=args.dtu_data_format,
            eth3d_extri_c2w=args.eth3d_extri_c2w,
            output_dir=output_dir,
            infer_mv_pointclouds=infer_fn,
            umeyama=umeyama,
            accuracy=accuracy,
            completion=completion,
            icp_threshold=icp_threshold,
            eval_frame_indices=eval_frame_indices,
            max_sequences=args.max_sequences,
            verbose=args.verbose,
        )
        summary["protocols"][protocol] = {
            "seq_map": str(seq_map),
            "metrics": metric,
            "num_seq_in_map": seq_total,
            "num_seq_evaluated_cap": int(args.max_sequences),
            "icp_threshold": icp_threshold,
            "skipped": [{"seq": s, "reason": r} for s, r in skipped],
        }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[mv-recon-core] done. summary: {summary_path}", flush=True)
    empty_protocols = []
    for protocol in protocols:
        m = summary["protocols"][protocol]["metrics"]
        print(
            f"[{protocol}] Acc-mean={m.get('Acc-mean'):.6f}, Comp-mean={m.get('Comp-mean'):.6f}, "
            f"NC-mean={m.get('NC-mean'):.6f}, n_seq={int(m.get('num_sequences', 0))}",
            flush=True,
        )
        if int(m.get("num_sequences", 0)) <= 0:
            skipped = summary["protocols"][protocol].get("skipped", [])
            empty_protocols.append(f"{protocol}: skipped={len(skipped)}")
    if empty_protocols:
        raise RuntimeError(
            "No valid sequences were evaluated for pointcloud protocol(s): "
            + "; ".join(empty_protocols)
            + f". Summary was written to {summary_path}"
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
