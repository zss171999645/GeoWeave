#!/usr/bin/env python3
"""Evaluate official/finetuned VGGT on the DA3 pose benchmark protocol."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_repo_root_on_syspath() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


ensure_repo_root_on_syspath()


DEFAULT_BENCHMARK_ROOT = repo_root() / "workspace" / "benchmark_dataset"
DEFAULT_DATA_ROOTS: Dict[str, Path] = {
    "eth3d": DEFAULT_BENCHMARK_ROOT / "eth3d",
    "hiroom": DEFAULT_BENCHMARK_ROOT / "hiroom" / "data",
    "scannetpp": DEFAULT_BENCHMARK_ROOT / "scannetpp",
    "dtu64": DEFAULT_BENCHMARK_ROOT / "dtu64",
    "7scenes": DEFAULT_BENCHMARK_ROOT / "7scenes",
}

DA3_ETH3D_SCENES: Sequence[str] = (
    "courtyard",
    "electro",
    "kicker",
    "pipes",
    "relief",
    "delivery_area",
    "facade",
    "office",
    "playground",
    "relief_2",
    "terrains",
)

DA3_ETH3D_FILTER_KEYS: Mapping[str, Sequence[str]] = {
    "delivery_area": ("711.JPG", "712.JPG", "713.JPG", "714.JPG"),
    "electro": ("9289.JPG", "9290.JPG", "9291.JPG", "9292.JPG", "9293.JPG", "9298.JPG"),
    "playground": ("587.JPG", "588.JPG", "589.JPG", "590.JPG", "591.JPG", "592.JPG"),
    "relief": (
        "427.JPG",
        "428.JPG",
        "429.JPG",
        "430.JPG",
        "431.JPG",
        "432.JPG",
        "433.JPG",
        "434.JPG",
        "435.JPG",
        "436.JPG",
        "437.JPG",
        "438.JPG",
    ),
    "relief_2": (
        "458.JPG",
        "459.JPG",
        "460.JPG",
        "461.JPG",
        "462.JPG",
        "463.JPG",
        "464.JPG",
        "465.JPG",
        "466.JPG",
        "467.JPG",
        "468.JPG",
    ),
}

DA3_SCANNETPP_SCENES: Sequence[str] = (
    "09c1414f1b",
    "1ada7a0617",
    "40aec5fffa",
    "3e8bba0176",
    "acd95847c5",
    "578511c8a9",
    "5f99900f09",
    "c4c04e6d6c",
    "f3d64c30f8",
    "7bc286c1b6",
    "c5439f4607",
    "286b55a2bf",
    "fb5a96b1a2",
    "7831862f02",
    "38d58a7a31",
    "bde1e479ad",
    "9071e139d9",
    "21d970d8de",
    "bcd2436daf",
    "cc5237fd77",
)

DA3_DTU64_SCENES: Sequence[str] = (
    "scan105",
    "scan114",
    "scan118",
    "scan122",
    "scan24",
    "scan37",
    "scan40",
    "scan55",
    "scan63",
    "scan65",
    "scan69",
    "scan83",
    "scan97",
)

DA3_7SCENES_SCENES: Sequence[str] = (
    "chess",
    "fire",
    "heads",
    "office",
    "pumpkin",
    "redkitchen",
    "stairs",
)

DEFAULT_VGGT_OFFICIAL_CKPT_ROOT = (
    "/horizon-bucket/saturn_v_dev/01_users/tao02.xie/projects/GFM/gfm/data/trained_model/vggt/official/VGGT-1B"
)
DEFAULT_PI3_LOAD_IMG_SIZE = 512
DEFAULT_IMAGE_LOAD_RETRIES = 8
DEFAULT_IMAGE_LOAD_RETRY_SLEEP = 0.5


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="hiroom,scannetpp,dtu64")
    parser.add_argument("--scenes", default="")
    parser.add_argument("--benchmark-root", default=str(DEFAULT_BENCHMARK_ROOT))
    parser.add_argument("--eth3d-root", default="")
    parser.add_argument("--hiroom-root", default="")
    parser.add_argument("--scannetpp-root", default="")
    parser.add_argument("--dtu64-root", default="")
    parser.add_argument("--7scenes-root", "--seven-scenes-root", dest="seven_scenes_root", default="")
    parser.add_argument("--dtu64-camera-root", default="")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--image-preprocess-style",
        choices=("da3_bench", "official_vggt"),
        default="da3_bench",
    )
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument(
        "--process-res-method",
        choices=("upper_bound_resize", "upper_bound_crop", "lower_bound_resize", "lower_bound_crop"),
        default="upper_bound_resize",
    )

    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-family", choices=("vggt", "pi3", "vggt_omega"), default="vggt")
    parser.add_argument("--pi3-config", default="", help="Native sparse Pi3 Hydra config path.")
    parser.add_argument(
        "--pi3-model-impl",
        default="",
        help="Pi3 implementation override: official or native_sparse. Empty auto-detects from config/checkpoint.",
    )
    parser.add_argument(
        "--pi3-native-root",
        default="aidi/third_party/pi3_training",
        help="Native sparse Pi3 source root used when --pi3-model-impl=native_sparse.",
    )
    parser.add_argument("--load-img-size", type=int, default=DEFAULT_PI3_LOAD_IMG_SIZE)
    parser.add_argument("--image-load-retries", type=int, default=DEFAULT_IMAGE_LOAD_RETRIES)
    parser.add_argument("--image-load-retry-sleep", type=float, default=DEFAULT_IMAGE_LOAD_RETRY_SLEEP)
    parser.add_argument("--vggt-model-tag", choices=("official", "finetuned"), default="official")
    parser.add_argument("--vggt-config", default="")
    parser.add_argument("--vggt-official-ckpt-root", default=DEFAULT_VGGT_OFFICIAL_CKPT_ROOT)
    parser.add_argument("--vggt-topk-override", type=int, default=0)
    parser.add_argument("--vggt-topk-block-override", type=int, default=None)
    parser.add_argument("--vggt-topk-merge-blocks-override", type=int, default=None)
    parser.add_argument("--vggt-source-downsample-enabled", action="store_true")
    parser.add_argument("--vggt-source-downsample-factor-override", type=int, default=None)
    parser.add_argument("--vggt-source-downsample-strategy-override", default="")
    parser.add_argument("--vggt-source-downsample-query-chunk-override", type=int, default=None)
    parser.add_argument("--vggt-source-downsample-coarse-topk-override", type=int, default=None)
    parser.add_argument("--vggt-source-downsample-coarse-ratio-override", type=float, default=None)
    parser.add_argument("--vggt-depth-frames-chunk-size-override", type=int, default=None)
    parser.add_argument("--vggt-point-frames-chunk-size-override", type=int, default=None)
    parser.add_argument("--vggt-keep-component-ckpts", action="store_true")
    parser.add_argument("--vggt-omega-repo", default="")
    parser.add_argument("--vggt-omega-checkpoint", default="")
    parser.add_argument("--vggt-omega-resolution", type=int, default=512)
    parser.add_argument("--vggt-omega-mode", choices=("balanced", "max_size"), default="balanced")
    parser.add_argument(
        "--vggt-omega-global-attention-mode",
        choices=("original", "camera_register_query"),
        default="original",
        help=(
            "Experimental VGGT-Omega inter-frame global attention mode. "
            "camera_register_query updates only camera/register queries while image tokens remain key/value context."
        ),
    )
    return parser.parse_args()


def parse_csv(value: str) -> List[str]:
    text = (value or "").strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def resolve_data_roots(args: argparse.Namespace) -> Dict[str, Path]:
    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    roots = {
        "eth3d": Path(args.eth3d_root).expanduser().resolve() if args.eth3d_root else benchmark_root / "eth3d",
        "hiroom": Path(args.hiroom_root).expanduser().resolve() if args.hiroom_root else benchmark_root / "hiroom" / "data",
        "scannetpp": Path(args.scannetpp_root).expanduser().resolve() if args.scannetpp_root else benchmark_root / "scannetpp",
        "dtu64": Path(args.dtu64_root).expanduser().resolve() if args.dtu64_root else benchmark_root / "dtu64",
        "7scenes": Path(args.seven_scenes_root).expanduser().resolve() if args.seven_scenes_root else benchmark_root / "7scenes",
    }
    return roots


def _nearest_multiple(value: int, patch_size: int) -> int:
    down = (value // patch_size) * patch_size
    up = down + patch_size
    return up if abs(up - value) <= abs(value - down) else down


def _resize_pil_image(img, target_size: int, method: str):
    import cv2
    from PIL import Image

    width, height = img.size
    if method in ("upper_bound_resize", "upper_bound_crop"):
        source_size = max(width, height)
    elif method in ("lower_bound_resize", "lower_bound_crop"):
        source_size = min(width, height)
    else:
        raise ValueError(f"Unsupported resize method: {method}")

    if source_size == target_size:
        return img

    scale = target_size / float(source_size)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(np.asarray(img), (new_width, new_height), interpolation=interpolation)
    return Image.fromarray(resized)


def _make_pil_divisible(img, patch_size: int, method: str):
    import cv2
    from PIL import Image

    width, height = img.size
    if method.endswith("resize"):
        new_width = max(1, _nearest_multiple(width, patch_size))
        new_height = max(1, _nearest_multiple(height, patch_size))
        if new_width == width and new_height == height:
            return img
        upscale = new_width > width or new_height > height
        interpolation = cv2.INTER_CUBIC if upscale else cv2.INTER_AREA
        resized = cv2.resize(np.asarray(img), (new_width, new_height), interpolation=interpolation)
        return Image.fromarray(resized)

    if method.endswith("crop"):
        new_width = max(patch_size, (width // patch_size) * patch_size)
        new_height = max(patch_size, (height // patch_size) * patch_size)
        if new_width == width and new_height == height:
            return img
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        return img.crop((left, top, left + new_width, top + new_height))

    raise ValueError(f"Unsupported resize method: {method}")


def preprocess_images_for_vggt(
    image_files: Sequence[str],
    image_preprocess_style: str = "da3_bench",
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
):
    if image_preprocess_style == "official_vggt":
        from easyvolcap.official_vggt.utils.load_fn import load_and_preprocess_images

        return load_and_preprocess_images([str(path) for path in image_files])

    if image_preprocess_style != "da3_bench":
        raise ValueError(f"Unsupported image preprocess style: {image_preprocess_style}")

    import torch
    from PIL import Image
    from torchvision import transforms as TF

    normalize = TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = TF.ToTensor()
    images = []
    sizes = []
    for image_path in image_files:
        img = Image.open(image_path).convert("RGB")
        img = _resize_pil_image(img, target_size=process_res, method=process_res_method)
        img = _make_pil_divisible(img, patch_size=14, method=process_res_method)
        tensor = normalize(to_tensor(img))
        images.append(tensor)
        sizes.append((tensor.shape[1], tensor.shape[2]))

    if len(set(sizes)) > 1:
        min_height = min(height for height, _ in sizes)
        min_width = min(width for _, width in sizes)
        cropped = []
        for tensor, (height, width) in zip(images, sizes):
            top = max(0, (height - min_height) // 2)
            left = max(0, (width - min_width) // 2)
            cropped.append(tensor[:, top : top + min_height, left : left + min_width])
        images = cropped

    return torch.stack(images)


def sample_frame_indices(num_frames: int, max_frames: int, seed: int = 42) -> List[int]:
    if max_frames <= 0 or num_frames <= max_frames:
        return list(range(num_frames))
    rng = random.Random(seed)
    indices = list(range(num_frames))
    rng.shuffle(indices)
    return sorted(indices[:max_frames])


def build_pair_index(num_frames: int) -> List[tuple[int, int]]:
    return list(combinations(range(num_frames), 2))


def closed_form_inverse_se3(se3: np.ndarray) -> np.ndarray:
    mats = np.asarray(se3, dtype=np.float64)
    if mats.ndim == 2:
        mats = mats[None]
    out = np.tile(np.eye(4, dtype=np.float64), (mats.shape[0], 1, 1))
    rot = mats[:, :3, :3]
    trans = mats[:, :3, 3]
    rot_inv = np.transpose(rot, (0, 2, 1))
    out[:, :3, :3] = rot_inv
    out[:, :3, 3] = -(rot_inv @ trans[..., None]).squeeze(-1)
    return out


def align_to_first_camera(camera_poses: np.ndarray) -> np.ndarray:
    camera_poses = np.asarray(camera_poses, dtype=np.float64)
    first_inv = closed_form_inverse_se3(camera_poses[0])[0]
    return camera_poses @ first_inv


def rotation_angle(rot_gt: np.ndarray, rot_pred: np.ndarray) -> np.ndarray:
    rel = rot_gt @ np.transpose(rot_pred, (0, 2, 1))
    cos = (np.trace(rel, axis1=1, axis2=2) - 1.0) * 0.5
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def compare_translation_by_angle(t_gt: np.ndarray, t_pred: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    t_gt = t_gt / (np.linalg.norm(t_gt, axis=1, keepdims=True) + eps)
    t_pred = t_pred / (np.linalg.norm(t_pred, axis=1, keepdims=True) + eps)
    sinsq = np.clip(1.0 - np.sum(t_gt * t_pred, axis=1) ** 2, eps, None)
    angle = np.arccos(np.sqrt(1.0 - sinsq))
    invalid = ~np.isfinite(angle)
    angle[invalid] = 1e6
    return angle


def translation_angle(t_gt: np.ndarray, t_pred: np.ndarray, ambiguity: bool = True) -> np.ndarray:
    rel = np.degrees(compare_translation_by_angle(t_gt, t_pred))
    if ambiguity:
        rel = np.minimum(rel, np.abs(180.0 - rel))
    return rel


def se3_to_relative_pose_error(pred_se3: np.ndarray, gt_se3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = build_pair_index(len(pred_se3))
    if not pairs:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    gt_inv = closed_form_inverse_se3(gt_se3)
    pred_inv = closed_form_inverse_se3(pred_se3)
    rel_gt = []
    rel_pred = []
    for idx_i, idx_j in pairs:
        rel_gt.append(gt_inv[idx_i] @ gt_se3[idx_j])
        rel_pred.append(pred_inv[idx_i] @ pred_se3[idx_j])
    rel_gt = np.asarray(rel_gt, dtype=np.float64)
    rel_pred = np.asarray(rel_pred, dtype=np.float64)
    return (
        rotation_angle(rel_gt[:, :3, :3], rel_pred[:, :3, :3]),
        translation_angle(rel_gt[:, :3, 3], rel_pred[:, :3, 3]),
    )


def calculate_auc_np(r_error: np.ndarray, t_error: np.ndarray, max_threshold: int = 30) -> float:
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    normalized = histogram.astype(np.float64) / float(len(max_errors))
    return float(np.mean(np.cumsum(normalized)))


def compute_pose_metrics(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> Dict[str, float]:
    pred_aligned = align_to_first_camera(pred_w2c)
    gt_aligned = align_to_first_camera(gt_w2c)
    r_error, t_error = se3_to_relative_pose_error(pred_aligned, gt_aligned)
    return {
        "Auc3": calculate_auc_np(r_error, t_error, max_threshold=3),
        "Auc5": calculate_auc_np(r_error, t_error, max_threshold=5),
        "Auc15": calculate_auc_np(r_error, t_error, max_threshold=15),
        "Auc30": calculate_auc_np(r_error, t_error, max_threshold=30),
    }


def build_summary(scene_metrics: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not scene_metrics:
        return {"Auc3": 0.0, "Auc30": 0.0, "num_scenes": 0}
    return {
        "Auc3": float(np.mean([float(item["Auc3"]) for item in scene_metrics])),
        "Auc30": float(np.mean([float(item["Auc30"]) for item in scene_metrics])),
        "num_scenes": int(len(scene_metrics)),
    }


def read_hiroom_scene_list(scene_list_path: Path) -> List[str]:
    if not scene_list_path.is_file():
        raise FileNotFoundError(f"HiRoom scene list not found: {scene_list_path}")
    return [line.strip() for line in scene_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def discover_scene_dirs(root: Path, preferred_marker: str = "") -> List[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    scene_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    if preferred_marker:
        marked = [path.name for path in scene_dirs if preferred_marker in path.name]
        if marked:
            return marked
    return [path.name for path in scene_dirs]


def _sort_camera_name(name: str) -> tuple[int, str]:
    return (0, f"{int(name):08d}") if str(name).isdigit() else (1, str(name))


def _load_evc_scene_data(scene_root: Path, read_camera_fn=None) -> SimpleNamespace:
    if read_camera_fn is None:
        from easyvolcap.utils.easy_utils import read_camera as read_camera_fn

    cameras = read_camera_fn(str(scene_root / "intri.yml"), str(scene_root / "extri.yml"), use_dict=False)
    image_files: List[str] = []
    extrinsics: List[np.ndarray] = []
    intrinsics: List[np.ndarray] = []

    for camera_name in sorted((str(name) for name in cameras.keys()), key=_sort_camera_name):
        frame_root = scene_root / "images" / camera_name
        if not frame_root.is_dir():
            continue
        image_path = next(
            (
                candidate
                for candidate in sorted(frame_root.iterdir())
                if candidate.is_file() and candidate.suffix.lower() in (".png", ".jpg", ".jpeg")
            ),
            None,
        )
        if image_path is None:
            continue
        camera = cameras[camera_name]
        image_files.append(str(image_path))
        extrinsics.append(as_homogeneous_extrinsics(np.asarray(camera.RT, dtype=np.float32))[0])
        intrinsics.append(np.asarray(camera.K, dtype=np.float32))

    return SimpleNamespace(
        image_files=image_files,
        extrinsics=np.asarray(extrinsics, dtype=np.float32),
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
    )


def _read_field(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item[name]
    return getattr(item, name)


def _intrinsics_from_colmap_camera(camera: Any) -> np.ndarray:
    model = str(_read_field(camera, "model"))
    params = np.asarray(_read_field(camera, "params"), dtype=np.float32)
    if model == "SIMPLE_PINHOLE":
        fx = fy = float(params[0])
        cx, cy = float(params[1]), float(params[2])
    elif model in ("PINHOLE", "OPENCV", "THIN_PRISM_FISHEYE"):
        fx, fy, cx, cy = [float(value) for value in params[:4]]
    else:
        raise NotImplementedError(f"Unsupported ETH3D camera model: {model}")

    ixt = np.eye(3, dtype=np.float32)
    ixt[0, 0] = fx
    ixt[1, 1] = fy
    ixt[0, 2] = cx
    ixt[1, 2] = cy
    return ixt


def load_eth3d_scene_data(
    root: Path,
    scene: str,
    read_camera_fn=None,
    read_cameras_text_fn=None,
    read_images_text_fn=None,
    qvec2rotmat_fn=None,
    filter_keys: Mapping[str, Sequence[str]] | None = None,
) -> SimpleNamespace:
    scene_root = root / scene
    if (scene_root / "images").is_dir() and (scene_root / "intri.yml").is_file() and (scene_root / "extri.yml").is_file():
        return _load_evc_scene_data(scene_root, read_camera_fn=read_camera_fn)

    prepared_image_root = scene_root / "images" / "custom_undistorted"
    prepared_cam_root = scene_root / "custom_undistorted_cam"
    if prepared_image_root.is_dir() and prepared_cam_root.is_dir():
        blocked = set(filter_keys.get(scene, ()) if filter_keys is not None else DA3_ETH3D_FILTER_KEYS.get(scene, ()))
        image_files: List[str] = []
        extrinsics: List[np.ndarray] = []
        intrinsics: List[np.ndarray] = []
        for image_path in sorted(prepared_image_root.iterdir()):
            if not image_path.is_file() or image_path.name in blocked:
                continue
            camera_path = prepared_cam_root / f"{image_path.stem}.npz"
            if not camera_path.is_file():
                continue
            camera = np.load(camera_path)
            if "K" in camera:
                ixt = np.asarray(camera["K"], dtype=np.float32)
            elif "intrinsics" in camera:
                ixt = np.asarray(camera["intrinsics"], dtype=np.float32)
            else:
                raise KeyError(f"Missing ETH3D camera intrinsics in {camera_path}: expected K or intrinsics")
            if "RT" in camera:
                ext = as_homogeneous_extrinsics(np.asarray(camera["RT"], dtype=np.float32))[0]
            elif "extrinsics" in camera:
                ext = as_homogeneous_extrinsics(np.asarray(camera["extrinsics"], dtype=np.float32))[0]
            else:
                ext = np.eye(4, dtype=np.float32)
                ext[:3, :3] = np.asarray(camera["R"], dtype=np.float32)
                ext[:3, 3] = np.asarray(camera["T"], dtype=np.float32).reshape(-1)[:3]
            image_files.append(str(image_path))
            extrinsics.append(ext)
            intrinsics.append(ixt)
        return SimpleNamespace(
            image_files=image_files,
            extrinsics=np.asarray(extrinsics, dtype=np.float32),
            intrinsics=np.asarray(intrinsics, dtype=np.float32),
        )

    calibration_root = next(
        (
            candidate
            for candidate in (
                scene_root / "dslr_calibration_jpg",
                scene_root / "calibration" / "dslr_calibration_jpg",
            )
            if (candidate / "cameras.txt").is_file() and (candidate / "images.txt").is_file()
        ),
        None,
    )
    image_root = next(
        (
            candidate
            for candidate in (
                scene_root / "images" / "dslr_images",
                scene_root / "images",
            )
            if candidate.is_dir()
        ),
        None,
    )
    if calibration_root is not None and image_root is not None:
        if read_cameras_text_fn is None or read_images_text_fn is None or qvec2rotmat_fn is None:
            from easyvolcap.utils.colmap_utils import qvec2rotmat as qvec2rotmat_fn
            from easyvolcap.utils.colmap_utils import read_cameras_text as read_cameras_text_fn
            from easyvolcap.utils.colmap_utils import read_images_text as read_images_text_fn

        cameras = read_cameras_text_fn(str(calibration_root / "cameras.txt"))
        images = read_images_text_fn(str(calibration_root / "images.txt"))
        blocked = set(filter_keys.get(scene, ()) if filter_keys is not None else DA3_ETH3D_FILTER_KEYS.get(scene, ()))

        image_files: List[str] = []
        extrinsics: List[np.ndarray] = []
        intrinsics: List[np.ndarray] = []
        for image in sorted(images.values(), key=lambda item: str(_read_field(item, "name"))):
            image_name = str(_read_field(image, "name"))
            image_leaf_name = Path(image_name).name
            if image_name in blocked or image_leaf_name in blocked:
                continue
            candidate_paths = [image_root / image_name]
            if image_leaf_name != image_name:
                candidate_paths.append(image_root / image_leaf_name)
            image_path = next((candidate for candidate in candidate_paths if candidate.is_file()), None)
            if image_path is None:
                continue
            camera = cameras[int(_read_field(image, "camera_id"))]
            ext = np.eye(4, dtype=np.float32)
            ext[:3, :3] = np.asarray(qvec2rotmat_fn(np.asarray(_read_field(image, "qvec"), dtype=np.float32)), dtype=np.float32)
            ext[:3, 3] = np.asarray(_read_field(image, "tvec"), dtype=np.float32).reshape(-1)[:3]
            image_files.append(str(image_path))
            extrinsics.append(ext)
            intrinsics.append(_intrinsics_from_colmap_camera(camera))
        return SimpleNamespace(
            image_files=image_files,
            extrinsics=np.asarray(extrinsics, dtype=np.float32),
            intrinsics=np.asarray(intrinsics, dtype=np.float32),
        )

    raise FileNotFoundError(f"Unsupported ETH3D layout for scene {scene}: {scene_root}")


def load_hiroom_scene_data(root: Path, scene: str) -> SimpleNamespace:
    scene_root = root / scene
    image_root = scene_root / "image"
    pose_root = scene_root / "pose"
    intrinsics = np.asarray(np.load(scene_root / "cam_K.npy"), dtype=np.float32)

    image_files: List[str] = []
    extrinsics: List[np.ndarray] = []
    intrinsics_list: List[np.ndarray] = []
    for image_path in sorted(image_root.iterdir()):
        if not image_path.is_file():
            continue
        frame_name = image_path.stem
        pose_path = pose_root / f"{frame_name}.npy"
        if not pose_path.is_file():
            continue
        image_files.append(str(image_path))
        extrinsics.append(np.asarray(np.load(pose_path), dtype=np.float32))
        intrinsics_list.append(intrinsics.copy())

    return SimpleNamespace(
        image_files=image_files,
        extrinsics=np.asarray(extrinsics, dtype=np.float32),
        intrinsics=np.asarray(intrinsics_list, dtype=np.float32),
    )


def read_dtu_camera_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]
    extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape((4, 4))
    intrinsics = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ").reshape((3, 3))
    return intrinsics, extrinsics


def reorder_dtu_reference_view(files: Sequence[Path]) -> List[Path]:
    files = list(files)
    if len(files) > 33:
        return [files[33], *files[:33], *files[34:]]
    return files


def load_dtu64_scene_data(
    root: Path,
    camera_root: Path,
    scene: str,
    read_camera_fn=None,
) -> SimpleNamespace:
    scene_root = root / scene
    image_root = scene_root / "image"
    if image_root.is_dir():
        files = reorder_dtu_reference_view(sorted(image_root.glob("*.png")))
        image_files: List[str] = []
        extrinsics: List[np.ndarray] = []
        intrinsics: List[np.ndarray] = []
        for image_path in files:
            cam_idx = int(image_path.stem)
            camera_path = camera_root / f"{cam_idx:08d}_cam.txt"
            if not camera_path.is_file():
                continue
            ixt, ext = read_dtu_camera_file(camera_path)
            image_files.append(str(image_path))
            extrinsics.append(ext)
            intrinsics.append(ixt)
        return SimpleNamespace(
            image_files=image_files,
            extrinsics=np.asarray(extrinsics, dtype=np.float32),
            intrinsics=np.asarray(intrinsics, dtype=np.float32),
        )

    image_root = scene_root / "images" / "00"
    scene_camera_root = scene_root / "cameras" / "00"
    if image_root.is_dir() and (scene_camera_root / "intri.yml").is_file() and (scene_camera_root / "extri.yml").is_file():
        if read_camera_fn is None:
            from easyvolcap.utils.easy_utils import read_camera as read_camera_fn

        cameras = read_camera_fn(str(scene_camera_root / "intri.yml"), str(scene_camera_root / "extri.yml"), use_dict=False)

        def _sort_key(name: str) -> tuple[int, str]:
            return (0, f"{int(name):08d}") if str(name).isdigit() else (1, str(name))

        ordered_names = sorted((str(name) for name in cameras.keys()), key=_sort_key)
        if len(ordered_names) > 33:
            ordered_names = [ordered_names[33], *ordered_names[:33], *ordered_names[34:]]

        image_files: List[str] = []
        extrinsics: List[np.ndarray] = []
        intrinsics: List[np.ndarray] = []
        for camera_name in ordered_names:
            image_path = None
            for suffix in (".png", ".jpg", ".jpeg"):
                candidate = image_root / f"{camera_name}{suffix}"
                if candidate.is_file():
                    image_path = candidate
                    break
            if image_path is None:
                continue
            camera = cameras[camera_name]
            image_files.append(str(image_path))
            extrinsics.append(as_homogeneous_extrinsics(np.asarray(camera.RT, dtype=np.float32))[0])
            intrinsics.append(np.asarray(camera.K, dtype=np.float32))

        return SimpleNamespace(
            image_files=image_files,
            extrinsics=np.asarray(extrinsics, dtype=np.float32),
            intrinsics=np.asarray(intrinsics, dtype=np.float32),
        )

    raise FileNotFoundError(f"Unsupported DTU64 layout for scene {scene}: {scene_root}")


def load_scannetpp_scene_data(
    root: Path,
    scene: str,
    read_model_fn=None,
    read_camera_fn=None,
) -> SimpleNamespace:
    scene_root = root / scene
    if (scene_root / "merge_dslr_iphone").is_dir():
        if read_model_fn is None:
            from easyvolcap.utils.colmap_utils import read_model as read_model_fn

        input_root = scene_root / "merge_dslr_iphone"
        colmap_root = input_root / "colmap" / "sparse_render_rgb"
        image_root = input_root / "images"
        cameras, images, _ = read_model_fn(str(colmap_root))
        name_to_id = {image.name: image_id for image_id, image in images.items()}
        names = sorted(image.name for image in images.values())
        names = [name for name in names if "iphone" in name]

        image_files: List[str] = []
        extrinsics: List[np.ndarray] = []
        intrinsics: List[np.ndarray] = []

        for name in names:
            image_path = image_root / name
            if not image_path.is_file():
                continue
            image = images[name_to_id[name]]
            camera = cameras[image.camera_id]
            ext = np.eye(4, dtype=np.float32)
            ext[:3, :3] = np.asarray(image.qvec2rotmat(), dtype=np.float32)
            ext[:3, 3] = np.asarray(image.tvec, dtype=np.float32)

            ixt = np.eye(3, dtype=np.float32)
            ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2] = np.asarray(camera.params[:4], dtype=np.float32)
            ixt[:2, 2] -= 0.5
            if getattr(camera, "model", "") == "OPENCV":
                import cv2

                dist = np.zeros(5, dtype=np.float32)
                dist[:4] = np.asarray(camera.params[4:8], dtype=np.float32)
                ixt, _ = cv2.getOptimalNewCameraMatrix(
                    ixt,
                    dist,
                    (int(camera.width), int(camera.height)),
                    1,
                    (int(camera.width), int(camera.height)),
                )

            image_files.append(str(image_path))
            extrinsics.append(ext)
            intrinsics.append(ixt)

        return SimpleNamespace(
            image_files=image_files,
            extrinsics=np.asarray(extrinsics, dtype=np.float32),
            intrinsics=np.asarray(intrinsics, dtype=np.float32),
        )

    if (scene_root / "images").is_dir() and (scene_root / "intri.yml").is_file() and (scene_root / "extri.yml").is_file():
        if read_camera_fn is None:
            from easyvolcap.utils.easy_utils import read_camera as read_camera_fn

        cameras = read_camera_fn(str(scene_root / "intri.yml"), str(scene_root / "extri.yml"), use_dict=False)
        image_files: List[str] = []
        extrinsics: List[np.ndarray] = []
        intrinsics: List[np.ndarray] = []

        for camera_name in sorted((str(name) for name in cameras.keys()), key=_sort_camera_name):
            image_root = scene_root / "images" / camera_name
            image_path = None
            for suffix in (".jpg", ".png", ".jpeg"):
                candidate = image_root / f"000000{suffix}"
                if candidate.is_file():
                    image_path = candidate
                    break
            if image_path is None:
                continue
            camera = cameras[camera_name]
            image_files.append(str(image_path))
            extrinsics.append(as_homogeneous_extrinsics(np.asarray(camera.RT, dtype=np.float32))[0])
            intrinsics.append(np.asarray(camera.K, dtype=np.float32))

        return SimpleNamespace(
            image_files=image_files,
            extrinsics=np.asarray(extrinsics, dtype=np.float32),
            intrinsics=np.asarray(intrinsics, dtype=np.float32),
        )

    raise FileNotFoundError(f"Unsupported ScanNet++ layout for scene {scene}: {scene_root}")


def load_7scenes_scene_data(
    root: Path,
    scene: str,
    read_camera_fn=None,
) -> SimpleNamespace:
    official_scene_root = next(
        (
            candidate
            for candidate in (root / "7Scenes" / scene, root / scene)
            if candidate.is_dir() and (candidate / "TestSplit.txt").is_file()
        ),
        None,
    )
    if official_scene_root is not None:
        split_lines = (official_scene_root / "TestSplit.txt").read_text(encoding="utf-8").splitlines()
        seq_ids = [
            int(match.group(1))
            for line in split_lines
            if (match := re.search(r"(\d+)", line.strip())) is not None
        ]
        fixed_ixt = np.array(
            [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        image_files: List[str] = []
        extrinsics: List[np.ndarray] = []
        intrinsics: List[np.ndarray] = []
        for seq_id in sorted(seq_ids):
            seq_root = official_scene_root / f"seq-{seq_id:02d}"
            if not seq_root.is_dir():
                continue
            for image_path in sorted(seq_root.glob("frame-*.color.png")):
                pose_path = seq_root / image_path.name.replace(".color.png", ".pose.txt")
                if not pose_path.is_file():
                    continue
                pose_c2w = np.loadtxt(pose_path, dtype=np.float32).reshape((4, 4))
                image_files.append(str(image_path))
                extrinsics.append(np.linalg.inv(pose_c2w).astype(np.float32))
                intrinsics.append(fixed_ixt.copy())
        return SimpleNamespace(
            image_files=image_files,
            extrinsics=np.asarray(extrinsics, dtype=np.float32),
            intrinsics=np.asarray(intrinsics, dtype=np.float32),
        )

    grouped_sequence_roots = [
        candidate
        for base_root in (root, root / "test")
        if base_root.is_dir()
        for candidate in sorted(base_root.glob(f"{scene}_seq-*"))
        if candidate.is_dir()
        and (candidate / "images").is_dir()
        and (candidate / "intri.yml").is_file()
        and (candidate / "extri.yml").is_file()
    ]
    if grouped_sequence_roots:
        image_files: List[str] = []
        extrinsics: List[np.ndarray] = []
        intrinsics: List[np.ndarray] = []
        for scene_root in grouped_sequence_roots:
            scene_data = _load_evc_scene_data(scene_root, read_camera_fn=read_camera_fn)
            image_files.extend(scene_data.image_files)
            extrinsics.extend(np.asarray(scene_data.extrinsics, dtype=np.float32))
            intrinsics.extend(np.asarray(scene_data.intrinsics, dtype=np.float32))
        return SimpleNamespace(
            image_files=image_files,
            extrinsics=np.asarray(extrinsics, dtype=np.float32),
            intrinsics=np.asarray(intrinsics, dtype=np.float32),
        )

    candidate_roots = [root / scene, root / "test" / scene]
    scene_root = next(
        (
            candidate
            for candidate in candidate_roots
            if candidate.is_dir()
            and (candidate / "images").is_dir()
            and (candidate / "intri.yml").is_file()
            and (candidate / "extri.yml").is_file()
        ),
        None,
    )
    if scene_root is None:
        raise FileNotFoundError(f"Unsupported 7Scenes layout for scene {scene}: {candidate_roots[0]}")
    return _load_evc_scene_data(scene_root, read_camera_fn=read_camera_fn)


def select_scene_names(dataset_name: str, roots: Mapping[str, Path], requested_scenes: Sequence[str]) -> List[str]:
    if requested_scenes:
        return [scene for scene in requested_scenes]
    if dataset_name == "eth3d":
        existing = [scene for scene in DA3_ETH3D_SCENES if (roots["eth3d"] / scene).is_dir()]
        return existing or list(DA3_ETH3D_SCENES)
    if dataset_name == "hiroom":
        return read_hiroom_scene_list(roots["hiroom"].parent / "selected_scene_list_val.txt")
    if dataset_name == "scannetpp":
        return list(DA3_SCANNETPP_SCENES)
    if dataset_name == "dtu64":
        return list(DA3_DTU64_SCENES)
    if dataset_name == "7scenes":
        official_root = roots["7scenes"] / "7Scenes"
        if official_root.is_dir():
            existing = [scene for scene in DA3_7SCENES_SCENES if (official_root / scene).is_dir()]
            if existing:
                return existing
        discovered = discover_scene_dirs(roots["7scenes"], preferred_marker="_seq-")
        grouped = []
        for scene in DA3_7SCENES_SCENES:
            prefix = f"{scene}_seq-"
            if any(name.startswith(prefix) for name in discovered):
                grouped.append(scene)
        return grouped or discovered
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_scene_data(dataset_name: str, roots: Mapping[str, Path], scene: str, dtu64_camera_root: Path | None) -> SimpleNamespace:
    if dataset_name == "eth3d":
        return load_eth3d_scene_data(roots["eth3d"], scene)
    if dataset_name == "hiroom":
        return load_hiroom_scene_data(roots["hiroom"], scene)
    if dataset_name == "scannetpp":
        return load_scannetpp_scene_data(roots["scannetpp"], scene)
    if dataset_name == "dtu64":
        camera_root = dtu64_camera_root or (roots["dtu64"] / "Cameras")
        return load_dtu64_scene_data(roots["dtu64"], camera_root, scene)
    if dataset_name == "7scenes":
        return load_7scenes_scene_data(roots["7scenes"], scene)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def subsample_scene_data(scene_data: SimpleNamespace, max_frames: int, seed: int) -> SimpleNamespace:
    indices = sample_frame_indices(len(scene_data.image_files), max_frames=max_frames, seed=seed)
    return SimpleNamespace(
        image_files=[scene_data.image_files[index] for index in indices],
        extrinsics=np.asarray(scene_data.extrinsics[indices], dtype=np.float32),
        intrinsics=np.asarray(scene_data.intrinsics[indices], dtype=np.float32),
    )


def default_output_dir(args: argparse.Namespace) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return repo_root() / "tmp" / f"vggt_da3_pose_{args.vggt_model_tag}_{timestamp}"


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def as_homogeneous_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    if extrinsics.ndim == 2:
        extrinsics = extrinsics[None]
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Unsupported extrinsics shape: {extrinsics.shape}")
    hom = np.tile(np.eye(4, dtype=np.float32), (extrinsics.shape[0], 1, 1))
    hom[:, :3, :4] = extrinsics
    return hom


def c2w_to_w2c(camera_poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(camera_poses, dtype=np.float32)
    if poses.ndim == 2:
        poses = poses[None]
    if poses.shape[-2:] != (4, 4):
        raise ValueError(f"Expected camera poses with shape (N,4,4), got {poses.shape}")
    return np.linalg.inv(poses).astype(np.float32)


def load_vggt_runtime_module():
    return load_module_from_path(
        "eval_pi3_relpose_distance_protocol_runtime",
        repo_root() / "aidi" / "scripts" / "baselines" / "eval_pi3_relpose_distance_protocol.py",
    )


def load_model(args: argparse.Namespace):
    runtime = load_vggt_runtime_module()
    device = runtime.resolve_device(args.device)
    if args.model_family == "vggt_omega":
        if not args.vggt_omega_repo:
            raise ValueError("--vggt-omega-repo is required when --model-family=vggt_omega")
        if not args.vggt_omega_checkpoint:
            raise ValueError("--vggt-omega-checkpoint is required when --model-family=vggt_omega")
        from aidi.scripts.vggt.vggt_omega_eval_utils import load_vggt_omega_model

        model, loaded_ckpt = load_vggt_omega_model(
            repo_path=args.vggt_omega_repo,
            checkpoint_path=args.vggt_omega_checkpoint,
            device=device,
            global_attention_mode=args.vggt_omega_global_attention_mode,
        )
        return model, device, runtime, loaded_ckpt
    model, loaded_ckpt = runtime.load_model_runtime(args=args, device=device)
    return model, device, runtime, loaded_ckpt


def infer_vggt_extrinsics_w2c(
    image_files: Sequence[str],
    model,
    device,
    runtime_module,
    args: argparse.Namespace,
) -> np.ndarray:
    import torch

    from easyvolcap.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from easyvolcap.utils.base_utils import dotdict

    images = preprocess_images_for_vggt(
        image_files=image_files,
        image_preprocess_style=args.image_preprocess_style,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
    ).to(device=device).unsqueeze(0)
    height, width = images.shape[-2:]
    batch = dotdict(
        images=images,
        meta=dotdict(
            iter=torch.tensor(0, device=images.device),
            H=torch.tensor([height], device=images.device),
            W=torch.tensor([width], device=images.device),
        ),
    )
    dtype = runtime_module.resolve_autocast_dtype(device)
    autocast_enabled = getattr(device, "type", "") == "cuda"
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            output = model(batch)
    if not hasattr(output, "cam_map"):
        raise RuntimeError("VGGT DA3 pose inference requires output.cam_map")
    extrinsics, _ = pose_encoding_to_extri_intri(output.cam_map, image_size_hw=(height, width))
    return as_homogeneous_extrinsics(extrinsics[0].detach().float().cpu().numpy())


def infer_pi3_extrinsics_w2c(
    image_files: Sequence[str],
    model,
    device,
    runtime_module,
    args: argparse.Namespace,
) -> np.ndarray:
    pred_c2w = runtime_module.infer_pi3_cameras_c2w(
        image_names=[str(path) for path in image_files],
        model=model,
        device=device,
        load_img_size=args.load_img_size,
        image_load_retries=args.image_load_retries,
        image_load_retry_sleep=args.image_load_retry_sleep,
    )
    return c2w_to_w2c(pred_c2w)


def infer_model_extrinsics_w2c(
    image_files: Sequence[str],
    model,
    device,
    runtime_module,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.model_family == "pi3":
        return infer_pi3_extrinsics_w2c(
            image_files=image_files,
            model=model,
            device=device,
            runtime_module=runtime_module,
            args=args,
        )
    if args.model_family == "vggt_omega":
        from aidi.scripts.vggt.vggt_omega_eval_utils import as_homogeneous_w2c, predict_vggt_omega

        _, extrinsics, _ = predict_vggt_omega(
            image_files=image_files,
            model=model,
            repo_path=args.vggt_omega_repo,
            image_resolution=args.vggt_omega_resolution,
            mode=args.vggt_omega_mode,
            device=device,
        )
        return as_homogeneous_w2c(extrinsics)
    return infer_vggt_extrinsics_w2c(
        image_files=image_files,
        model=model,
        device=device,
        runtime_module=runtime_module,
        args=args,
    )


def evaluate_dataset(
    dataset_name: str,
    roots: Mapping[str, Path],
    model,
    device,
    runtime_module,
    args: argparse.Namespace,
    output_root: Path,
    dtu64_camera_root: Path | None,
) -> Dict[str, Any]:
    requested_scenes = parse_csv(args.scenes)
    scene_names = select_scene_names(dataset_name, roots, requested_scenes=requested_scenes)
    dataset_dir = output_root / "model_results" / dataset_name
    metrics_by_scene: Dict[str, Dict[str, float]] = {}

    for scene in scene_names:
        scene_data = load_scene_data(dataset_name, roots, scene, dtu64_camera_root=dtu64_camera_root)
        if len(scene_data.image_files) < 2:
            continue
        scene_data = subsample_scene_data(scene_data, max_frames=args.max_frames, seed=args.seed)
        pred_w2c = infer_model_extrinsics_w2c(
            image_files=scene_data.image_files,
            model=model,
            device=device,
            runtime_module=runtime_module,
            args=args,
        )
        metrics = compute_pose_metrics(pred_w2c=pred_w2c, gt_w2c=scene_data.extrinsics)
        metrics_by_scene[scene] = metrics

        scene_dir = dataset_dir / scene.replace("/", "__")
        scene_dir.mkdir(parents=True, exist_ok=True)
        np.save(scene_dir / "pred_extrinsics_w2c.npy", pred_w2c)
        np.save(scene_dir / "gt_extrinsics_w2c.npy", scene_data.extrinsics)
        save_json(scene_dir / "scene_metrics.json", metrics)
        (scene_dir / "image_files.txt").write_text("\n".join(scene_data.image_files) + "\n", encoding="utf-8")
        if args.verbose:
            print(
                f"[{dataset_name}] {scene}: "
                f"Auc3={metrics['Auc3']:.4f} "
                f"Auc30={metrics['Auc30']:.4f} "
                f"frames={len(scene_data.image_files)}",
                flush=True,
            )

    summary = build_summary(list(metrics_by_scene.values()))
    payload = {
        "dataset": dataset_name,
        "summary": summary,
        "scenes": metrics_by_scene,
    }
    save_json(output_root / "metric_results" / f"{dataset_name}_pose.json", payload)
    return payload


def main() -> int:
    args = parse_args()
    datasets = parse_csv(args.datasets)
    roots = resolve_data_roots(args)
    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(args)
    dtu64_camera_root = (
        Path(args.dtu64_camera_root).expanduser().resolve()
        if args.dtu64_camera_root
        else roots["dtu64"] / "Cameras"
    )

    model, device, runtime_module, loaded_ckpt = load_model(args)
    all_results: Dict[str, Any] = {
        "model_family": args.model_family,
        "model_tag": args.vggt_model_tag if args.model_family == "vggt" else args.model_family,
        "loaded_ckpt": loaded_ckpt,
        "device": str(device),
        "datasets": {},
    }
    if args.model_family == "vggt_omega":
        all_results["vggt_omega_global_attention_mode"] = args.vggt_omega_global_attention_mode
    for dataset_name in datasets:
        result = evaluate_dataset(
            dataset_name=dataset_name,
            roots=roots,
            model=model,
            device=device,
            runtime_module=runtime_module,
            args=args,
            output_root=output_root,
            dtu64_camera_root=dtu64_camera_root,
        )
        all_results["datasets"][dataset_name] = result["summary"]
    save_json(output_root / "metric_results" / "summary.json", all_results)
    print(json.dumps(all_results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
