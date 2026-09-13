#!/usr/bin/env python3
"""PI3 RE10K pose evaluator under the current aligned RE10K protocol."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


DEFAULT_RE10K_ROOT = (
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/datasets/re10k/processed_pose1800_clusterfix/test"
)
DEFAULT_PI3_MODEL = "yyfz233/Pi3"
DEFAULT_SEED = 20260215
DEFAULT_POOL_SIZE = 10
DEFAULT_N_SRCS = 9
DEFAULT_LOAD_IMG_SIZE = 518


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_repo_root_on_syspath() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


ensure_repo_root_on_syspath()


def setup_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PI3 evaluator aligned to current RE10K paper10 protocol")
    parser.add_argument("--re10k-root", default=DEFAULT_RE10K_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--n-srcs", type=int, default=DEFAULT_N_SRCS)
    parser.add_argument(
        "--use-scene-order",
        action="store_true",
        help="Load sorted scene frames directly. Use this for pre-staged tuple roots.",
    )
    parser.add_argument(
        "--eval-frame-indices",
        default="",
        help="Comma-separated input indices used for pose metrics after full-input inference.",
    )
    parser.add_argument(
        "--translation-error-mode",
        default="relative_transform",
        choices=["relative_transform", "centers"],
        help=(
            "Pose translation angular error mode. relative_transform preserves the legacy evaluator; "
            "centers uses world-frame camera-center direction errors."
        ),
    )
    parser.add_argument(
        "--auc-combine",
        default="max",
        choices=["max", "min"],
        help="AUC combination rule: max preserves legacy max(error_r,error_t); min is paper-style min(RRA,RTA).",
    )
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--scene-filter", default="")
    parser.add_argument("--load-img-size", type=int, default=DEFAULT_LOAD_IMG_SIZE)
    parser.add_argument("--image-load-retries", type=int, default=8)
    parser.add_argument("--image-load-retry-sleep", type=float, default=0.5)
    parser.add_argument("--camera-cache-root", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--model-tag", default="pi3")
    return parser.parse_args(list(argv) if argv is not None else None)


def default_model_path() -> str:
    candidates = (
        Path("/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/pretrained/pi3/yyfz233_Pi3_model.safetensors"),
        Path("/horizon-bucket/saturn_v_dev/01_users/horizon/projects/meshx/baseline/pretrained/pi3/yyfz233_Pi3_model.safetensors"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return DEFAULT_PI3_MODEL


def default_output_path(args: argparse.Namespace) -> Path:
    base = f"re10k_pi3_aligned_{args.model_tag}_seed{args.seed}.json"
    return Path("/tmp") / base


def load_images_for_pi3(filelist: List[str], new_width: int, device: str, verbose: bool):
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as tvf
    from PIL import Image

    to_tensor = tvf.ToTensor()
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
        imgs.append(to_tensor(resized))
    tensor = torch.stack(imgs, dim=0).to(device)

    patch_h, patch_w = tensor.shape[-2] // 14, tensor.shape[-1] // 14
    tensor = F.interpolate(
        tensor,
        (patch_h * 14, patch_w * 14),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).unsqueeze(0)
    return tensor


def load_pi3_model(Pi3, ckpt: str, device):
    from aidi.scripts.baselines.pi3_checkpoint_loader import load_pi3_model_for_eval

    return load_pi3_model_for_eval(
        ckpt,
        device,
        official_pi3_cls=Pi3,
        native_root=os.environ.get("PI3_NATIVE_ROOT", ""),
        model_impl=os.environ.get("PI3_MODEL_IMPL", ""),
        config_path=os.environ.get("PI3_CONFIG", ""),
    )


def camera_poses_to_w2c(camera_poses: Any) -> np.ndarray:
    poses = np.asarray(camera_poses, dtype=np.float64)
    if poses.ndim == 4:
        if poses.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for camera_poses, got {poses.shape}")
        poses = poses[0]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected (N,4,4) camera_poses, got {poses.shape}")
    return np.linalg.inv(poses)[:, :3, :4]


def extract_pred_extrinsics(predictions: Dict[str, Any]) -> np.ndarray:
    if "camera_poses" not in predictions:
        raise KeyError("camera_poses")
    camera_poses = predictions["camera_poses"]
    if hasattr(camera_poses, "detach"):
        camera_poses = camera_poses.detach().float().cpu().numpy()
    return camera_poses_to_w2c(camera_poses)


def stable_scene_seed(scene_key: str, seed: int = DEFAULT_SEED) -> int:
    payload = f"{int(seed)}:{scene_key}".encode("utf-8", errors="ignore")
    digest = hashlib.sha1(payload).digest()[:8]
    return int.from_bytes(digest, byteorder="little", signed=False)


def sample_scene_pool_indices(scene_key: str, total_frames: int, pool_size: int = DEFAULT_POOL_SIZE, seed: int = DEFAULT_SEED) -> List[int]:
    n = int(total_frames)
    if n <= 0:
        return [0]
    k = min(max(1, int(pool_size)), n)
    rng = random.Random(stable_scene_seed(scene_key=scene_key, seed=seed))
    if k < n:
        return list(rng.sample(range(n), k))
    choices = list(range(n))
    rng.shuffle(choices)
    return choices


def select_scene_input_indices(
    scene_key: str,
    total_frames: int,
    seed: int,
    pool_size: int,
    n_srcs: int,
    use_scene_order: bool = False,
) -> List[int]:
    n = int(total_frames)
    if n <= 0:
        return []
    if use_scene_order:
        k = n if int(pool_size) <= 0 else min(n, int(pool_size))
        return list(range(k))
    pool_indices = sample_scene_pool_indices(
        scene_key=scene_key,
        total_frames=n,
        pool_size=pool_size,
        seed=seed,
    )
    target_idx = int(pool_indices[0])
    return [target_idx] + [int(i) for i in pool_indices[1 : 1 + int(n_srcs)]]


def parse_eval_frame_indices(value: str) -> List[int]:
    indices: List[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        index = int(token)
        if index < 0:
            raise ValueError(f"eval-frame-indices must be non-negative, got {index}")
        indices.append(index)
    return indices


def subset_pose_array_for_eval(poses: np.ndarray, eval_indices: Sequence[int]) -> np.ndarray:
    if not eval_indices:
        return poses
    arr = np.asarray(poses)
    max_index = max(int(index) for index in eval_indices)
    if max_index >= arr.shape[0]:
        raise IndexError(f"eval-frame-indices include {max_index}, but pose array has {arr.shape[0]} frames")
    return arr[np.asarray([int(index) for index in eval_indices], dtype=np.int64)]


def build_summary(metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {
            "cam:pose_auc_10_mean": 0.0,
            "cam:pose_auc_20_mean": 0.0,
            "cam:pose_auc_30_mean": 0.0,
            "metrics_count": 0,
        }

    def _mean(key: str) -> float:
        values = [float(item[key]) for item in metrics if key in item]
        return float(np.mean(values)) if values else 0.0

    summary = {
        "cam:pose_auc_10_mean": _mean("cam:pose_auc_10"),
        "cam:pose_auc_20_mean": _mean("cam:pose_auc_20"),
        "cam:pose_auc_30_mean": _mean("cam:pose_auc_30"),
        "metrics_count": int(len(metrics)),
    }
    optional_keys = (
        "cam:rotation_accuracy_01",
        "cam:rotation_accuracy_03",
        "cam:rotation_accuracy_05",
        "cam:rotation_accuracy_15",
        "cam:translation_accuracy_01",
        "cam:translation_accuracy_03",
        "cam:translation_accuracy_05",
        "cam:translation_accuracy_15",
        "cam:translation_scale",
    )
    for key in optional_keys:
        if any(key in item for item in metrics):
            summary[f"{key}_mean"] = _mean(key)
    return summary


def resolve_device(args: argparse.Namespace):
    import torch

    return torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))


def resolve_autocast_dtype(device):
    import torch

    if getattr(device, "type", "") != "cuda":
        return torch.float32
    major = torch.cuda.get_device_capability(device=device)[0]
    return torch.bfloat16 if major >= 8 else torch.float16


def load_pi3_model_runtime(model_path: str, device):
    ensure_repo_root_on_syspath()
    from easyvolcap.utils.pi3.models.pi3 import Pi3

    resolved_path = model_path or default_model_path()
    model, loaded_ckpt = load_pi3_model(Pi3, resolved_path, device)
    return model, loaded_ckpt


def load_images_with_retry(image_names: List[str], new_width: int, device: str, retries: int, retry_sleep: float):
    import time

    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            return load_images_for_pi3(
                filelist=image_names,
                new_width=new_width,
                device=device,
                verbose=False,
            )
        except OSError as exc:
            if attempt >= attempts:
                raise
            print(
                f"[image-retry] attempt={attempt}/{attempts} "
                f"sleep={retry_sleep:.2f}s error={type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(max(0.0, float(retry_sleep)))


def scene_key_from_dir(scene_dir: Path) -> str:
    return f"{scene_dir.parent.name}/{scene_dir.name}"


def resolve_scene_roots(re10k_root: Path, scene_filter: str = "", limit_scenes: int = 0) -> List[Path]:
    if not re10k_root.is_dir():
        raise FileNotFoundError(f"RE10K root not found: {re10k_root}")

    data_roots_file = re10k_root / "data_roots.txt"
    scene_roots: List[Path]
    if data_roots_file.is_file():
        try:
            lines = [line.strip() for line in data_roots_file.read_text().splitlines() if line.strip()]
            scene_roots = [(re10k_root / line).resolve() for line in lines]
        except OSError:
            scene_roots = sorted([path.resolve() for path in re10k_root.iterdir() if path.is_dir()])
    else:
        scene_roots = sorted([path.resolve() for path in re10k_root.iterdir() if path.is_dir()])

    if scene_filter:
        scene_roots = [path for path in scene_roots if scene_filter in path.name]
    if limit_scenes > 0:
        scene_roots = scene_roots[: int(limit_scenes)]
    return scene_roots


def materialize_camera_files(camera_root: Path, cache_root: Path | None = None) -> Path:
    camera_root = Path(camera_root)
    if cache_root is None:
        cache_root = Path("/tmp/pi3_re10k_cam_cache")
    cache_root = Path(cache_root)
    scene_name = camera_root.parents[1].name
    split_name = camera_root.parents[2].name if len(camera_root.parents) >= 3 else "split"
    local_root = cache_root / split_name / scene_name / "cameras" / camera_root.name
    local_root.mkdir(parents=True, exist_ok=True)

    for name in ("intri.yml", "extri.yml"):
        src = camera_root / name
        dst = local_root / name
        if (not dst.is_file()) or dst.stat().st_mtime < src.stat().st_mtime or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
    return local_root


def load_scene_inputs(
    scene_dir: Path,
    seed: int,
    pool_size: int,
    n_srcs: int,
    use_scene_order: bool = False,
    camera_cache_root: Path | str | None = None,
) -> Dict[str, Any]:
    from easyvolcap.utils.easy_utils import read_camera

    camera_root = materialize_camera_files(scene_dir / "cameras" / "00", cache_root=camera_cache_root)
    cams = read_camera(str(camera_root))
    cam_names = sorted(cams.keys(), key=lambda x: int(x))
    if not cam_names:
        raise RuntimeError(f"No camera names found under {scene_dir}")

    scene_key = scene_key_from_dir(scene_dir)
    sampled_indices = select_scene_input_indices(
        scene_key=scene_key,
        total_frames=len(cam_names),
        seed=seed,
        pool_size=pool_size,
        n_srcs=n_srcs,
        use_scene_order=use_scene_order,
    )
    min_expected = 1 if use_scene_order else int(n_srcs) + 1
    if len(sampled_indices) < min_expected:
        raise RuntimeError(
            f"Scene {scene_key} only produced {len(sampled_indices)} sampled frames, "
            f"expected at least {min_expected}"
        )

    sampled_names = [cam_names[i] for i in sampled_indices]
    image_paths = [str(scene_dir / "images" / "00" / f"{name}.jpg") for name in sampled_names]
    missing = [path for path in image_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing sampled images under {scene_dir}: {missing[:3]}")

    gt_w2c = np.stack([np.asarray(cams[name].RT, dtype=np.float64) for name in sampled_names], axis=0)
    return {
        "scene_key": scene_key,
        "scene_name": scene_dir.name,
        "sampled_indices": sampled_indices,
        "sampled_names": sampled_names,
        "image_paths": image_paths,
        "gt_w2c": gt_w2c,
        "target_name": sampled_names[0],
    }


def compute_pose_metrics_with_options(
    gt_w2c: np.ndarray,
    pred_w2c: np.ndarray,
    translation_error_mode: str = "relative_transform",
    auc_combine: str = "max",
) -> Dict[str, float]:
    import torch

    from easyvolcap.utils.cam_utils import (
        calculate_auc_np,
        camera_to_relative_degree,
        camera_to_relative_degree_centers,
    )

    gt = torch.as_tensor(gt_w2c, dtype=torch.float64)
    pred = torch.as_tensor(pred_w2c, dtype=torch.float64)
    mode = str(translation_error_mode or "relative_transform").strip().lower()
    if mode in {"relative_transform", "relative", "legacy"}:
        r_rel, t_rel = camera_to_relative_degree(gt, pred)
    elif mode in {"centers", "center", "camera_centers"}:
        r_rel, t_rel = camera_to_relative_degree_centers(gt, pred)
    else:
        raise ValueError(f"Unknown translation_error_mode={translation_error_mode!r}")

    combine = str(auc_combine or "max").strip().lower()

    metric: Dict[str, float] = {}
    for thresh in (1, 3, 5, 15):
        metric[f"cam:rotation_accuracy_{thresh:02d}"] = float((r_rel < thresh).double().mean().item())
        metric[f"cam:translation_accuracy_{thresh:02d}"] = float((t_rel < thresh).double().mean().item())
    for thresh in (1, 3, 5, 10, 20, 30):
        metric[f"cam:pose_auc_{thresh:02d}"] = float(
            calculate_auc_np(
                r_rel.cpu().numpy(),
                t_rel.cpu().numpy(),
                threshold=thresh,
                combine=combine,
            )
        )

    pred_norm = torch.norm(pred[:, 1:, :3, 3], dim=-1).clamp(min=1e-12) if pred.ndim == 4 else torch.norm(pred[1:, :3, 3], dim=-1).clamp(min=1e-12)
    gt_norm = torch.norm(gt[:, 1:, :3, 3], dim=-1) if gt.ndim == 4 else torch.norm(gt[1:, :3, 3], dim=-1)
    metric["cam:translation_scale"] = float((gt_norm / pred_norm).mean().item())
    return metric


def compute_pose_metrics(gt_w2c: np.ndarray, pred_w2c: np.ndarray) -> Dict[str, float]:
    return compute_pose_metrics_with_options(gt_w2c, pred_w2c)


def process_scene(model, scene_dir: Path, args: argparse.Namespace, device, dtype) -> Dict[str, Any]:
    import torch

    scene_inputs = load_scene_inputs(
        scene_dir=scene_dir,
        seed=args.seed,
        pool_size=args.pool_size,
        n_srcs=args.n_srcs,
        use_scene_order=bool(args.use_scene_order),
        camera_cache_root=args.camera_cache_root or None,
    )
    images = load_images_with_retry(
        image_names=scene_inputs["image_paths"],
        new_width=args.load_img_size,
        device=str(device),
        retries=args.image_load_retries,
        retry_sleep=args.image_load_retry_sleep,
    )

    with torch.no_grad():
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=dtype)
            if getattr(device, "type", "") == "cuda"
            else contextlib.nullcontext()
        )
        with amp_ctx:
            predictions = model(images)

    pred_w2c = extract_pred_extrinsics(predictions)
    eval_frame_indices = parse_eval_frame_indices(args.eval_frame_indices)
    gt_w2c_eval = subset_pose_array_for_eval(scene_inputs["gt_w2c"], eval_frame_indices)
    pred_w2c_eval = subset_pose_array_for_eval(pred_w2c, eval_frame_indices)
    metric = compute_pose_metrics_with_options(
        gt_w2c_eval,
        pred_w2c_eval,
        translation_error_mode=args.translation_error_mode,
        auc_combine=args.auc_combine,
    )
    metric.update(
        {
            "path": scene_inputs["scene_key"],
            "camera": scene_inputs["target_name"],
            "frame": scene_inputs["target_name"],
            "sampled_indices": scene_inputs["sampled_indices"],
            "sampled_names": scene_inputs["sampled_names"],
            "eval_frame_indices": eval_frame_indices,
        }
    )
    return metric


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = resolve_device(args)
    dtype = resolve_autocast_dtype(device)
    model, loaded_ckpt = load_pi3_model_runtime(args.model_path, device)
    scene_roots = resolve_scene_roots(
        re10k_root=Path(args.re10k_root),
        scene_filter=args.scene_filter,
        limit_scenes=args.limit_scenes,
    )
    metrics = []
    failures = []
    for idx, scene_dir in enumerate(scene_roots, start=1):
        print(f"[scene {idx}/{len(scene_roots)}] {scene_dir.name}", flush=True)
        try:
            metric = process_scene(
                model=model,
                scene_dir=scene_dir,
                args=args,
                device=device,
                dtype=dtype,
            )
        except Exception as exc:
            failures.append({"path": scene_key_from_dir(scene_dir), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[scene-fail] {scene_dir.name}: {type(exc).__name__}: {exc}", flush=True)
            continue

        metrics.append(metric)
        print(
            f"[scene-done] {metric['path']} "
            f"AUC30={metric['cam:pose_auc_30']:.6f} "
            f"AUC20={metric['cam:pose_auc_20']:.6f} "
            f"AUC10={metric['cam:pose_auc_10']:.6f}",
            flush=True,
        )

    summary = build_summary(metrics)
    result = {
        "implementation": "pi3_aligned_re10k",
        "model_tag": args.model_tag,
        "model_path": args.model_path or default_model_path(),
        "loaded_model": loaded_ckpt,
        "re10k_root": str(args.re10k_root),
        "seed": int(args.seed),
        "pool_size": int(args.pool_size),
        "n_srcs": int(args.n_srcs),
        "use_scene_order": bool(args.use_scene_order),
        "eval_frame_indices": parse_eval_frame_indices(args.eval_frame_indices),
        "translation_error_mode": str(args.translation_error_mode),
        "auc_combine": str(args.auc_combine),
        "load_img_size": int(args.load_img_size),
        "limit_scenes": int(args.limit_scenes),
        "scene_filter": str(args.scene_filter),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "summary": summary,
        "metrics": metrics,
        "failures": failures,
    }
    return result


def main():
    args = setup_args()
    result = run(args)
    output_path = Path(args.output) if args.output else default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    summary = result["summary"]
    print(
        f"[done] scenes={summary['metrics_count']} "
        f"AUC30={summary['cam:pose_auc_30_mean']:.6f} "
        f"AUC20={summary['cam:pose_auc_20_mean']:.6f} "
        f"AUC10={summary['cam:pose_auc_10_mean']:.6f} "
        f"-> {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
