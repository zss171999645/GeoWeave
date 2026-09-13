#!/usr/bin/env python3
"""Lightweight VGGT RE10K pose evaluator aligned to the current paper10 protocol."""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


DEFAULT_OFFICIAL_CKPT_ROOT = (
    "/horizon-bucket/saturn_v_dev/01_users/tao02.xie/projects/GFM/gfm/data/trained_model/vggt/official/VGGT-1B"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_repo_root_on_syspath() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


ensure_repo_root_on_syspath()


from aidi.scripts.baselines.eval_pi3_re10k_pose_official import (
    DEFAULT_LOAD_IMG_SIZE,
    DEFAULT_N_SRCS,
    DEFAULT_POOL_SIZE,
    DEFAULT_RE10K_ROOT,
    DEFAULT_SEED,
    build_summary,
    load_scene_inputs,
    resolve_scene_roots,
    scene_key_from_dir,
)


def setup_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight VGGT evaluator for RE10K paper10 pose metrics.")
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
    parser.add_argument("--image-mode", default="crop", choices=["crop", "pad"])
    parser.add_argument("--image-load-retries", type=int, default=8)
    parser.add_argument("--image-load-retry-sleep", type=float, default=0.5)
    parser.add_argument(
        "--camera-cache-root",
        default="",
        help="Optional camera-yaml cache root. Defaults to a process-specific /tmp path.",
    )
    parser.add_argument("--model-path", default="", help="Optional full checkpoint path.")
    parser.add_argument("--config", default="", help="Training config path when --model-path is used.")
    parser.add_argument("--official-ckpt-root", default=DEFAULT_OFFICIAL_CKPT_ROOT)
    parser.add_argument("--device", default="")
    parser.add_argument("--devices", default="", help="Comma-separated device list for multi-GPU sharded evaluation.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--worker-name", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--model-tag", default="vggt_official")
    return parser.parse_args(list(argv) if argv is not None else None)


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

    pred_norm = (
        torch.norm(pred[:, 1:, :3, 3], dim=-1).clamp(min=1e-12)
        if pred.ndim == 4
        else torch.norm(pred[1:, :3, 3], dim=-1).clamp(min=1e-12)
    )
    gt_norm = torch.norm(gt[:, 1:, :3, 3], dim=-1) if gt.ndim == 4 else torch.norm(gt[1:, :3, 3], dim=-1)
    metric["cam:translation_scale"] = float((gt_norm / pred_norm).mean().item())
    return metric


def resolve_camera_cache_root(args: argparse.Namespace, pid: int | None = None) -> Path:
    if str(args.camera_cache_root).strip():
        return Path(str(args.camera_cache_root)).expanduser()
    process_id = int(os.getpid() if pid is None else pid)
    return Path("/tmp") / f"re10k_vggt_cam_cache_{process_id}"


def default_output_path(args: argparse.Namespace) -> Path:
    return Path("/tmp") / f"re10k_vggt_lightweight_{args.model_tag}_seed{args.seed}.json"


def resolve_device(args: argparse.Namespace):
    import torch

    return torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))


def parse_devices(text: str) -> List[str]:
    devices: List[str] = []
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            devices.append(f"cuda:{token}")
        else:
            devices.append(token)
    return devices


def shard_scene_roots(scene_roots: Sequence[Path], num_shards: int, shard_index: int) -> List[Path]:
    if num_shards <= 1:
        return list(scene_roots)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"Invalid shard_index={shard_index} for num_shards={num_shards}")
    return [scene_dir for idx, scene_dir in enumerate(scene_roots) if idx % num_shards == shard_index]


def should_launch_multi_gpu(args: argparse.Namespace) -> bool:
    return len(parse_devices(args.devices)) > 1 and int(args.num_shards) == 1 and int(args.shard_index) == 0


def resolve_autocast_dtype(device):
    import torch

    if getattr(device, "type", "") != "cuda":
        return torch.float32
    major = torch.cuda.get_device_capability(device=device)[0]
    return torch.bfloat16 if major >= 8 else torch.float16


def shard_output_path(output_path: Path, shard_index: int, num_shards: int) -> Path:
    stem = output_path.stem
    suffix = output_path.suffix or ".json"
    return output_path.with_name(f"{stem}.shard{shard_index:02d}of{num_shards:02d}{suffix}")


def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return dict(state_dict)


def remap_special_token_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    state_dict = dict(state_dict)

    def _remap_suffix(old_suffix: str, new_suffix: str) -> None:
        for key in list(state_dict.keys()):
            if not key.endswith(old_suffix):
                continue
            if key.endswith(new_suffix):
                continue
            new_key = key[: -len(old_suffix)] + new_suffix
            if new_key in state_dict:
                continue
            state_dict[new_key] = state_dict.pop(key)

    _remap_suffix("camera_token", "special_tokens.camera_token")
    _remap_suffix("register_token", "special_tokens.register_token")
    _remap_suffix("patch_embed.cls_token", "patch_embed.special_tokens.cls_token")
    _remap_suffix("patch_embed.pos_embed", "patch_embed.special_tokens.pos_embed")
    _remap_suffix("patch_embed.register_tokens", "patch_embed.special_tokens.register_tokens")
    _remap_suffix("patch_embed.mask_token", "patch_embed.special_tokens.mask_token")
    return state_dict


def remap_sampler_module_keys(state_dict: Dict[str, Any], keep_vggt_prefix: bool = False) -> Dict[str, Any]:
    remapped = {}
    prefix_map = {
        "sampler.agg_regator.": "aggregator.",
        "sampler.cam_decoder.": "camera_head.",
        "sampler.xyz_decoder.": "point_head.",
        "sampler.dpt_decoder.": "depth_head.",
        "sampler.tra_decoder.": "track_head.",
    }
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in prefix_map.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        if keep_vggt_prefix and any(new_key.startswith(prefix) for prefix in prefix_map.values()):
            new_key = f"vggt.{new_key}"
        remapped[new_key] = value
    return remapped


def normalize_state_dict_keys(state_dict: Dict[str, Any], keep_vggt_prefix: bool = False) -> Dict[str, Any]:
    normalized = strip_module_prefix(state_dict)
    normalized = remap_sampler_module_keys(normalized, keep_vggt_prefix=keep_vggt_prefix)
    normalized = remap_special_token_keys(normalized)
    if normalized and all(key.startswith("vggt.") for key in normalized):
        if keep_vggt_prefix:
            return normalized
        normalized = {key[len("vggt."):]: value for key, value in normalized.items()}
    elif keep_vggt_prefix and normalized:
        bare_prefixes = ("aggregator.", "camera_head.", "point_head.", "depth_head.", "track_head.")
        if any(key.startswith(bare_prefixes) for key in normalized):
            normalized = {f"vggt.{key}": value for key, value in normalized.items()}
    return normalized


def load_module_ckpt(module, ckpt_path: Path, name: str) -> None:
    import torch

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = state.get("model", state)
    state_dict = strip_module_prefix(state_dict)
    if name == "aggregator":
        state_dict = remap_special_token_keys(state_dict)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Error(s) in loading state_dict for {name}: missing={missing}, unexpected={unexpected}")


def build_official_model_cfg_from_config(config_path: str):
    from easyvolcap.engine.config import Config
    from easyvolcap.official_vggt.models.vggt import VGGT
    from easyvolcap.utils.base_utils import dotdict

    cfg = Config.fromfile(config_path)
    raw_model_cfg = dotdict(cfg.model_cfg)
    raw_model_cfg.pop("_delete_", None)
    raw_type = raw_model_cfg.pop("type", "")

    if raw_type == "OfficialVGGTModel" or "vggt_cfg" in raw_model_cfg:
        official_model_cfg = dotdict(raw_model_cfg)
    else:
        official_template = Config.fromfile(str(repo_root() / "configs/models/vggt_official.yaml"))
        official_model_cfg = dotdict(official_template.model_cfg)
        official_model_cfg.pop("_delete_", None)
        official_model_cfg.pop("type", None)

        sampler_cfg = dotdict(raw_model_cfg.get("sampler_cfg", {}))
        accepted_vggt_keys = {key for key in inspect.signature(VGGT.__init__).parameters if key != "self"}
        vggt_cfg = dotdict(official_model_cfg.get("vggt_cfg", {}))

        for key, value in sampler_cfg.items():
            if key in accepted_vggt_keys:
                vggt_cfg[key] = value

        if "indexer_cfg" in sampler_cfg:
            vggt_cfg["indexer_cfg"] = sampler_cfg["indexer_cfg"]

        if "aggregator_cfg" in sampler_cfg:
            vggt_cfg["aggregator_cfg"] = sampler_cfg["aggregator_cfg"]

        use_chunkwise_bp = bool(sampler_cfg.get("use_chunkwise_bp_dpt_decoder", False))
        if use_chunkwise_bp:
            depth_head_cfg = dotdict(official_model_cfg.get("depth_head_cfg", {}))
            point_head_cfg = dotdict(official_model_cfg.get("point_head_cfg", {}))
            depth_head_cfg["type"] = "DPTHeadWithChunkwiseBP"
            point_head_cfg["type"] = "DPTHeadWithChunkwiseBP"
            official_model_cfg["depth_head_cfg"] = depth_head_cfg
            official_model_cfg["point_head_cfg"] = point_head_cfg

        official_model_cfg["vggt_cfg"] = vggt_cfg

    official_model_cfg["pretrained_path"] = ""
    for key in ("agg_ckpt", "cam_ckpt", "xyz_ckpt", "dpt_ckpt", "tra_ckpt"):
        official_model_cfg[key] = ""
    return official_model_cfg


def build_model_from_training_checkpoint(device, checkpoint: dict, config_path: str):
    from easyvolcap.models.official_vggt_model import OfficialVGGTModel

    if not config_path:
        raise RuntimeError(
            "Training checkpoint bridge requires --config /path/to/your_train_config.yaml "
            "so OfficialVGGTModel can be built with the matching architecture."
        )
    model_cfg = build_official_model_cfg_from_config(config_path)

    wrapped = OfficialVGGTModel(**model_cfg).to(device=device).eval()
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = normalize_state_dict_keys(state_dict, keep_vggt_prefix=True)
    missing, unexpected = wrapped.load_state_dict(state_dict, strict=False)
    missing = [key for key in missing if "indexer" not in key]
    unexpected = [key for key in unexpected if "indexer" not in key]
    if missing or unexpected:
        print(
            "[checkpoint-bridge] OfficialVGGTModel load mismatches "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    return wrapped.vggt


def config_enables_sparse_indexer(config_path: str) -> bool:
    if not config_path:
        return False
    try:
        from easyvolcap.engine.config import Config

        cfg = Config.fromfile(config_path)
        model_cfg = cfg.get("model_cfg", {})
        vggt_cfg = model_cfg.get("vggt_cfg", {})
        indexer_cfg = vggt_cfg.get("indexer_cfg", {})
        return bool(indexer_cfg.get("enable_sparse", False))
    except Exception as exc:
        print(f"[checkpoint-bridge] cannot inspect config sparse flag: {exc}", flush=True)
        return False


def load_model(device, model_path: str, official_ckpt_root: str, config_path: str):
    import torch
    from easyvolcap.official_vggt.models.vggt import VGGT

    print("[re10k-lightweight] Initializing VGGT model...", flush=True)
    model = VGGT()
    loaded_ref = model_path or official_ckpt_root
    if model_path:
        print(f"[re10k-lightweight] USING FULL CHECKPOINT {model_path}", flush=True)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if config_enables_sparse_indexer(config_path):
            print(
                "[checkpoint-bridge] config enables sparse indexer; "
                f"using OfficialVGGTModel config bridge (config={config_path})",
                flush=True,
            )
            model = build_model_from_training_checkpoint(device, checkpoint, config_path=config_path)
        else:
            state_dict = checkpoint.get("model", checkpoint)
            state_dict = normalize_state_dict_keys(state_dict)
            fallback_reason = None
            try:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                missing = [key for key in missing if "indexer" not in key]
                unexpected = [key for key in unexpected if "indexer" not in key]
                if missing or unexpected:
                    fallback_reason = f"missing={len(missing)} unexpected={len(unexpected)}"
            except RuntimeError as exc:
                fallback_reason = f"runtime_error={exc}"
            if fallback_reason is not None:
                print(
                    "[checkpoint-bridge] direct VGGT load has mismatches, "
                    f"fallback to OfficialVGGTModel config bridge (config={config_path or 'missing'}, reason={fallback_reason})",
                    flush=True,
                )
                model = build_model_from_training_checkpoint(device, checkpoint, config_path=config_path)
    else:
        ckpt_root = Path(official_ckpt_root)
        print(f"[re10k-lightweight] USING SPLIT OFFICIAL CKPTS FROM {ckpt_root}", flush=True)
        load_module_ckpt(model.aggregator, ckpt_root / "aggregator.pt", "aggregator")
        load_module_ckpt(model.camera_head, ckpt_root / "camera.pt", "camera")
        load_module_ckpt(model.point_head, ckpt_root / "point.pt", "point")
        load_module_ckpt(model.depth_head, ckpt_root / "depth.pt", "depth")
        load_module_ckpt(model.track_head, ckpt_root / "track.pt", "track")

    model.eval()
    model = model.to(device)
    return model, loaded_ref


def load_images_with_retry(
    image_names: List[str],
    image_mode: str,
    load_img_size: int,
    device,
    retries: int,
    retry_sleep: float,
):
    from easyvolcap.official_vggt.utils.load_fn import load_and_preprocess_images_evc_compatible

    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            return load_and_preprocess_images_evc_compatible(
                image_names,
                mode=image_mode,
                target_size=load_img_size,
            ).to(device)
        except OSError as exc:
            if attempt >= attempts:
                raise
            print(
                f"[image-retry] attempt={attempt}/{attempts} "
                f"sleep={retry_sleep:.2f}s error={type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(max(0.0, float(retry_sleep)))


def predict_extrinsics(model, images, device, dtype) -> np.ndarray:
    import torch

    from easyvolcap.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri

    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if getattr(device, "type", "") == "cuda"
        else contextlib.nullcontext()
    )
    with torch.no_grad():
        with amp_ctx:
            predictions = model(images)
    extrinsic, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], image_size_hw=images.shape[-2:])
    return extrinsic[0].detach().float().cpu().numpy()


def process_scene(model, scene_dir: Path, args: argparse.Namespace, device, dtype) -> Dict[str, Any]:
    scene_inputs = load_scene_inputs(
        scene_dir=scene_dir,
        seed=args.seed,
        pool_size=args.pool_size,
        n_srcs=args.n_srcs,
        use_scene_order=bool(args.use_scene_order),
        camera_cache_root=resolve_camera_cache_root(args),
    )
    images = load_images_with_retry(
        image_names=scene_inputs["image_paths"],
        image_mode=args.image_mode,
        load_img_size=args.load_img_size,
        device=device,
        retries=args.image_load_retries,
        retry_sleep=args.image_load_retry_sleep,
    )
    pred_w2c = predict_extrinsics(model=model, images=images, device=device, dtype=dtype)
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


def build_result_payload(
    args: argparse.Namespace,
    loaded_model: str,
    device: Any,
    dtype: Any,
    metrics: Sequence[Dict[str, Any]],
    failures: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "implementation": "vggt_re10k_lightweight",
        "model_tag": args.model_tag,
        "model_path": args.model_path,
        "config": args.config,
        "official_ckpt_root": args.official_ckpt_root,
        "loaded_model": loaded_model,
        "re10k_root": str(args.re10k_root),
        "seed": int(args.seed),
        "pool_size": int(args.pool_size),
        "n_srcs": int(args.n_srcs),
        "use_scene_order": bool(args.use_scene_order),
        "eval_frame_indices": parse_eval_frame_indices(args.eval_frame_indices),
        "translation_error_mode": str(args.translation_error_mode),
        "auc_combine": str(args.auc_combine),
        "load_img_size": int(args.load_img_size),
        "image_mode": str(args.image_mode),
        "camera_cache_root": str(resolve_camera_cache_root(args)),
        "devices": parse_devices(args.devices),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "worker_name": str(args.worker_name),
        "limit_scenes": int(args.limit_scenes),
        "scene_filter": str(args.scene_filter),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "summary": build_summary(metrics),
        "metrics": list(metrics),
        "failures": list(failures),
    }


def run_single_process(args: argparse.Namespace) -> Dict[str, Any]:
    device = resolve_device(args)
    dtype = resolve_autocast_dtype(device)
    model, loaded_model = load_model(
        device=device,
        model_path=args.model_path,
        official_ckpt_root=args.official_ckpt_root,
        config_path=args.config,
    )
    scene_roots = resolve_scene_roots(
        re10k_root=Path(args.re10k_root),
        scene_filter=args.scene_filter,
        limit_scenes=args.limit_scenes,
    )
    scene_roots = shard_scene_roots(scene_roots, int(args.num_shards), int(args.shard_index))
    if int(args.num_shards) > 1:
        worker_label = args.worker_name or f"shard{int(args.shard_index):02d}/{int(args.num_shards):02d}"
        print(
            f"[re10k-lightweight] worker={worker_label} scenes={len(scene_roots)} "
            f"device={device}",
            flush=True,
        )
    metrics: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for idx, scene_dir in enumerate(scene_roots, start=1):
        print(f"[scene {idx}/{len(scene_roots)}] {scene_dir.name}", flush=True)
        try:
            metric = process_scene(model=model, scene_dir=scene_dir, args=args, device=device, dtype=dtype)
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
    return build_result_payload(
        args=args,
        loaded_model=str(loaded_model),
        device=device,
        dtype=dtype,
        metrics=metrics,
        failures=failures,
    )


def build_worker_command(
    args: argparse.Namespace,
    *,
    device: str,
    shard_index: int,
    num_shards: int,
    output_path: Path,
) -> List[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--re10k-root",
        str(args.re10k_root),
        "--seed",
        str(args.seed),
        "--pool-size",
        str(args.pool_size),
        "--n-srcs",
        str(args.n_srcs),
        "--eval-frame-indices",
        str(args.eval_frame_indices),
        "--translation-error-mode",
        str(args.translation_error_mode),
        "--auc-combine",
        str(args.auc_combine),
        "--limit-scenes",
        str(args.limit_scenes),
        "--scene-filter",
        str(args.scene_filter),
        "--load-img-size",
        str(args.load_img_size),
        "--image-mode",
        str(args.image_mode),
        "--image-load-retries",
        str(args.image_load_retries),
        "--image-load-retry-sleep",
        str(args.image_load_retry_sleep),
        "--camera-cache-root",
        str(args.camera_cache_root),
        "--device",
        str(device),
        "--num-shards",
        str(num_shards),
        "--shard-index",
        str(shard_index),
        "--worker-name",
        f"shard{shard_index:02d}",
        "--output",
        str(output_path),
        "--model-tag",
        str(args.model_tag),
    ]
    if args.use_scene_order:
        command.append("--use-scene-order")
    if args.model_path:
        command.extend(["--model-path", str(args.model_path)])
    if args.config:
        command.extend(["--config", str(args.config)])
    if args.official_ckpt_root:
        command.extend(["--official-ckpt-root", str(args.official_ckpt_root)])
    return command


def build_worker_launch_spec(device: str, base_env: Dict[str, str] | None = None) -> tuple[Dict[str, str], str]:
    env = dict(base_env or {})
    normalized_device = str(device).strip()
    if normalized_device.startswith("cuda:"):
        physical_index = normalized_device.split(":", 1)[1]
        env["CUDA_VISIBLE_DEVICES"] = physical_index
        return env, "cuda:0"
    return env, normalized_device


def merge_worker_payloads(args: argparse.Namespace, payloads: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metrics: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    devices: List[str] = []
    loaded_model = ""
    dtype = ""
    for payload in payloads:
        metrics.extend(payload.get("metrics", []))
        failures.extend(payload.get("failures", []))
        loaded_model = loaded_model or str(payload.get("loaded_model", ""))
        dtype = dtype or str(payload.get("dtype", ""))
        for device in payload.get("devices", []) or [payload.get("device", "")]:
            device = str(device).strip()
            if device and device not in devices:
                devices.append(device)
    merged = build_result_payload(
        args=args,
        loaded_model=loaded_model,
        device=",".join(devices),
        dtype=dtype or "float32",
        metrics=metrics,
        failures=failures,
    )
    merged["devices"] = devices
    merged["worker_payload_count"] = int(len(payloads))
    return merged


def run_multi_gpu(args: argparse.Namespace) -> Dict[str, Any]:
    devices = parse_devices(args.devices)
    if len(devices) <= 1:
        raise ValueError(f"run_multi_gpu requires 2+ devices, got {devices}")
    output_path = Path(args.output) if args.output else default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_shards = len(devices)
    worker_outputs = [shard_output_path(output_path, shard_index=i, num_shards=num_shards) for i in range(num_shards)]
    processes = []
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    for shard_index, (device, worker_output) in enumerate(zip(devices, worker_outputs)):
        worker_output.parent.mkdir(parents=True, exist_ok=True)
        worker_env, worker_device = build_worker_launch_spec(device, env)
        command = build_worker_command(
            args,
            device=worker_device,
            shard_index=shard_index,
            num_shards=num_shards,
            output_path=worker_output,
        )
        print(f"[re10k-lightweight] launch shard={shard_index}/{num_shards} device={device}", flush=True)
        processes.append(
            (shard_index, device, worker_output, subprocess.Popen(command, env=worker_env, cwd=repo_root()))
        )

    failures = []
    for shard_index, device, worker_output, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append((shard_index, device, return_code, worker_output))
    if failures:
        details = ", ".join(
            f"shard={shard_index} device={device} rc={return_code} output={worker_output}"
            for shard_index, device, return_code, worker_output in failures
        )
        raise RuntimeError(f"Multi-GPU RE10K lightweight evaluation failed: {details}")

    payloads = []
    for worker_output in worker_outputs:
        with worker_output.open("r", encoding="utf-8") as f:
            payloads.append(json.load(f))
    return merge_worker_payloads(args, payloads)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if should_launch_multi_gpu(args):
        return run_multi_gpu(args)
    return run_single_process(args)


def main(argv: Sequence[str] | None = None) -> int:
    args = setup_args(argv)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
