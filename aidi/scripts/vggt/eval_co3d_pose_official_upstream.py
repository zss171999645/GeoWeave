#!/usr/bin/env python3
"""Vendored CO3D evaluator aligned to facebookresearch/vggt evaluation/test_co3d.py."""

from __future__ import annotations

import argparse
import gzip
import inspect
import json
import logging
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from easyvolcap.engine.config import Config
from easyvolcap.models.official_vggt_model import OfficialVGGTModel
from easyvolcap.official_vggt.models.vggt import VGGT
from easyvolcap.official_vggt.utils.geometry import closed_form_inverse_se3
from easyvolcap.official_vggt.utils.load_fn import load_and_preprocess_images
from easyvolcap.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
from easyvolcap.official_vggt.utils.rotation import mat_to_quat
from easyvolcap.utils.base_utils import dotdict


logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="dinov2")

torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False

SEEN_CATEGORIES: Sequence[str] = (
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
)


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vendored upstream VGGT CO3D evaluator")
    parser.add_argument("--debug", action="store_true", help="Upstream compatibility flag.")
    parser.add_argument("--debug-category", default="", help="Single category shortcut for smoke tests.")
    parser.add_argument("--categories", default="", help="Comma-separated categories to evaluate.")
    parser.add_argument("--fast-eval", action="store_true", default=False, help="Only evaluate 10 sequences/category.")
    parser.add_argument("--min-num-images", type=int, default=50)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--co3d-dir", "--co3d-image-root", dest="co3d_image_root", required=True)
    parser.add_argument("--co3d-anno-dir", type=str, required=True)
    parser.add_argument("--model-path", default="", help="Optional full-model checkpoint path.")
    parser.add_argument(
        "--config",
        default="",
        help="Model config used when loading a training-style full checkpoint such as 34.pt.",
    )
    parser.add_argument(
        "--official-ckpt-root",
        default="/horizon-bucket/saturn_v_dev/01_users/tao02.xie/projects/GFM/gfm/data/trained_model/vggt/official/VGGT-1B",
        help="Root containing split official ckpts when --model-path is empty.",
    )
    parser.add_argument("--device", default="", help="Torch device, default cuda if available.")
    parser.add_argument("--image-mode", default="crop", choices=["crop", "pad"])
    parser.add_argument("--load-img-size", type=int, default=518)
    parser.add_argument(
        "--anno-camera-convention",
        default="pt3d_to_opencv",
        help="Ignored here; vendored upstream evaluator assumes preprocess_co3d.py-style PT3D annotations.",
    )
    parser.add_argument("--image-load-retries", type=int, default=8)
    parser.add_argument("--image-load-retry-sleep", type=float, default=0.5)
    parser.add_argument("--model-tag", default="official")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def convert_pt3d_RT_to_opencv(Rot, Trans):
    rot_pt3d = np.array(Rot)
    trans_pt3d = np.array(Trans)

    trans_pt3d[:2] *= -1
    rot_pt3d[:, :2] *= -1
    rot_pt3d = rot_pt3d.transpose(1, 0)
    extri_opencv = np.hstack((rot_pt3d, trans_pt3d[:, None]))
    return extri_opencv


def build_pair_index(N, B=1):
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
    return i1, i2


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)
    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def calculate_auc_np(r_error, t_error, max_threshold=30):
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

    relative_pose_gt = gt_se3[pair_idx_i1].bmm(closed_form_inverse_se3(gt_se3[pair_idx_i2]))
    relative_pose_pred = pred_se3[pair_idx_i1].bmm(closed_form_inverse_se3(pred_se3[pair_idx_i2]))

    rel_rangle_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])
    rel_tangle_deg = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3])

    return rel_rangle_deg, rel_tangle_deg


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return dict(state_dict)


def remap_special_token_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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


def remap_sampler_module_keys(state_dict: Dict[str, torch.Tensor], keep_vggt_prefix: bool = False) -> Dict[str, torch.Tensor]:
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


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor], keep_vggt_prefix: bool = False) -> Dict[str, torch.Tensor]:
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


def load_module_ckpt(module: torch.nn.Module, ckpt_path: Path, name: str) -> None:
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = state.get("model", state)
    state_dict = strip_module_prefix(state_dict)
    if name == "aggregator":
        state_dict = remap_special_token_keys(state_dict)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Error(s) in loading state_dict for {name}: missing={missing}, unexpected={unexpected}")


def build_official_model_cfg_from_config(config_path: str) -> dotdict:
    cfg = Config.fromfile(config_path)
    raw_model_cfg = dotdict(cfg.model_cfg)
    raw_model_cfg.pop("_delete_", None)
    raw_type = raw_model_cfg.pop("type", "")

    if raw_type == "OfficialVGGTModel" or "vggt_cfg" in raw_model_cfg:
        official_model_cfg = dotdict(raw_model_cfg)
    else:
        repo_root = Path(__file__).resolve().parents[3]
        official_template = Config.fromfile(str(repo_root / "configs/models/vggt_official.yaml"))
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
            f"missing={len(missing)} unexpected={len(unexpected)}"
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
        print(f"[checkpoint-bridge] cannot inspect config sparse flag: {exc}")
        return False


def load_model(device, model_path, official_ckpt_root, config_path):
    print("Initializing and loading VGGT model...")
    model = VGGT()
    if model_path:
        print(f"USING FULL CHECKPOINT {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if config_enables_sparse_indexer(config_path):
            print(
                "[checkpoint-bridge] config enables sparse indexer; "
                f"using OfficialVGGTModel config bridge (config={config_path})"
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
                    f"fallback to OfficialVGGTModel config bridge (config={config_path or 'missing'}, reason={fallback_reason})"
                )
                model = build_model_from_training_checkpoint(device, checkpoint, config_path=config_path)
    else:
        ckpt_root = Path(official_ckpt_root)
        print(f"USING SPLIT OFFICIAL CKPTS FROM {ckpt_root}")
        load_module_ckpt(model.aggregator, ckpt_root / "aggregator.pt", "aggregator")
        load_module_ckpt(model.camera_head, ckpt_root / "camera.pt", "camera")
        load_module_ckpt(model.point_head, ckpt_root / "point.pt", "point")
        load_module_ckpt(model.depth_head, ckpt_root / "depth.pt", "depth")
        load_module_ckpt(model.track_head, ckpt_root / "track.pt", "track")
    model.eval()
    model = model.to(device)
    return model


def load_and_preprocess_images_with_retry(image_names, mode, target_size, retries, retry_sleep):
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            return load_and_preprocess_images(image_names, mode=mode, target_size=target_size)
        except OSError as exc:
            if attempt >= attempts:
                raise
            print(
                f"[image-retry] attempt={attempt}/{attempts} "
                f"sleep={retry_sleep:.2f}s error={type(exc).__name__}: {exc}"
            )
            time.sleep(max(0.0, float(retry_sleep)))


def parse_category_list(args: argparse.Namespace) -> List[str]:
    if args.debug_category:
        return [args.debug_category]
    if args.categories:
        return [x.strip() for x in args.categories.split(",") if x.strip()]
    if args.debug:
        return ["parkingmeter"]
    return list(SEEN_CATEGORIES)


def load_annotation(path: Path) -> Dict[str, List[dict]]:
    with gzip.open(path, "rb") as f:
        return json.loads(f.read())


def process_sequence(
    model,
    seq_name,
    seq_data,
    category,
    co3d_dir,
    min_num_images,
    num_frames,
    device,
    dtype,
    image_mode,
    load_img_size,
    image_load_retries,
    image_load_retry_sleep,
):
    if len(seq_data) < min_num_images:
        return None

    metadata = []
    for data in seq_data:
        if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
            return None
        extri_opencv = convert_pt3d_RT_to_opencv(data["R"], data["T"])
        metadata.append({"filepath": data["filepath"], "extri": extri_opencv})

    ids = np.random.choice(len(metadata), num_frames, replace=False)
    print("Image ids", ids)

    image_names = [os.path.join(co3d_dir, metadata[i]["filepath"]) for i in ids]
    gt_extri = [np.array(metadata[i]["extri"]) for i in ids]
    gt_extri = np.stack(gt_extri, axis=0)

    images = load_and_preprocess_images_with_retry(
        image_names,
        mode=image_mode,
        target_size=load_img_size,
        retries=image_load_retries,
        retry_sleep=image_load_retry_sleep,
    ).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    with torch.cuda.amp.autocast(dtype=torch.float64):
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        del intrinsic
        pred_extrinsic = extrinsic[0]

    with torch.cuda.amp.autocast(dtype=torch.float64):
        gt_extrinsic = torch.from_numpy(gt_extri).to(device)
        add_row = torch.tensor([0, 0, 0, 1], device=device).expand(pred_extrinsic.size(0), 1, 4)

        pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
        gt_se3 = torch.cat((gt_extrinsic, add_row), dim=1)

        rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, num_frames)

        Racc_5 = (rel_rangle_deg < 5).float().mean().item()
        Tacc_5 = (rel_tangle_deg < 5).float().mean().item()

        print(f"{category} sequence {seq_name} R_ACC@5: {Racc_5:.4f}")
        print(f"{category} sequence {seq_name} T_ACC@5: {Tacc_5:.4f}")

        r_error = rel_rangle_deg.cpu().numpy()
        t_error = rel_tangle_deg.cpu().numpy()

    auc_30, _ = calculate_auc_np(r_error, t_error, max_threshold=30)
    auc_20, _ = calculate_auc_np(r_error, t_error, max_threshold=20)
    auc_15, _ = calculate_auc_np(r_error, t_error, max_threshold=15)
    auc_10, _ = calculate_auc_np(r_error, t_error, max_threshold=10)
    auc_5, _ = calculate_auc_np(r_error, t_error, max_threshold=5)
    auc_3, _ = calculate_auc_np(r_error, t_error, max_threshold=3)

    return {
        "category": category,
        "sequence_name": seq_name,
        "path": f"{category}/{seq_name}",
        "num_images_total": int(len(seq_data)),
        "num_frames_eval": int(num_frames),
        "sampled_ids": ids.astype(int).tolist(),
        "sampled_filepaths": [metadata[i]["filepath"] for i in ids],
        "R_ACC_5": float(Racc_5),
        "T_ACC_5": float(Tacc_5),
        "AUC_30": float(auc_30),
        "AUC_20": float(auc_20),
        "AUC_15": float(auc_15),
        "AUC_10": float(auc_10),
        "AUC_5": float(auc_5),
        "AUC_3": float(auc_3),
        "cam:pose_auc_10": float(auc_10),
        "cam:pose_auc_20": float(auc_20),
        "cam:pose_auc_30": float(auc_30),
        "r_error": r_error.tolist(),
        "t_error": t_error.tolist(),
    }


def summarize_category(sequence_results: Sequence[dict]) -> dict:
    rError = np.array([x for result in sequence_results for x in result["r_error"]], dtype=np.float64)
    tError = np.array([x for result in sequence_results for x in result["t_error"]], dtype=np.float64)

    Auc_30, _ = calculate_auc_np(rError, tError, max_threshold=30)
    Auc_15, _ = calculate_auc_np(rError, tError, max_threshold=15)
    Auc_5, _ = calculate_auc_np(rError, tError, max_threshold=5)
    Auc_3, _ = calculate_auc_np(rError, tError, max_threshold=3)
    Auc_20, _ = calculate_auc_np(rError, tError, max_threshold=20)
    Auc_10, _ = calculate_auc_np(rError, tError, max_threshold=10)

    return {
        "num_sequences": len(sequence_results),
        "num_pairs_total": int(len(rError)),
        "AUC_30": float(Auc_30),
        "AUC_20": float(Auc_20),
        "AUC_15": float(Auc_15),
        "AUC_10": float(Auc_10),
        "AUC_5": float(Auc_5),
        "AUC_3": float(Auc_3),
        "cam:pose_auc_10_mean": float(Auc_10),
        "cam:pose_auc_20_mean": float(Auc_20),
        "cam:pose_auc_30_mean": float(Auc_30),
    }


def mean_metric(results: Dict[str, dict], key: str) -> float | None:
    values = [float(result[key]) for result in results.values() if key in result]
    if not values:
        return None
    return float(np.mean(values))


def default_output_path(args: argparse.Namespace) -> Path:
    base = f"co3d_official_upstream_{args.model_tag}_seed{args.seed}_frames{args.num_frames}.json"
    return Path("/tmp") / base


def main():
    args = setup_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = load_model(
        device,
        model_path=args.model_path,
        official_ckpt_root=args.official_ckpt_root,
        config_path=args.config,
    )
    set_random_seeds(args.seed)

    categories = parse_category_list(args)
    per_category_results = {}
    per_sequence_results = {}

    for category in categories:
        print(f"Loading annotation for {category} test set")
        annotation_file = os.path.join(args.co3d_anno_dir, f"{category}_test.jgz")

        try:
            annotation = load_annotation(Path(annotation_file))
        except FileNotFoundError:
            print(f"Annotation file not found for {category}, skipping")
            continue

        sequence_results = []

        seq_names = sorted(list(annotation.keys()))
        if args.fast_eval and len(seq_names) >= 10:
            seq_names = random.sample(seq_names, 10)
        seq_names = sorted(seq_names)

        print("Testing Sequences: ")
        print(seq_names)

        for seq_name in seq_names:
            seq_data = annotation[seq_name]
            print("-" * 50)
            print(f"Processing {seq_name} for {category} test set")
            if (args.debug or args.debug_category) and not os.path.exists(os.path.join(args.co3d_image_root, category, seq_name)):
                print(f"Skipping {seq_name} (not found)")
                continue

            seq_result = process_sequence(
                model,
                seq_name,
                seq_data,
                category,
                args.co3d_image_root,
                args.min_num_images,
                args.num_frames,
                device,
                dtype,
                args.image_mode,
                args.load_img_size,
                args.image_load_retries,
                args.image_load_retry_sleep,
            )

            print("-" * 50)

            if seq_result is not None:
                sequence_results.append(seq_result)
                print(
                    f"[sequence] {category}/{seq_name} "
                    f"AUC30={seq_result['AUC_30']:.6f} "
                    f"R_ACC5={seq_result['R_ACC_5']:.6f} "
                    f"T_ACC5={seq_result['T_ACC_5']:.6f}"
                )

        if not sequence_results:
            print(f"No valid sequences found for {category}, skipping")
            continue

        per_sequence_results[category] = sequence_results
        per_category_results[category] = summarize_category(sequence_results)

        summary = per_category_results[category]
        print("=" * 80)
        print(
            f"AUC of {category} test set: "
            f"{summary['AUC_30']:.4f} (AUC@30), "
            f"{summary['AUC_15']:.4f} (AUC@15), "
            f"{summary['AUC_5']:.4f} (AUC@5), "
            f"{summary['AUC_3']:.4f} (AUC@3)"
        )
        mean_AUC_30_by_now = np.mean([per_category_results[c]["AUC_30"] for c in per_category_results])
        mean_AUC_15_by_now = np.mean([per_category_results[c]["AUC_15"] for c in per_category_results])
        mean_AUC_5_by_now = np.mean([per_category_results[c]["AUC_5"] for c in per_category_results])
        mean_AUC_3_by_now = np.mean([per_category_results[c]["AUC_3"] for c in per_category_results])
        print(
            f"Mean AUC of categories by now: "
            f"{mean_AUC_30_by_now:.4f} (AUC@30), "
            f"{mean_AUC_15_by_now:.4f} (AUC@15), "
            f"{mean_AUC_5_by_now:.4f} (AUC@5), "
            f"{mean_AUC_3_by_now:.4f} (AUC@3)"
        )
        print("=" * 80)

    if not per_category_results:
        raise RuntimeError("No category produced valid results.")

    overall_summary = {
        "AUC_30_mean": mean_metric(per_category_results, "AUC_30"),
        "AUC_20_mean": mean_metric(per_category_results, "AUC_20"),
        "AUC_15_mean": mean_metric(per_category_results, "AUC_15"),
        "AUC_10_mean": mean_metric(per_category_results, "AUC_10"),
        "AUC_5_mean": mean_metric(per_category_results, "AUC_5"),
        "AUC_3_mean": mean_metric(per_category_results, "AUC_3"),
        "cam:pose_auc_10_mean": mean_metric(per_category_results, "cam:pose_auc_10_mean"),
        "cam:pose_auc_20_mean": mean_metric(per_category_results, "cam:pose_auc_20_mean"),
        "cam:pose_auc_30_mean": mean_metric(per_category_results, "cam:pose_auc_30_mean"),
        "num_categories": len(per_category_results),
        "num_sequences": int(sum(len(results) for results in per_sequence_results.values())),
    }

    metrics = [item for results in per_sequence_results.values() for item in results]
    result = {
        "implementation": "vendored_upstream_test_co3d",
        "model_tag": args.model_tag,
        "model_path": args.model_path,
        "official_ckpt_root": args.official_ckpt_root,
        "co3d_dir": str(args.co3d_image_root),
        "co3d_anno_dir": str(args.co3d_anno_dir),
        "categories": categories,
        "fast_eval": bool(args.fast_eval),
        "min_num_images": int(args.min_num_images),
        "num_frames": int(args.num_frames),
        "seed": int(args.seed),
        "image_mode": args.image_mode,
        "load_img_size": int(args.load_img_size),
        "image_load_retries": int(args.image_load_retries),
        "image_load_retry_sleep": float(args.image_load_retry_sleep),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "metrics": metrics,
        "per_category_results": per_category_results,
        "per_sequence_results": per_sequence_results,
        "summary": overall_summary,
    }

    output_path = Path(args.output) if args.output else default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(
        f"[done] AUC30={overall_summary['AUC_30_mean']:.6f} "
        f"categories={overall_summary['num_categories']} "
        f"sequences={overall_summary['num_sequences']} "
        f"-> {output_path}"
    )


if __name__ == "__main__":
    main()
