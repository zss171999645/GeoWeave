#!/usr/bin/env python3
"""Pi3 CO3D pose evaluator under the aligned official CO3D protocol."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_repo_root_on_syspath() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_module_from_file(module_name: str, path: Path):
    ensure_repo_root_on_syspath()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_upstream_module():
    return load_module_from_file(
        "eval_co3d_pose_official_upstream_runtime",
        repo_root() / "aidi" / "scripts" / "vggt" / "eval_co3d_pose_official_upstream.py",
    )


def load_pi3_core_module():
    return load_module_from_file(
        "eval_pi3_mv_recon_core_runtime",
        repo_root() / "aidi" / "scripts" / "baselines" / "eval_pi3_mv_recon_core.py",
    )


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pi3 CO3D evaluator aligned to current official CO3D protocol")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-category", default="")
    parser.add_argument("--categories", default="")
    parser.add_argument("--fast-eval", action="store_true", default=False)
    parser.add_argument("--min-num-images", type=int, default=50)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--co3d-image-root", required=True)
    parser.add_argument("--co3d-anno-dir", required=True)
    parser.add_argument(
        "--model-path",
        default="",
        help="Pi3 safetensors/checkpoint path or HuggingFace id. Empty means auto-discover local default then HF id.",
    )
    parser.add_argument("--device", default="")
    parser.add_argument("--load-img-size", type=int, default=518)
    parser.add_argument("--image-load-retries", type=int, default=8)
    parser.add_argument("--image-load-retry-sleep", type=float, default=0.5)
    parser.add_argument("--model-tag", default="pi3")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def parse_category_list(args: argparse.Namespace) -> List[str]:
    if args.debug_category:
        return [args.debug_category]
    if args.categories:
        return [x.strip() for x in args.categories.split(",") if x.strip()]
    if args.debug:
        return ["parkingmeter"]
    return list(SEEN_CATEGORIES)


def load_annotation(path: Path) -> Dict[str, List[dict]]:
    with gzip.open(path, "rb") as handle:
        return json.loads(handle.read())


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


def mean_metric(results: Dict[str, dict], key: str) -> float | None:
    values = [float(result[key]) for result in results.values() if key in result]
    if not values:
        return None
    return float(np.mean(values))


def build_overall_summary(per_category_results: Dict[str, dict], num_sequences: int) -> dict:
    return {
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
        "num_sequences": int(num_sequences),
    }


def build_scene_pools_from_manifest(manifest: Sequence[dict]) -> Dict[str, dict]:
    pools: Dict[str, dict] = {}
    for item in manifest:
        category = item["category"]
        sequence_name = item["sequence_name"]
        path = f"{category}/{sequence_name}"
        entry = pools.setdefault(path, {"frame_indices": [], "filepaths": []})
        entry["frame_indices"].append(int(item["frame_index"]))
        entry["filepaths"].append(str(item["filepath"]))
    return pools


def maybe_load_scene_pools(image_root: str) -> Dict[str, dict]:
    meta_dir = Path(image_root) / "_meta"
    manifest_path = meta_dir / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r") as handle:
            manifest = json.load(handle)
        if isinstance(manifest, list):
            return build_scene_pools_from_manifest(manifest)
    return {}


def default_model_path() -> str:
    candidates = (
        Path("/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/pretrained/pi3/yyfz233_Pi3_model.safetensors"),
        Path("/horizon-bucket/saturn_v_dev/01_users/horizon/projects/meshx/baseline/pretrained/pi3/yyfz233_Pi3_model.safetensors"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return "yyfz233/Pi3"


def default_output_path(args: argparse.Namespace) -> Path:
    base = f"co3d_pi3_aligned_{args.model_tag}_seed{args.seed}_frames{args.num_frames}.json"
    return Path("/tmp") / base


def resolve_device(args: argparse.Namespace):
    import torch

    return torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))


def resolve_autocast_dtype(device) -> Any:
    import torch

    if getattr(device, "type", "") != "cuda":
        return torch.float32
    major = torch.cuda.get_device_capability(device=device)[0]
    return torch.bfloat16 if major >= 8 else torch.float16


def load_pi3_model_runtime(model_path: str, device):
    import torch
    from easyvolcap.utils.pi3.models.pi3 import Pi3
    from aidi.scripts.baselines.pi3_checkpoint_loader import load_pi3_model_for_eval

    pi3_core = load_pi3_core_module()
    resolved_path = model_path or default_model_path()
    model, loaded_ckpt = load_pi3_model_for_eval(
        resolved_path,
        device,
        official_pi3_cls=Pi3,
        native_root=os.environ.get("PI3_NATIVE_ROOT", ""),
        model_impl=os.environ.get("PI3_MODEL_IMPL", ""),
        config_path=os.environ.get("PI3_CONFIG", ""),
    )
    if isinstance(model, torch.nn.Module):
        model = model.to(device).eval()
    return model, loaded_ckpt, pi3_core


def load_images_with_retry(pi3_core, image_names: List[str], new_width: int, device: str, verbose: bool, retries: int, retry_sleep: float):
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            return pi3_core.load_images_for_pi3(
                filelist=image_names,
                new_width=new_width,
                device=device,
                verbose=verbose,
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


def process_sequence(
    model,
    seq_name: str,
    seq_data: Sequence[dict],
    category: str,
    args: argparse.Namespace,
    device,
    dtype,
    pi3_core,
    upstream,
    scene_pools: Dict[str, dict],
):
    import torch

    if len(seq_data) < args.min_num_images:
        return None

    metadata = []
    for data in seq_data:
        if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
            return None
        extri_opencv = upstream.convert_pt3d_RT_to_opencv(data["R"], data["T"])
        metadata.append({"filepath": data["filepath"], "extri": extri_opencv})

    path = f"{category}/{seq_name}"
    if path in scene_pools:
        pool = scene_pools[path]
        ids = np.asarray(pool["frame_indices"], dtype=np.int64)
        image_names = [os.path.join(args.co3d_image_root, filepath) for filepath in pool["filepaths"]]
    else:
        ids = np.random.choice(len(metadata), args.num_frames, replace=False)
        image_names = [os.path.join(args.co3d_image_root, metadata[i]["filepath"]) for i in ids]
    gt_extri = np.stack([np.array(metadata[i]["extri"]) for i in ids], axis=0)

    images = load_images_with_retry(
        pi3_core=pi3_core,
        image_names=image_names,
        new_width=args.load_img_size,
        device=str(device),
        verbose=False,
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

    pred_extrinsic = torch.from_numpy(extract_pred_extrinsics(predictions)).to(device=device, dtype=torch.float64)
    gt_extrinsic = torch.from_numpy(gt_extri).to(device=device, dtype=torch.float64)

    add_row = torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float64).expand(pred_extrinsic.size(0), 1, 4)
    pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
    gt_se3 = torch.cat((gt_extrinsic, add_row), dim=1)

    rel_rangle_deg, rel_tangle_deg = upstream.se3_to_relative_pose_error(pred_se3, gt_se3, args.num_frames)
    r_error = rel_rangle_deg.cpu().numpy()
    t_error = rel_tangle_deg.cpu().numpy()

    racc_5 = float((rel_rangle_deg < 5).float().mean().item())
    tacc_5 = float((rel_tangle_deg < 5).float().mean().item())
    auc_30, _ = upstream.calculate_auc_np(r_error, t_error, max_threshold=30)
    auc_20, _ = upstream.calculate_auc_np(r_error, t_error, max_threshold=20)
    auc_15, _ = upstream.calculate_auc_np(r_error, t_error, max_threshold=15)
    auc_10, _ = upstream.calculate_auc_np(r_error, t_error, max_threshold=10)
    auc_5, _ = upstream.calculate_auc_np(r_error, t_error, max_threshold=5)
    auc_3, _ = upstream.calculate_auc_np(r_error, t_error, max_threshold=3)

    return {
        "category": category,
        "sequence_name": seq_name,
        "path": f"{category}/{seq_name}",
        "num_images_total": int(len(seq_data)),
        "num_frames_eval": int(args.num_frames),
        "sampled_ids": ids.astype(int).tolist(),
        "sampled_filepaths": [metadata[i]["filepath"] for i in ids],
        "R_ACC_5": float(racc_5),
        "T_ACC_5": float(tacc_5),
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


def main():
    args = setup_args()
    upstream = load_upstream_module()
    device = resolve_device(args)
    dtype = resolve_autocast_dtype(device)
    model, loaded_ckpt, pi3_core = load_pi3_model_runtime(args.model_path, device)
    scene_pools = maybe_load_scene_pools(args.co3d_image_root)

    upstream.set_random_seeds(args.seed)
    categories = parse_category_list(args)
    per_category_results: Dict[str, dict] = {}
    per_sequence_results: Dict[str, List[dict]] = {}

    for category in categories:
        annotation_file = Path(args.co3d_anno_dir) / f"{category}_test.jgz"
        try:
            annotation = load_annotation(annotation_file)
        except FileNotFoundError:
            print(f"Annotation file not found for {category}, skipping", flush=True)
            continue

        seq_names = sorted(list(annotation.keys()))
        if args.fast_eval and len(seq_names) >= 10:
            seq_names = random.sample(seq_names, 10)
        seq_names = sorted(seq_names)

        sequence_results: List[dict] = []
        print(f"Testing sequences for {category}: {seq_names}", flush=True)

        for seq_name in seq_names:
            print("-" * 50, flush=True)
            print(f"Processing {seq_name} for {category} test set", flush=True)
            if (args.debug or args.debug_category) and not os.path.exists(os.path.join(args.co3d_image_root, category, seq_name)):
                print(f"Skipping {seq_name} (not found)", flush=True)
                continue

            seq_result = process_sequence(
                model=model,
                seq_name=seq_name,
                seq_data=annotation[seq_name],
                category=category,
                args=args,
                device=device,
                dtype=dtype,
                pi3_core=pi3_core,
                upstream=upstream,
                scene_pools=scene_pools,
            )
            print("-" * 50, flush=True)

            if seq_result is None:
                continue
            sequence_results.append(seq_result)
            print(
                f"[sequence] {category}/{seq_name} "
                f"AUC30={seq_result['AUC_30']:.6f} "
                f"R_ACC5={seq_result['R_ACC_5']:.6f} "
                f"T_ACC5={seq_result['T_ACC_5']:.6f}",
                flush=True,
            )

        if not sequence_results:
            print(f"No valid sequences found for {category}, skipping", flush=True)
            continue

        per_sequence_results[category] = sequence_results
        per_category_results[category] = upstream.summarize_category(sequence_results)
        summary = per_category_results[category]
        print(
            f"[category-summary] {category} "
            f"AUC30={summary['AUC_30']:.6f} "
            f"AUC20={summary['AUC_20']:.6f} "
            f"AUC10={summary['AUC_10']:.6f}",
            flush=True,
        )

    if not per_category_results:
        raise RuntimeError("No category produced valid results.")

    metrics = [item for results in per_sequence_results.values() for item in results]
    overall_summary = build_overall_summary(
        per_category_results=per_category_results,
        num_sequences=sum(len(results) for results in per_sequence_results.values()),
    )
    result = {
        "implementation": "pi3_aligned_official_co3d",
        "model_tag": args.model_tag,
        "model_path": args.model_path or default_model_path(),
        "loaded_model": loaded_ckpt,
        "co3d_dir": str(args.co3d_image_root),
        "co3d_anno_dir": str(args.co3d_anno_dir),
        "categories": categories,
        "fast_eval": bool(args.fast_eval),
        "min_num_images": int(args.min_num_images),
        "num_frames": int(args.num_frames),
        "seed": int(args.seed),
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
        f"-> {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
