#!/usr/bin/env python3
"""Evaluate VGGT / Pi3 on MegaDepth-1500 or ScanNet-1500 relative pose benchmarks."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import time
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
    "megadepth1500": DEFAULT_BENCHMARK_ROOT / "megadepth1500",
    "scannet1500": DEFAULT_BENCHMARK_ROOT / "scannet1500",
}
DEFAULT_MANIFEST_ROOTS: Dict[str, Path] = {
    "megadepth1500": DEFAULT_BENCHMARK_ROOT / "megadepth1500",
    "scannet1500": DEFAULT_BENCHMARK_ROOT / "scannet1500",
}
DEFAULT_VGGT_OFFICIAL_CKPT_ROOT = (
    "/horizon-bucket/saturn_v_dev/01_users/tao02.xie/projects/GFM/gfm/data/trained_model/vggt/official/VGGT-1B"
)
DEFAULT_PI3_LOAD_IMG_SIZE = 512
DEFAULT_IMAGE_LOAD_RETRIES = 8
DEFAULT_IMAGE_LOAD_RETRY_SLEEP = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("megadepth1500", "scannet1500"), required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--manifest-root", default="")
    parser.add_argument("--pair-limit", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-family", choices=("vggt", "pi3"), default="vggt")
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
    return parser.parse_args()


def resolve_dataset_root(args: argparse.Namespace) -> Path:
    if args.dataset_root:
        return Path(args.dataset_root).expanduser().resolve()
    return DEFAULT_DATA_ROOTS[args.dataset].expanduser().resolve()


def resolve_manifest_root(args: argparse.Namespace) -> Path:
    if args.manifest_root:
        return Path(args.manifest_root).expanduser().resolve()
    return DEFAULT_MANIFEST_ROOTS[args.dataset].expanduser().resolve()


def default_output_path(args: argparse.Namespace) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_tag = f"{args.model_family}_{args.vggt_model_tag}" if args.model_family == "vggt" else args.model_family
    return repo_root() / "tmp" / f"{args.dataset}_{model_tag}_{timestamp}.json"


def load_runtime_module():
    path = repo_root() / "aidi" / "scripts" / "baselines" / "eval_pi3_relpose_distance_protocol.py"
    spec = importlib.util.spec_from_file_location("eval_pi3_relpose_distance_protocol_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading runtime module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(args: argparse.Namespace):
    runtime = load_runtime_module()
    device = runtime.resolve_device(args.device)
    model, loaded_ckpt = runtime.load_model_runtime(args=args, device=device)
    return model, device, runtime, loaded_ckpt


def call_runtime_infer_cameras_c2w(runtime, args: argparse.Namespace, image_names: List[str], model, device):
    infer_fn = runtime.infer_cameras_c2w
    supported = inspect.signature(infer_fn).parameters
    candidate_kwargs = {
        "args": args,
        "image_names": image_names,
        "model": model,
        "device": device,
        "load_img_size": args.load_img_size,
        "image_load_retries": args.image_load_retries,
        "image_load_retry_sleep": args.image_load_retry_sleep,
        "verbose": args.verbose,
    }
    infer_kwargs = {name: value for name, value in candidate_kwargs.items() if name in supported}
    return infer_fn(**infer_kwargs)


def as_homogeneous_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim == 2:
        pose = pose[None]
    if pose.shape[-2:] == (4, 4):
        return pose
    if pose.shape[-2:] != (3, 4):
        raise ValueError(f"Unsupported pose shape: {pose.shape}")
    hom = np.tile(np.eye(4, dtype=np.float32), (pose.shape[0], 1, 1))
    hom[:, :3, :4] = pose
    return hom


def normalize_pose_stack(poses: np.ndarray, convention: str) -> np.ndarray:
    hom = as_homogeneous_pose(poses)
    if convention == "c2w":
        return hom
    if convention == "w2c":
        return np.linalg.inv(hom).astype(np.float32)
    raise ValueError(f"Unsupported pose convention: {convention}")


def rotation_error_deg(rel_gt: np.ndarray, rel_pred: np.ndarray) -> float:
    rel_r = rel_pred @ rel_gt.T
    tr = float(np.clip((np.trace(rel_r) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(tr)))


def translation_error_deg(vec_gt: np.ndarray, vec_pred: np.ndarray) -> float:
    gt_n = float(np.linalg.norm(vec_gt))
    pred_n = float(np.linalg.norm(vec_pred))
    if gt_n < 1e-12 or pred_n < 1e-12:
        return 0.0
    cos = float(np.clip(np.dot(vec_gt, vec_pred) / (gt_n * pred_n), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    hom = as_homogeneous_pose(c2w)
    return np.linalg.inv(hom).astype(np.float32)


def calculate_auc_np(error_r: np.ndarray, error_t: np.ndarray, threshold: int = 30, combine: str = "max") -> float:
    if error_r.ndim < 2:
        error_r = error_r[None]
        error_t = error_t[None]

    combine = (combine or "max").strip().lower()
    if combine in {"max", "legacy", "legacy_max", "max_err"}:
        error_max = np.maximum(error_r, error_t)
        auc = 0.0
        for batch_index in range(error_max.shape[0]):
            bins = np.arange(threshold + 1)
            histogram, _ = np.histogram(error_max[batch_index], bins=bins)
            histogram_norm = histogram.astype(np.float64) / float(len(error_max[batch_index]))
            auc += float(np.mean(np.cumsum(histogram_norm)))
        return auc / float(error_max.shape[0])

    if combine not in {"min", "paper", "min_rra_rta", "min_rra"}:
        raise ValueError(f"Unknown combine={combine!r}, expected min|max.")

    thresholds = np.arange(1, threshold + 1, dtype=np.float64)
    auc = 0.0
    for batch_index in range(error_r.shape[0]):
        rra = np.mean(error_r[batch_index][:, None] < thresholds[None, :], axis=0)
        rta = np.mean(error_t[batch_index][:, None] < thresholds[None, :], axis=0)
        auc += float(np.mean(np.minimum(rra, rta)))
    return auc / float(error_r.shape[0])


def calculate_pose_auc(errors_r: Sequence[float], errors_t: Sequence[float], threshold: int) -> float:
    r_error = np.asarray(list(errors_r), dtype=np.float64)[None]
    t_error = np.asarray(list(errors_t), dtype=np.float64)[None]
    return float(calculate_auc_np(r_error, t_error, threshold=threshold, combine="max"))


def compute_pair_pose_metrics(gt_c2w: np.ndarray, pred_c2w: np.ndarray) -> Dict[str, float]:
    gt_c2w = normalize_pose_stack(gt_c2w, convention="c2w")
    pred_c2w = normalize_pose_stack(pred_c2w, convention="c2w")
    if gt_c2w.shape[0] != 2 or pred_c2w.shape[0] != 2:
        raise ValueError("Pairwise benchmark expects exactly two poses per sample")

    gt_w2c = c2w_to_w2c(gt_c2w)
    pred_w2c = c2w_to_w2c(pred_c2w)

    rel_gt = gt_w2c[0] @ np.linalg.inv(gt_w2c[1])
    rel_pred = pred_w2c[0] @ np.linalg.inv(pred_w2c[1])

    rot_gt = rel_gt[:3, :3]
    rot_pred = rel_pred[:3, :3]
    trans_gt = rel_gt[:3, 3]
    trans_pred = rel_pred[:3, 3]
    rot_error = rotation_error_deg(rot_gt, rot_pred)
    trans_error_raw = translation_error_deg(trans_gt, trans_pred)
    trans_error = min(trans_error_raw, abs(180.0 - trans_error_raw))
    return {
        "rotation_error_deg": float(rot_error),
        "translation_error_deg": float(trans_error),
        "pose_auc_05": calculate_pose_auc([rot_error], [trans_error], threshold=5),
        "pose_auc_10": calculate_pose_auc([rot_error], [trans_error], threshold=10),
        "pose_auc_20": calculate_pose_auc([rot_error], [trans_error], threshold=20),
    }


def build_summary(pair_metrics: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not pair_metrics:
        return {
            "AUC@5": 0.0,
            "AUC@10": 0.0,
            "AUC@20": 0.0,
            "rotation_error_median": 0.0,
            "translation_error_median": 0.0,
            "num_pairs": 0,
        }

    def _mean(key: str) -> float:
        return float(np.mean([float(item[key]) for item in pair_metrics]))

    def _median(key: str) -> float:
        return float(np.median([float(item[key]) for item in pair_metrics]))

    return {
        "AUC@5": _mean("pose_auc_05"),
        "AUC@10": _mean("pose_auc_10"),
        "AUC@20": _mean("pose_auc_20"),
        "rotation_error_median": _median("rotation_error_deg"),
        "translation_error_median": _median("translation_error_deg"),
        "num_pairs": int(len(pair_metrics)),
    }


def resolve_existing_path(root: Path, value: Any) -> Path:
    raw = Path(str(value))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(root / raw)
        candidates.append(root / raw.as_posix().lstrip("/"))
        candidates.append(root / raw.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Failed resolving path {value} under {root}")


def load_txt_matrix(path: Path) -> np.ndarray:
    rows = [
        [float(item) for item in line.split()]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.shape[0] >= 3 and matrix.shape[1] >= 3:
        return matrix[:3, :3]
    raise ValueError(f"Unsupported matrix shape in {path}: {matrix.shape}")


def parse_megadepth_pair_infos(pair_infos: Any) -> List[tuple[int, int]]:
    if isinstance(pair_infos, np.ndarray):
        items = pair_infos.tolist()
    else:
        items = list(pair_infos)
    parsed: List[tuple[int, int]] = []
    for item in items:
        pair = item.tolist() if isinstance(item, np.ndarray) else item
        if isinstance(pair, (list, tuple)) and len(pair) >= 2 and np.isscalar(pair[0]) and np.isscalar(pair[1]):
            parsed.append((int(pair[0]), int(pair[1])))
            continue
        pair = pair[0] if isinstance(pair, (list, tuple)) and pair else pair
        if isinstance(pair, np.ndarray):
            pair = pair.tolist()
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            parsed.append((int(pair[0]), int(pair[1])))
            continue
        raise ValueError(f"Unsupported MegaDepth pair info format: {item!r}")
    return parsed


def load_megadepth_pairs(manifest_root: Path, dataset_root: Path, pair_limit: int = 0) -> List[SimpleNamespace]:
    scene_list_path = manifest_root / "megadepth_test_1500.txt"
    if not scene_list_path.is_file():
        raise FileNotFoundError(f"MegaDepth-1500 scene list not found: {scene_list_path}")

    samples: List[SimpleNamespace] = []
    scene_names = [line.strip() for line in scene_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for scene_name in scene_names:
        scene_info_path = manifest_root / f"{scene_name}.npz"
        if not scene_info_path.is_file():
            raise FileNotFoundError(f"MegaDepth scene info not found: {scene_info_path}")
        with np.load(scene_info_path, allow_pickle=True) as scene_info:
            image_paths = [resolve_existing_path(dataset_root, path) for path in scene_info["image_paths"].tolist()]
            poses = normalize_pose_stack(np.asarray(scene_info["poses"], dtype=np.float32), convention="w2c")
            intrinsics = np.asarray(scene_info["intrinsics"], dtype=np.float32).reshape((-1, 3, 3))
            pair_indices = parse_megadepth_pair_infos(scene_info["pair_infos"])

        for local_index, (idx0, idx1) in enumerate(pair_indices):
            samples.append(
                SimpleNamespace(
                    pair_id=f"{scene_name}:{local_index:04d}",
                    scene_name=scene_name,
                    image_paths=[str(image_paths[idx0]), str(image_paths[idx1])],
                    gt_c2w=np.asarray([poses[idx0], poses[idx1]], dtype=np.float32),
                    intrinsics=np.asarray([intrinsics[idx0], intrinsics[idx1]], dtype=np.float32),
                )
            )
            if pair_limit > 0 and len(samples) >= pair_limit:
                return samples
    return samples


def load_scannet_intrinsics(scene_root: Path, scene_name: str, manifest_root: Path) -> np.ndarray:
    manifest_intrinsics = manifest_root / "intrinsics.npz"
    if manifest_intrinsics.is_file():
        with np.load(manifest_intrinsics, allow_pickle=True) as payload:
            if scene_name in payload.files:
                return np.asarray(payload[scene_name], dtype=np.float32).reshape((3, 3))
    intrinsic_path = scene_root / "intrinsic" / "intrinsic_color.txt"
    if intrinsic_path.is_file():
        return load_txt_matrix(intrinsic_path)
    raise FileNotFoundError(f"Missing ScanNet intrinsics for {scene_name}")


def parse_scannet_name_array(name_array: np.ndarray) -> List[tuple[str, int, int]]:
    names = np.asarray(name_array)
    if names.ndim != 2 or names.shape[1] < 4:
        raise ValueError(f"Unsupported ScanNet pair array shape: {names.shape}")
    pairs: List[tuple[str, int, int]] = []
    for row in names.tolist():
        scene_id, seq_id, image0_id, image1_id = [int(value) for value in row[:4]]
        scene_name = f"scene{scene_id:04d}_{seq_id:02d}"
        pairs.append((scene_name, image0_id, image1_id))
    return pairs


def load_scannet_pairs(manifest_root: Path, dataset_root: Path, pair_limit: int = 0) -> List[SimpleNamespace]:
    pair_list_path = manifest_root / "scannet_test.txt"
    if not pair_list_path.is_file():
        raise FileNotFoundError(f"ScanNet-1500 list not found: {pair_list_path}")
    listed_files = [line.strip() for line in pair_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not listed_files:
        raise RuntimeError(f"No ScanNet pair manifests listed in {pair_list_path}")

    samples: List[SimpleNamespace] = []
    for listed_file in listed_files:
        manifest_path = manifest_root / listed_file
        if not manifest_path.is_file():
            raise FileNotFoundError(f"ScanNet pair manifest not found: {manifest_path}")
        with np.load(manifest_path, allow_pickle=True) as payload:
            pair_key = "name" if "name" in payload.files else payload.files[0]
            pair_specs = parse_scannet_name_array(payload[pair_key])

        for local_index, (scene_name, image0_id, image1_id) in enumerate(pair_specs):
            scene_root = dataset_root / scene_name
            pose0 = np.loadtxt(scene_root / "pose" / f"{image0_id}.txt", dtype=np.float32)
            pose1 = np.loadtxt(scene_root / "pose" / f"{image1_id}.txt", dtype=np.float32)
            intrinsic = load_scannet_intrinsics(scene_root=scene_root, scene_name=scene_name, manifest_root=manifest_root)
            samples.append(
                SimpleNamespace(
                    pair_id=f"{scene_name}:{local_index:04d}",
                    scene_name=scene_name,
                    image_paths=[
                        str((scene_root / "color" / f"{image0_id}.jpg").resolve()),
                        str((scene_root / "color" / f"{image1_id}.jpg").resolve()),
                    ],
                    gt_c2w=np.asarray([pose0, pose1], dtype=np.float32),
                    intrinsics=np.asarray([intrinsic, intrinsic], dtype=np.float32),
                )
            )
            if pair_limit > 0 and len(samples) >= pair_limit:
                return samples
    return samples


def load_benchmark_pairs(dataset_name: str, manifest_root: Path, dataset_root: Path, pair_limit: int = 0) -> List[SimpleNamespace]:
    if dataset_name == "megadepth1500":
        return load_megadepth_pairs(manifest_root=manifest_root, dataset_root=dataset_root, pair_limit=pair_limit)
    if dataset_name == "scannet1500":
        return load_scannet_pairs(manifest_root=manifest_root, dataset_root=dataset_root, pair_limit=pair_limit)
    raise ValueError(f"Unsupported benchmark dataset: {dataset_name}")


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args)
    manifest_root = resolve_manifest_root(args)
    pair_records = load_benchmark_pairs(args.dataset, manifest_root=manifest_root, dataset_root=dataset_root, pair_limit=args.pair_limit)
    model, device, runtime, loaded_ckpt = load_model(args)

    per_pair_metrics: List[Dict[str, Any]] = []
    for record in pair_records:
        pred_c2w = call_runtime_infer_cameras_c2w(
            runtime=runtime,
            args=args,
            image_names=record.image_paths,
            model=model,
            device=device,
        )
        metric = compute_pair_pose_metrics(gt_c2w=record.gt_c2w, pred_c2w=pred_c2w)
        metric.update(
            {
                "pair_id": record.pair_id,
                "scene_name": record.scene_name,
                "image_paths": list(record.image_paths),
            }
        )
        per_pair_metrics.append(metric)
        if args.verbose:
            print(
                f"[{record.pair_id}] "
                f"AUC20={metric['pose_auc_20']:.6f} "
                f"AUC10={metric['pose_auc_10']:.6f} "
                f"AUC5={metric['pose_auc_05']:.6f} "
                f"rot={metric['rotation_error_deg']:.4f} "
                f"trans={metric['translation_error_deg']:.4f}",
                flush=True,
            )

    summary = build_summary(per_pair_metrics)
    payload = {
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "manifest_root": str(manifest_root),
        "model_family": args.model_family,
        "loaded_ckpt": str(loaded_ckpt),
        "summary": summary,
        "per_pair_metrics": per_pair_metrics,
    }
    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
