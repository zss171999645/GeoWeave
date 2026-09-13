#!/usr/bin/env python3
"""PI3 video pose evaluator for Sintel/TUM/ScanNet without Hydra."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_DATA_ROOTS: Dict[str, str] = {
    "sintel": "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/datasets/pi3_relpose_exact_benchmark/sintel",
    "tum": "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/datasets/pi3_relpose_exact_benchmark/tum_dynamics",
    "scannetv2": "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/datasets/pi3_relpose_exact_benchmark/scannet_exact100",
    "vkitti2": "/horizon-bucket/saturn_v_dev/users/tao02.xie/datasets/vkitti2",
}

PAPER_TARGETS: Dict[str, Dict[str, float]] = {
    "sintel": {"ATE": 0.074, "RPE trans": 0.040, "RPE rot": 0.282},
    "tum": {"ATE": 0.014, "RPE trans": 0.009, "RPE rot": 0.312},
    "scannetv2": {"ATE": 0.031, "RPE trans": 0.013, "RPE rot": 0.347},
    "vkitti2": {"ATE": 0.0, "RPE trans": 0.0, "RPE rot": 0.0},
}

SINTEL_SEQUENCES: Sequence[str] = (
    "alley_2",
    "ambush_4",
    "ambush_5",
    "ambush_6",
    "cave_2",
    "cave_4",
    "market_2",
    "market_5",
    "market_6",
    "shaman_3",
    "sleeping_1",
    "sleeping_2",
    "temple_2",
    "temple_3",
)

DEFAULT_LOAD_IMG_SIZE = 512
DEFAULT_POSE_EVAL_STRIDE = 1
EVC_TARGET_NUM_FRAMES = 90
SUPPORTED_IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
DEFAULT_VGGT_OFFICIAL_CKPT_ROOT = (
    "/horizon-bucket/saturn_v_dev/01_users/tao02.xie/projects/GFM/gfm/data/trained_model/vggt/official/VGGT-1B"
)
DEFAULT_PI3_RELPOSE_EVO_UTILS_CANDIDATES: Tuple[str, ...] = (
    "tmp/external_refs/pi3-official2/relpose/evo_utils.py",
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/external_refs/pi3-official2_clean/relpose/evo_utils.py",
)
DEFAULT_VGGT_FINETUNED_CONFIG = "aidi/configs/vggt/finetune_5090_sparse_topk2048_l9_19_p34_record.yaml"
DEFAULT_VGGT_OFFICIAL_CONFIG = "configs/exps/vggt/vggt_official_eval_paper.yaml"


class DatasetSpec:
    def __init__(
        self,
        name: str,
        img_subdir: str,
        img_ext: str,
        anno_path: str,
        anno_format: str,
        sequence_names: Sequence[str] | None = None,
    ) -> None:
        self.name = name
        self.img_subdir = img_subdir
        self.img_ext = img_ext
        self.anno_path = anno_path
        self.anno_format = anno_format
        self.sequence_names = sequence_names


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "sintel": DatasetSpec(
        name="sintel",
        img_subdir="final/{seq}",
        img_ext="png",
        anno_path="camdata_left/{seq}",
        anno_format="sintel",
        sequence_names=SINTEL_SEQUENCES,
    ),
    "tum": DatasetSpec(
        name="tum",
        img_subdir="{seq}/rgb_90",
        img_ext="png",
        anno_path="{seq}/groundtruth_90.txt",
        anno_format="tum",
        sequence_names=None,
    ),
    "scannetv2": DatasetSpec(
        name="scannetv2",
        img_subdir="{seq}/color_90",
        img_ext="jpg",
        anno_path="{seq}/pose_90.txt",
        anno_format="replica",
        sequence_names=None,
    ),
    "vkitti2": DatasetSpec(
        name="vkitti2",
        img_subdir="{seq}/color_90",
        img_ext="jpg",
        anno_path="{seq}/pose_90.txt",
        anno_format="replica",
        sequence_names=None,
    ),
}

_RE10K_RUNTIME = None
_EVO_UTILS_RUNTIME = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_repo_root_on_syspath() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


ensure_repo_root_on_syspath()


def load_module_from_file(module_name: str, path: Path):
    ensure_repo_root_on_syspath()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_re10k_runtime():
    global _RE10K_RUNTIME
    if _RE10K_RUNTIME is None:
        _RE10K_RUNTIME = load_module_from_file(
            "eval_pi3_re10k_pose_official_runtime",
            repo_root() / "aidi" / "scripts" / "baselines" / "eval_pi3_re10k_pose_official.py",
        )
    return _RE10K_RUNTIME


def ensure_headless_matplotlib_backend() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        from evo.tools import settings as evo_settings

        evo_settings.SETTINGS.plot_backend = "Agg"
    except Exception:
        pass


def resolve_evo_utils_path() -> Path:
    for candidate in DEFAULT_PI3_RELPOSE_EVO_UTILS_CANDIDATES:
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = repo_root() / candidate_path
        candidate_path = candidate_path.resolve()
        if candidate_path.is_file():
            return candidate_path
    searched = [str((repo_root() / item).resolve()) if not Path(item).is_absolute() else item for item in DEFAULT_PI3_RELPOSE_EVO_UTILS_CANDIDATES]
    raise FileNotFoundError(f"Cannot find PI3 relpose evo_utils.py in candidates: {searched}")


def load_evo_utils_runtime():
    global _EVO_UTILS_RUNTIME
    if _EVO_UTILS_RUNTIME is None:
        ensure_headless_matplotlib_backend()
        _EVO_UTILS_RUNTIME = load_module_from_file(
            "pi3_relpose_evo_utils_runtime",
            resolve_evo_utils_path(),
        )
    return _EVO_UTILS_RUNTIME


def load_read_camera():
    from easyvolcap.utils.easy_utils import read_camera

    return read_camera


def parse_dataset_names(value: str) -> List[str]:
    requested = (value or "all").strip().lower()
    if requested == "all":
        return list(DATASET_SPECS.keys())
    names = [item.strip() for item in requested.split(",") if item.strip()]
    unknown = [item for item in names if item not in DATASET_SPECS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    return names


def parse_eval_frame_indices(value: str) -> List[int]:
    raw = (value or "").strip()
    if not raw:
        return []
    indices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if any(index < 0 for index in indices):
        raise ValueError(f"eval frame indices must be non-negative: {indices}")
    return indices


def subset_pose_array_for_eval(poses: np.ndarray, eval_indices: Sequence[int]) -> np.ndarray:
    if not eval_indices:
        return np.asarray(poses)
    poses = np.asarray(poses)
    max_index = int(poses.shape[0]) - 1
    missing = [int(index) for index in eval_indices if int(index) > max_index]
    if missing:
        raise IndexError(f"eval frame indices out of range for length={poses.shape[0]}: {missing}")
    return poses[np.asarray([int(index) for index in eval_indices], dtype=np.int64)]


def load_replica_pose_file(pose_path: Path) -> np.ndarray:
    rows: List[np.ndarray] = []
    for line in pose_path.read_text(encoding="utf-8").strip().splitlines():
        values = [float(item) for item in line.split()]
        if len(values) != 16:
            raise ValueError(f"Expected 16 values per pose row in {pose_path}")
        rows.append(np.asarray(values, dtype=np.float64).reshape(4, 4))
    if not rows:
        raise ValueError(f"Empty pose file: {pose_path}")
    return np.stack(rows, axis=0)


def load_official_gt_c2w_for_eval(spec: DatasetSpec, gt_path: Path, stride: int, eval_indices: Sequence[int]) -> np.ndarray:
    if spec.anno_format != "replica":
        raise NotImplementedError(
            f"eval-frame-indices is only implemented for replica-style official pose files, got {spec.name}/{spec.anno_format}"
        )
    gt_c2w = load_replica_pose_file(gt_path)
    gt_c2w = apply_pose_eval_stride_to_gt([gt_c2w], stride)[0]
    return subset_pose_array_for_eval(gt_c2w, eval_indices)


def build_dataset_summary(seq_metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not seq_metrics:
        return {"ATE": 0.0, "RPE trans": 0.0, "RPE rot": 0.0, "num_sequences": 0}

    def _mean(key: str) -> float:
        values = [float(item[key]) for item in seq_metrics if key in item]
        return float(np.mean(values)) if values else 0.0

    return {
        "ATE": _mean("ATE"),
        "RPE trans": _mean("RPE trans"),
        "RPE rot": _mean("RPE rot"),
        "num_sequences": int(len(seq_metrics)),
    }


def detect_layout(spec: DatasetSpec, root: Path, require_official_layout: bool = False) -> str:
    layout = None
    if spec.name == "sintel":
        if (root / "final").is_dir() and (root / "camdata_left").is_dir():
            layout = "official"
        else:
            raise FileNotFoundError(f"Cannot detect supported layout for dataset={spec.name} under {root}")

    seq_dirs = sorted([path for path in root.iterdir() if path.is_dir()]) if root.is_dir() else []
    if layout is None and spec.name == "tum":
        if any((seq / "rgb_90").is_dir() and (seq / "groundtruth_90.txt").is_file() for seq in seq_dirs):
            layout = "official"
        elif any((seq / "images" / "00").is_dir() and (seq / "cameras" / "00").is_dir() for seq in seq_dirs):
            layout = "evc"
    elif layout is None and spec.name == "scannetv2":
        if any((seq / "color_90").is_dir() and (seq / "pose_90.txt").is_file() for seq in seq_dirs):
            layout = "official"
        elif any((seq / "images").is_dir() and (seq / "intri.yml").is_file() and (seq / "extri.yml").is_file() for seq in seq_dirs):
            layout = "evc"
    elif layout is None and spec.name == "vkitti2":
        if any((seq / "color_90").is_dir() and (seq / "pose_90.txt").is_file() for seq in seq_dirs):
            layout = "official"

    if layout is None:
        raise FileNotFoundError(
            f"Cannot detect supported layout for dataset={spec.name} under {root}. "
            "Expected PI3 official layout or EVC image/camera layout."
        )
    if require_official_layout and layout != "official":
        raise RuntimeError(
            f"Dataset {spec.name} under {root} is not in official layout. "
            "Run the strict exact preparation script first or disable --require-official-layout."
        )
    return layout


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PI3 video pose benchmark without Hydra.")
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset list or 'all'.")
    parser.add_argument("--sintel-root", default=DEFAULT_DATA_ROOTS["sintel"])
    parser.add_argument("--tum-root", default=DEFAULT_DATA_ROOTS["tum"])
    parser.add_argument("--scannet-root", default=DEFAULT_DATA_ROOTS["scannetv2"])
    parser.add_argument("--vkitti-root", default=DEFAULT_DATA_ROOTS["vkitti2"])
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-family", choices=("pi3", "vggt"), default="pi3")
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
    parser.add_argument("--device", default="")
    parser.add_argument("--load-img-size", type=int, default=DEFAULT_LOAD_IMG_SIZE)
    parser.add_argument("--pose-eval-stride", type=int, default=DEFAULT_POSE_EVAL_STRIDE)
    parser.add_argument("--image-load-retries", type=int, default=8)
    parser.add_argument("--image-load-retry-sleep", type=float, default=0.5)
    parser.add_argument("--limit-seqs", type=int, default=0)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--eval-frame-indices",
        default="",
        help="Comma-separated 0-based frame indices to score after pose_eval_stride; all input frames are still used for inference.",
    )
    parser.add_argument("--model-tag", default="pi3")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--require-official-layout", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--vggt-model-tag", choices=("official", "finetuned"), default="finetuned")
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


def resolve_vggt_config_path(model_tag: str, provided_path: str) -> str:
    if provided_path:
        return provided_path
    if model_tag == "official":
        return DEFAULT_VGGT_OFFICIAL_CONFIG
    return DEFAULT_VGGT_FINETUNED_CONFIG


def load_vggt_eval_config(cfg_path: str | Path):
    from aidi.scripts.baselines.eval_config_utils import load_resolved_config

    return load_resolved_config(cfg_path)


def collect_vggt_indexer_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    mapping = {
        "topk_block": getattr(args, "vggt_topk_block_override", None),
        "topk_merge_blocks": getattr(args, "vggt_topk_merge_blocks_override", None),
        "source_downsample_factor": getattr(args, "vggt_source_downsample_factor_override", None),
        "source_downsample_query_chunk": getattr(args, "vggt_source_downsample_query_chunk_override", None),
        "source_downsample_coarse_topk": getattr(args, "vggt_source_downsample_coarse_topk_override", None),
        "source_downsample_coarse_ratio": getattr(args, "vggt_source_downsample_coarse_ratio_override", None),
    }
    for key, value in mapping.items():
        if value is not None:
            overrides[key] = value
    strategy = getattr(args, "vggt_source_downsample_strategy_override", "")
    if strategy:
        overrides["source_downsample_strategy"] = strategy
    if getattr(args, "vggt_source_downsample_enabled", False):
        overrides["source_downsample_enabled"] = True
    return overrides


def collect_vggt_head_chunk_overrides(args: argparse.Namespace) -> Dict[str, int]:
    overrides: Dict[str, int] = {}
    if getattr(args, "vggt_depth_frames_chunk_size_override", None) is not None:
        overrides["depth_frames_chunk_size"] = int(args.vggt_depth_frames_chunk_size_override)
    if getattr(args, "vggt_point_frames_chunk_size_override", None) is not None:
        overrides["point_frames_chunk_size"] = int(args.vggt_point_frames_chunk_size_override)
    return overrides


def build_vggt_model_cfg(
    raw_model_cfg: Mapping[str, Any],
    model_tag: str,
    official_ckpt_root: str,
    topk_override: int,
    clear_component_ckpts: bool,
    indexer_overrides: Mapping[str, Any] | None = None,
    head_chunk_overrides: Mapping[str, int] | None = None,
):
    from easyvolcap.utils.base_utils import dotdict

    model_cfg = dotdict(copy.deepcopy(dict(raw_model_cfg)))
    model_cfg.pop("type", None)
    model_cfg.pretrained_path = ""
    if model_tag == "official":
        ckpt_root = Path(official_ckpt_root)
        model_cfg.agg_ckpt = str(ckpt_root / "aggregator.pt")
        model_cfg.cam_ckpt = str(ckpt_root / "camera.pt")
        model_cfg.xyz_ckpt = str(ckpt_root / "point.pt")
        model_cfg.dpt_ckpt = str(ckpt_root / "depth.pt")
        model_cfg.tra_ckpt = str(ckpt_root / "track.pt")
    elif clear_component_ckpts:
        for key in ("agg_ckpt", "cam_ckpt", "xyz_ckpt", "dpt_ckpt", "tra_ckpt"):
            model_cfg[key] = ""

    if topk_override > 0:
        model_cfg.setdefault("vggt_cfg", dotdict())
        model_cfg.vggt_cfg.setdefault("indexer_cfg", dotdict())
        model_cfg.vggt_cfg.indexer_cfg.topk = int(topk_override)
    if indexer_overrides:
        model_cfg.setdefault("vggt_cfg", dotdict())
        model_cfg.vggt_cfg.setdefault("indexer_cfg", dotdict())
        for key, value in indexer_overrides.items():
            model_cfg.vggt_cfg.indexer_cfg[key] = value
    if head_chunk_overrides:
        model_cfg.setdefault("vggt_cfg", dotdict())
        if "depth_frames_chunk_size" in head_chunk_overrides:
            model_cfg.vggt_cfg.setdefault("depth_head_cfg", dotdict())
            model_cfg.vggt_cfg.depth_head_cfg.setdefault("chunk_cfg", dotdict())
            model_cfg.vggt_cfg.depth_head_cfg.chunk_cfg.frames_chunk_size = int(
                head_chunk_overrides["depth_frames_chunk_size"]
            )
        if "point_frames_chunk_size" in head_chunk_overrides:
            model_cfg.vggt_cfg.setdefault("point_head_cfg", dotdict())
            model_cfg.vggt_cfg.point_head_cfg.setdefault("chunk_cfg", dotdict())
            model_cfg.vggt_cfg.point_head_cfg.chunk_cfg.frames_chunk_size = int(
                head_chunk_overrides["point_frames_chunk_size"]
            )
    return model_cfg


def resolve_dataset_root(args: argparse.Namespace, dataset_name: str) -> Path:
    mapping = {
        "sintel": getattr(args, "sintel_root", DEFAULT_DATA_ROOTS["sintel"]),
        "tum": getattr(args, "tum_root", DEFAULT_DATA_ROOTS["tum"]),
        "scannetv2": getattr(args, "scannet_root", DEFAULT_DATA_ROOTS["scannetv2"]),
        "vkitti2": getattr(args, "vkitti_root", DEFAULT_DATA_ROOTS["vkitti2"]),
    }
    return Path(mapping[dataset_name]).expanduser().resolve()


def default_output_dir(args: argparse.Namespace, dataset_names: Sequence[str]) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_tag = "-".join(dataset_names)
    return repo_root() / "tmp" / f"pi3_relpose_distance_{dataset_tag}_{args.model_tag}_{timestamp}"


def list_sequence_names(spec: DatasetSpec, root: Path, layout: str) -> List[str]:
    if spec.sequence_names is not None:
        if layout == "official":
            return [seq for seq in spec.sequence_names if (root / spec.img_subdir.format(seq=seq)).is_dir()]
        return [seq for seq in spec.sequence_names if (root / seq).is_dir()]

    seq_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if layout == "official":
        return [path.name for path in seq_dirs]

    if spec.name == "tum":
        return [path.name for path in seq_dirs if (path / "images" / "00").is_dir() and (path / "cameras" / "00").is_dir()]
    if spec.name == "scannetv2":
        return [path.name for path in seq_dirs if (path / "images").is_dir() and (path / "intri.yml").is_file() and (path / "extri.yml").is_file()]
    return [path.name for path in seq_dirs]


def list_images_for_sequence(spec: DatasetSpec, root: Path, seq: str) -> List[str]:
    img_dir = root / spec.img_subdir.format(seq=seq)
    preferred = [path for path in sorted(img_dir.glob(f"*.{spec.img_ext}")) if path.is_file()]
    if preferred:
        return [str(path) for path in preferred]
    return [
        str(path)
        for path in sorted(img_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ]


def resolve_gt_path(spec: DatasetSpec, root: Path, seq: str) -> Path:
    return root / spec.anno_path.format(seq=seq)


def resolve_first_image_file(candidates: Sequence[Path]) -> Path | None:
    for path in candidates:
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            return path
    return None


def build_tum_evc_image_map(seq_root: Path) -> Dict[str, Path]:
    image_dir = seq_root / "images" / "00"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing TUM EVC image dir: {image_dir}")
    image_map = {
        path.stem: path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
    }
    if not image_map:
        raise FileNotFoundError(f"No supported TUM EVC images found under {image_dir}")
    return image_map


def build_scannet_evc_image_map(seq_root: Path) -> Dict[str, Path]:
    image_root = seq_root / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing ScanNet EVC image dir: {image_root}")

    image_map: Dict[str, Path] = {}
    for frame_dir in sorted(image_root.iterdir()):
        if not frame_dir.is_dir():
            continue
        image_path = resolve_first_image_file(
            sorted([path for path in frame_dir.iterdir() if path.is_file()], key=lambda path: path.name)
        )
        if image_path is not None:
            image_map[frame_dir.name] = image_path
    if not image_map:
        raise FileNotFoundError(f"No supported ScanNet EVC images found under {image_root}")
    return image_map


def camera_rt_to_c2w(rt: np.ndarray, dataset_name: str = "") -> np.ndarray:
    rt = np.asarray(rt, dtype=np.float64)
    if rt.shape != (3, 4):
        raise ValueError(f"Expected camera RT with shape (3, 4), got {rt.shape}")
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :4] = rt
    if dataset_name == "scannetv2":
        return c2w
    return np.linalg.inv(c2w)


def align_image_map_and_camera_names(image_map: Mapping[str, Path], camera_names: Sequence[str]) -> List[str]:
    camera_name_set = set(camera_names)
    aligned = [name for name in sorted(image_map.keys(), key=lambda item: int(item)) if name in camera_name_set]
    if not aligned:
        raise RuntimeError("No overlapping frame names between images and cameras")
    return aligned


def build_c2w_stack_from_cameras(
    cameras: Mapping[str, Any],
    ordered_names: Sequence[str],
    dataset_name: str = "",
) -> np.ndarray:
    poses = [
        camera_rt_to_c2w(np.asarray(cameras[name].RT, dtype=np.float64), dataset_name=dataset_name)
        for name in ordered_names
    ]
    if not poses:
        raise RuntimeError("No camera poses available for ordered sequence names")
    return np.stack(poses, axis=0)


def subsample_evc_sequence(
    ordered_names: Sequence[str],
    gt_c2w: np.ndarray,
    target_num_frames: int = EVC_TARGET_NUM_FRAMES,
) -> Tuple[List[str], np.ndarray]:
    if target_num_frames <= 0 or len(ordered_names) <= target_num_frames:
        return list(ordered_names), gt_c2w
    # Match the MonST3R/PI3 dataset preparation scripts: keep the prefix window
    # of 270 frames and downsample with stride 3 to produce at most 90 frames.
    source_limit = min(len(ordered_names), target_num_frames * 3)
    indices = np.arange(0, source_limit, 3, dtype=np.int64)
    if len(indices) > target_num_frames:
        indices = indices[:target_num_frames]
    selected_names = [ordered_names[int(index)] for index in indices]
    return selected_names, gt_c2w[indices]


def resolve_evc_cameras(dataset_name: str, seq_root: Path):
    read_camera = load_read_camera()
    if dataset_name == "tum":
        intri = seq_root / "cameras" / "00" / "intri.yml"
        extri = seq_root / "cameras" / "00" / "extri.yml"
    elif dataset_name == "scannetv2":
        intri = seq_root / "intri.yml"
        extri = seq_root / "extri.yml"
    else:
        raise ValueError(f"EVC camera resolution is not implemented for dataset={dataset_name}")
    if (not intri.is_file()) or (not extri.is_file()):
        raise FileNotFoundError(f"Missing camera files: {intri}, {extri}")
    return read_camera(str(intri), str(extri), use_dict=False)


def load_sequence_inputs(
    spec: DatasetSpec,
    root: Path,
    seq: str,
    layout: str,
) -> Tuple[List[str], List[np.ndarray]]:
    if layout == "official":
        image_names = list_images_for_sequence(spec, root, seq)
        gt_traj_file = resolve_gt_path(spec, root, seq)
        return image_names, [gt_traj_file]

    seq_root = root / seq
    if spec.name == "tum":
        image_map = build_tum_evc_image_map(seq_root)
    elif spec.name == "scannetv2":
        image_map = build_scannet_evc_image_map(seq_root)
    else:
        raise ValueError(f"EVC layout is not implemented for dataset={spec.name}")

    cameras = resolve_evc_cameras(spec.name, seq_root)
    ordered_names = align_image_map_and_camera_names(image_map=image_map, camera_names=list(cameras.keys()))
    gt_c2w = build_c2w_stack_from_cameras(cameras=cameras, ordered_names=ordered_names, dataset_name=spec.name)
    ordered_names, gt_c2w = subsample_evc_sequence(ordered_names=ordered_names, gt_c2w=gt_c2w)
    image_names = [str(image_map[name]) for name in ordered_names]
    return image_names, [gt_c2w]


def apply_pose_eval_stride_to_gt(gt_traj: Sequence[np.ndarray], stride: int) -> List[np.ndarray]:
    if stride <= 1:
        return [np.asarray(item) for item in gt_traj]
    return [np.asarray(item)[::stride] for item in gt_traj]


def write_csv_row(file_path: Path, row: Dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not file_path.is_file()
    with file_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_vggt_model_runtime(args: argparse.Namespace, device):
    import torch

    from easyvolcap.models.official_vggt_model import OfficialVGGTModel

    cfg_path = resolve_vggt_config_path(model_tag=args.vggt_model_tag, provided_path=args.vggt_config)
    cfg = load_vggt_eval_config(cfg_path)
    model_cfg = build_vggt_model_cfg(
        raw_model_cfg=cfg.model_cfg,
        model_tag=args.vggt_model_tag,
        official_ckpt_root=args.vggt_official_ckpt_root,
        topk_override=args.vggt_topk_override,
        clear_component_ckpts=not args.vggt_keep_component_ckpts,
        indexer_overrides=collect_vggt_indexer_overrides(args),
        head_chunk_overrides=collect_vggt_head_chunk_overrides(args),
    )
    model = OfficialVGGTModel(**model_cfg).to(device=device).eval()
    if args.vggt_model_tag == "official":
        loaded_ckpt = str(Path(args.vggt_official_ckpt_root))
    else:
        if not args.model_path:
            raise ValueError("--model-path is required when --model-family=vggt and --vggt-model-tag=finetuned")
        ckpt_path = Path(args.model_path)
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        state_dict, _ = OfficialVGGTModel._remap_special_token_keys(state_dict)
        model.load_state_dict(state_dict, strict=False)
        loaded_ckpt = str(ckpt_path)
    return model, loaded_ckpt


def load_pi3_model_runtime(args: argparse.Namespace, device):
    pi3_config = getattr(args, "pi3_config", "") or os.environ.get("PI3_CONFIG", "")
    pi3_model_impl = getattr(args, "pi3_model_impl", "") or os.environ.get("PI3_MODEL_IMPL", "")
    pi3_native_root = (
        getattr(args, "pi3_native_root", "")
        or os.environ.get("PI3_NATIVE_ROOT", "")
        or "aidi/third_party/pi3_training"
    )
    if pi3_config or pi3_model_impl:
        from aidi.scripts.baselines.pi3_checkpoint_loader import load_pi3_model_for_eval
        from easyvolcap.utils.pi3.models.pi3 import Pi3

        runtime = load_re10k_runtime()
        ckpt = args.model_path or runtime.default_model_path()
        return load_pi3_model_for_eval(
            ckpt,
            device,
            official_pi3_cls=Pi3,
            native_root=pi3_native_root,
            model_impl=pi3_model_impl,
            config_path=pi3_config,
        )
    runtime = load_re10k_runtime()
    return runtime.load_pi3_model_runtime(model_path=args.model_path, device=device)


def load_images_with_retry(image_names: List[str], new_width: int, device: str, retries: int, retry_sleep: float):
    runtime = load_re10k_runtime()
    return runtime.load_images_with_retry(
        image_names=image_names,
        new_width=new_width,
        device=device,
        retries=retries,
        retry_sleep=retry_sleep,
    )


def resolve_device(device_arg: str):
    runtime = load_re10k_runtime()
    args = argparse.Namespace(device=device_arg)
    return runtime.resolve_device(args)


def resolve_autocast_dtype(device):
    runtime = load_re10k_runtime()
    return runtime.resolve_autocast_dtype(device)


def load_model_runtime(args: argparse.Namespace, device):
    if args.model_family == "vggt":
        return load_vggt_model_runtime(args=args, device=device)
    return load_pi3_model_runtime(args, device)


def vggt_extrinsics_to_c2w(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.ndim == 2:
        extrinsics = extrinsics[None]
    if extrinsics.shape[-2:] == (3, 4):
        hom = np.tile(np.eye(4, dtype=extrinsics.dtype), (extrinsics.shape[0], 1, 1))
        hom[:, :3, :4] = extrinsics
    elif extrinsics.shape[-2:] == (4, 4):
        hom = extrinsics
    else:
        raise ValueError(f"Expected extrinsics with shape (N,3,4) or (N,4,4), got {extrinsics.shape}")
    return np.linalg.inv(hom)


def infer_pi3_cameras_c2w(
    image_names: List[str],
    model,
    device,
    load_img_size: int,
    image_load_retries: int,
    image_load_retry_sleep: float,
):
    import torch

    imgs = load_images_with_retry(
        image_names=image_names,
        new_width=load_img_size,
        device=str(device),
        retries=image_load_retries,
        retry_sleep=image_load_retry_sleep,
    )
    dtype = resolve_autocast_dtype(device)
    autocast_enabled = getattr(device, "type", "") == "cuda"
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            pred = model(imgs)
    return pred["camera_poses"][0].detach().float().cpu().numpy()


def infer_vggt_cameras_c2w(
    image_names: List[str],
    model,
    device,
    verbose: bool,
):
    import torch

    from easyvolcap.official_vggt.utils.load_fn import load_and_preprocess_images as load_and_preprocess_vggt_images
    from easyvolcap.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from easyvolcap.utils.base_utils import dotdict

    images = load_and_preprocess_vggt_images([str(path) for path in image_names]).to(device=device).unsqueeze(0)
    h, w = images.shape[-2:]
    batch = dotdict(
        images=images,
        meta=dotdict(
            iter=torch.tensor(0, device=images.device),
            H=torch.tensor([h], device=images.device),
            W=torch.tensor([w], device=images.device),
        ),
    )
    dtype = resolve_autocast_dtype(device)
    autocast_enabled = getattr(device, "type", "") == "cuda"
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            output = model(batch)
    if not hasattr(output, "cam_map"):
        raise RuntimeError("VGGT relpose inference requires output.cam_map")
    extrinsics, _ = pose_encoding_to_extri_intri(output.cam_map, image_size_hw=(h, w))
    pred_c2w = vggt_extrinsics_to_c2w(extrinsics[0].detach().float().cpu().numpy()).astype(np.float32)
    if verbose:
        print(f"[vggt-relpose] decoded {pred_c2w.shape[0]} poses at image_size_hw=({h}, {w})", flush=True)
    return pred_c2w


def infer_cameras_c2w(
    args: argparse.Namespace,
    image_names: List[str],
    model,
    device,
    load_img_size: int,
    image_load_retries: int,
    image_load_retry_sleep: float,
):
    if args.model_family == "vggt":
        return infer_vggt_cameras_c2w(
            image_names=image_names,
            model=model,
            device=device,
            verbose=args.verbose,
        )
    return infer_pi3_cameras_c2w(
        image_names=image_names,
        model=model,
        device=device,
        load_img_size=load_img_size,
        image_load_retries=image_load_retries,
        image_load_retry_sleep=image_load_retry_sleep,
    )


def evaluate_dataset(
    dataset_name: str,
    args: argparse.Namespace,
    model,
    loaded_ckpt: str,
    output_root: Path,
) -> Dict[str, Any]:
    spec = DATASET_SPECS[dataset_name]
    evo_utils = load_evo_utils_runtime()
    root = resolve_dataset_root(args, dataset_name)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found for {dataset_name}: {root}")

    layout = detect_layout(spec, root, require_official_layout=args.require_official_layout)
    seq_names = list_sequence_names(spec, root, layout)
    if args.limit_seqs > 0:
        seq_names = seq_names[: args.limit_seqs]
    eval_frame_indices = parse_eval_frame_indices(getattr(args, "eval_frame_indices", ""))

    dataset_root = output_root / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)

    seq_results: List[Dict[str, Any]] = []
    for seq in seq_names:
        image_names, gt_payload = load_sequence_inputs(spec=spec, root=root, seq=seq, layout=layout)
        if args.pose_eval_stride > 1:
            image_names = image_names[:: args.pose_eval_stride]
        if not image_names:
            continue

        pred_poses = infer_cameras_c2w(
            args=args,
            image_names=image_names,
            model=model,
            device=resolve_device(args.device),
            load_img_size=args.load_img_size,
            image_load_retries=args.image_load_retries,
            image_load_retry_sleep=args.image_load_retry_sleep,
        )
        pred_poses_eval = subset_pose_array_for_eval(pred_poses, eval_frame_indices)
        pred_traj = evo_utils.get_tum_poses(pred_poses_eval)

        seq_root = dataset_root / seq
        seq_root.mkdir(parents=True, exist_ok=True)
        evo_utils.save_tum_poses(pred_traj, str(seq_root / "pred_traj.txt"), verbose=args.verbose)
        if eval_frame_indices:
            np.save(seq_root / "pred_poses_full.npy", pred_poses)
            (seq_root / "eval_frame_indices.json").write_text(
                json.dumps([int(index) for index in eval_frame_indices], indent=2),
                encoding="utf-8",
            )
        np.save(seq_root / "pred_poses.npy", pred_poses_eval)

        if layout == "official":
            if eval_frame_indices:
                gt_c2w = load_official_gt_c2w_for_eval(
                    spec=spec,
                    gt_path=Path(gt_payload[0]),
                    stride=args.pose_eval_stride,
                    eval_indices=eval_frame_indices,
                )
                gt_traj = evo_utils.get_tum_poses(gt_c2w)
            else:
                try:
                    gt_traj = evo_utils.load_traj(
                        gt_traj_file=str(gt_payload[0]),
                        traj_format=spec.anno_format,
                        stride=args.pose_eval_stride,
                    )
                except np.linalg.LinAlgError:
                    print(f"Warning: failed to load ground truth trajectory for {dataset_name}/{seq}, skipping.")
                    continue
        else:
            gt_c2w = apply_pose_eval_stride_to_gt(gt_payload, args.pose_eval_stride)[0]
            gt_c2w = subset_pose_array_for_eval(gt_c2w, eval_frame_indices)
            gt_traj = evo_utils.get_tum_poses(gt_c2w)
        ate, rpe_trans, rpe_rot = evo_utils.eval_metrics(
            pred_traj=pred_traj,
            gt_traj=gt_traj,
            seq=seq,
            filename=str(seq_root / "eval_metric.txt"),
            verbose=args.verbose,
        )
        if not args.skip_plot:
            evo_utils.plot_trajectory(
                pred_traj=pred_traj,
                gt_traj=gt_traj,
                title=seq,
                filename=str(seq_root / "vis.png"),
                align=True,
                correct_scale=True,
                verbose=args.verbose,
            )

        row = {
            "dataset": dataset_name,
            "seq": seq,
            "ATE": float(ate),
            "RPE trans": float(rpe_trans),
            "RPE rot": float(rpe_rot),
        }
        write_csv_row(dataset_root / "seq_metrics.csv", row)
        seq_results.append(row)

    dataset_summary = build_dataset_summary(seq_results)
    write_csv_row(
        output_root / f"{dataset_name}-metric.csv",
        {
            "dataset": dataset_name,
            "ATE": dataset_summary["ATE"],
            "RPE trans": dataset_summary["RPE trans"],
            "RPE rot": dataset_summary["RPE rot"],
            "num_sequences": dataset_summary["num_sequences"],
        },
    )
    return {
        "dataset": dataset_name,
        "root": str(root),
        "layout": layout,
        "paper_target": PAPER_TARGETS[dataset_name],
        "summary": dataset_summary,
        "num_sequences": len(seq_results),
        "ckpt": loaded_ckpt,
    }


def build_run_summary(dataset_results: Sequence[Dict[str, Any]], dataset_names: Sequence[str], loaded_ckpt: str, output_root: Path) -> Dict[str, Any]:
    return {
        "datasets": list(dataset_names),
        "ckpt": loaded_ckpt,
        "output_root": str(output_root),
        "results": list(dataset_results),
    }


def main() -> None:
    args = setup_args()
    dataset_names = parse_dataset_names(args.datasets)
    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(args, dataset_names)
    output_root.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model, loaded_ckpt = load_model_runtime(args, device)

    dataset_results = []
    for dataset_name in dataset_names:
        dataset_results.append(
            evaluate_dataset(
                dataset_name=dataset_name,
                args=args,
                model=model,
                loaded_ckpt=loaded_ckpt,
                output_root=output_root,
            )
        )

    summary = build_run_summary(dataset_results, dataset_names, loaded_ckpt, output_root)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
