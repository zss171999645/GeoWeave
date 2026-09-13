#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import glob
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAVE_ROOT = "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline"
DEFAULT_OFFICIAL_CKPT_ROOT = "/horizon-bucket/saturn_v_dev/01_users/tao02.xie/projects/GFM/gfm/data/trained_model/vggt/official/VGGT-1B"
DEFAULT_PI3_CKPT = "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/pretrained/pi3/yyfz233_Pi3_model.safetensors"
DEFAULT_PI3_ROOT = "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/external_refs/pi3-official2_clean"
DEFAULT_RE10K_ROOT = "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/datasets/re10k/processed_pose1800_clusterfix/test"
KNOWN_5090_OOM_OFFICIAL_VIDEODEPTH_TASKS = frozenset(
    {
        "bonn_videodepth",
        "7scenes_videodepth",
        "vkitti2_videodepth",
    }
)


class TaskSpec:
    __slots__ = ("task", "family", "dataset", "variant", "adapter", "mode")

    def __init__(
        self,
        *,
        task: str,
        family: str,
        dataset: str,
        variant: str,
        adapter: str,
        mode: str = "",
    ) -> None:
        self.task = task
        self.family = family
        self.dataset = dataset
        self.variant = variant
        self.adapter = adapter
        self.mode = mode


def _task_spec(*, task: str, family: str, dataset: str, variant: str, adapter: str, **kwargs: str) -> TaskSpec:
    return TaskSpec(task=task, family=family, dataset=dataset, variant=variant, adapter=adapter, **kwargs)


TASK_REGISTRY: Dict[str, TaskSpec] = {
    "re10k_pose": _task_spec(
        task="re10k_pose",
        family="pose",
        dataset="re10k",
        variant="pose",
        adapter="re10k_pose_lightweight",
    ),
    "co3dv2_pose": _task_spec(
        task="co3dv2_pose",
        family="pose",
        dataset="co3dv2",
        variant="pose",
        adapter="co3dv2_pose_upstream",
    ),
    "eth3d_pose": _task_spec(
        task="eth3d_pose",
        family="pose",
        dataset="eth3d",
        variant="pose",
        adapter="da3_pose",
    ),
    "hiroom_pose": _task_spec(
        task="hiroom_pose",
        family="pose",
        dataset="hiroom",
        variant="pose",
        adapter="da3_pose",
    ),
    "scannetpp_pose": _task_spec(
        task="scannetpp_pose",
        family="pose",
        dataset="scannetpp",
        variant="pose",
        adapter="da3_pose",
    ),
    "dtu64_pose": _task_spec(
        task="dtu64_pose",
        family="pose",
        dataset="dtu64",
        variant="pose",
        adapter="da3_pose",
    ),
    "7scenes_pose": _task_spec(
        task="7scenes_pose",
        family="pose",
        dataset="7scenes",
        variant="pose",
        adapter="da3_pose",
    ),
    "megadepth1500_pose": _task_spec(
        task="megadepth1500_pose",
        family="pose",
        dataset="megadepth1500",
        variant="pose",
        adapter="relpose1500",
    ),
    "scannet1500_pose": _task_spec(
        task="scannet1500_pose",
        family="pose",
        dataset="scannet1500",
        variant="pose",
        adapter="relpose1500",
    ),
    "sintel_pose": _task_spec(
        task="sintel_pose",
        family="pose",
        dataset="sintel",
        variant="pose",
        adapter="relpose_distance",
        mode="sintel",
    ),
    "tum_pose": _task_spec(
        task="tum_pose",
        family="pose",
        dataset="tum",
        variant="pose",
        adapter="relpose_distance",
        mode="tum",
    ),
    "scannet_pose": _task_spec(
        task="scannet_pose",
        family="pose",
        dataset="scannet",
        variant="pose",
        adapter="relpose_distance",
        mode="scannetv2",
    ),
}

for depth_dataset in (
    "sintel",
    "bonn",
    "kitti",
    "eth3d",
    "7scenes",
    "blendedmvs",
    "mvs_synth",
    "scannetpp",
    "vkitti2",
    "co3dv2",
    "diode",
):
    TASK_REGISTRY[f"{depth_dataset}_monodepth"] = _task_spec(
        task=f"{depth_dataset}_monodepth",
        family="depth",
        dataset=depth_dataset,
        variant="monodepth",
        adapter="depth_protocol",
        mode="monodepth",
    )
    TASK_REGISTRY[f"{depth_dataset}_videodepth"] = _task_spec(
        task=f"{depth_dataset}_videodepth",
        family="depth",
        dataset=depth_dataset,
        variant="videodepth",
        adapter="depth_protocol",
        mode="videodepth",
    )

TASK_REGISTRY["dtu_pointcloud"] = _task_spec(
    task="dtu_pointcloud",
    family="pointcloud",
    dataset="dtu",
    variant="pointcloud",
    adapter="pointcloud_pi3_style",
)
TASK_REGISTRY["eth3d_pointcloud"] = _task_spec(
    task="eth3d_pointcloud",
    family="pointcloud",
    dataset="eth3d",
    variant="pointcloud",
    adapter="pointcloud_pi3_style",
)
TASK_REGISTRY["7scenes_sparse_pointcloud"] = _task_spec(
    task="7scenes_sparse_pointcloud",
    family="pointcloud",
    dataset="7scenes",
    variant="sparse_pointcloud",
    adapter="pointcloud_pi3_style",
    mode="sparse",
)
TASK_REGISTRY["7scenes_dense_pointcloud"] = _task_spec(
    task="7scenes_dense_pointcloud",
    family="pointcloud",
    dataset="7scenes",
    variant="dense_pointcloud",
    adapter="pointcloud_pi3_style",
    mode="dense",
)
TASK_REGISTRY["nrgbd_sparse_pointcloud"] = _task_spec(
    task="nrgbd_sparse_pointcloud",
    family="pointcloud",
    dataset="nrgbd",
    variant="sparse_pointcloud",
    adapter="pointcloud_pi3_style",
    mode="sparse",
)
TASK_REGISTRY["nrgbd_dense_pointcloud"] = _task_spec(
    task="nrgbd_dense_pointcloud",
    family="pointcloud",
    dataset="nrgbd",
    variant="dense_pointcloud",
    adapter="pointcloud_pi3_style",
    mode="dense",
)


ALL_FAMILIES = ("pose", "depth", "pointcloud")
ALL_DATASETS = tuple(dict.fromkeys(spec.dataset for spec in TASK_REGISTRY.values()))
DEFAULT_DA3_ROOT_FLAGS = {
    "eth3d": "--eth3d-root",
    "hiroom": "--hiroom-root",
    "scannetpp": "--scannetpp-root",
    "dtu64": "--dtu64-root",
    "7scenes": "--7scenes-root",
}
DEFAULT_RELPOSE_DISTANCE_ROOT_FLAGS = {
    "sintel": "--sintel-root",
    "tum": "--tum-root",
    "scannet": "--scannet-root",
}
POSE_TASK_BY_DATASET = {
    spec.dataset: task_id
    for task_id, spec in TASK_REGISTRY.items()
    if spec.family == "pose" and spec.variant == "pose"
}
DEPTH_TASK_BY_DATASET_AND_MODE = {
    (spec.dataset, spec.mode): task_id
    for task_id, spec in TASK_REGISTRY.items()
    if spec.family == "depth"
}
POINTCLOUD_TASK_BY_DATASET_AND_MODE = {
    (spec.dataset, spec.mode or "pointcloud"): task_id
    for task_id, spec in TASK_REGISTRY.items()
    if spec.family == "pointcloud"
}


def build_default_config() -> Dict[str, Any]:
    return {
        "run": {
            "name": "vggt_unified_eval",
            "gpu_ids": "auto",
            "device": "auto",
            "python_bin": sys.executable or "python3",
            "save_root": DEFAULT_SAVE_ROOT,
            "output_root": "tmp/vggt_unified_eval",
            "dry_run": False,
            "skip_existing": False,
            "continue_on_error": False,
            "env": {},
            "allow_co3dv2_videodepth": False,
            "auto_skip_known_oom_official_videodepth_on_5090": True,
            "force_run_known_oom_tasks": [],
        },
        "model": {
            "kind": "official",
            "checkpoint": "",
            "config": "",
            "official_ckpt_root": DEFAULT_OFFICIAL_CKPT_ROOT,
            "pi3_root": DEFAULT_PI3_ROOT,
            "pi3_model_impl": "",
            "pi3_config": "",
            "pi3_native_root": "",
        },
        "datasets": {},
        "tasks": {
            task_id: {
                "enabled": False,
            }
            for task_id in TASK_REGISTRY
        },
    }


DEFAULT_CONFIG = build_default_config()


def deep_update(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def merge_mapping(target: Dict[str, Any], override: Mapping[str, Any]) -> None:
    for key, value in override.items():
        target[key] = copy.deepcopy(value)


def extract_shared_block(block: Mapping[str, Any], reserved_keys: Sequence[str]) -> Dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in block.items()
        if key not in set(reserved_keys)
    }


def apply_dataset_centric_tasks(cfg: Dict[str, Any]) -> None:
    for dataset_name, dataset_cfg in cfg.get("datasets", {}).items():
        if not isinstance(dataset_cfg, Mapping):
            continue

        pose_cfg = dataset_cfg.get("pose")
        if isinstance(pose_cfg, Mapping):
            task_id = POSE_TASK_BY_DATASET.get(dataset_name)
            if task_id:
                merge_mapping(cfg["tasks"].setdefault(task_id, {}), pose_cfg)

        depth_cfg = dataset_cfg.get("depth")
        if isinstance(depth_cfg, Mapping):
            shared_depth = extract_shared_block(depth_cfg, ("monodepth", "videodepth"))
            for mode in ("monodepth", "videodepth"):
                mode_cfg = depth_cfg.get(mode)
                if not isinstance(mode_cfg, Mapping):
                    continue
                task_id = DEPTH_TASK_BY_DATASET_AND_MODE.get((dataset_name, mode))
                if not task_id:
                    continue
                task_cfg = cfg["tasks"].setdefault(task_id, {})
                merge_mapping(task_cfg, shared_depth)
                merge_mapping(task_cfg, mode_cfg)

        pointcloud_cfg = dataset_cfg.get("pointcloud")
        if isinstance(pointcloud_cfg, Mapping):
            single_task_id = POINTCLOUD_TASK_BY_DATASET_AND_MODE.get((dataset_name, "pointcloud"))
            if single_task_id:
                merge_mapping(cfg["tasks"].setdefault(single_task_id, {}), pointcloud_cfg)
                continue

            shared_pointcloud = extract_shared_block(pointcloud_cfg, ("sparse", "dense"))
            for mode in ("sparse", "dense"):
                mode_cfg = pointcloud_cfg.get(mode)
                if not isinstance(mode_cfg, Mapping):
                    continue
                task_id = POINTCLOUD_TASK_BY_DATASET_AND_MODE.get((dataset_name, mode))
                if not task_id:
                    continue
                task_cfg = cfg["tasks"].setdefault(task_id, {})
                merge_mapping(task_cfg, shared_pointcloud)
                merge_mapping(task_cfg, mode_cfg)


def resolve_path(text: str, repo_root: Path) -> str:
    if not text:
        return ""
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    return str((repo_root / path).resolve())


def maybe_resolve_path(text: Any, repo_root: Path) -> str:
    if text is None:
        return ""
    value = str(text).strip()
    if not value:
        return ""
    return resolve_path(value, repo_root)


def ensure_path_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def co3dv2_evc_root_ready(root: Path, *, min_categories: int, min_scenes: int) -> tuple[bool, str]:
    if not root.is_dir():
        return False, "not a directory"
    try:
        categories = sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.name)
    except OSError as exc:
        return False, f"cannot list categories: {exc}"
    nonempty_categories = 0
    scene_count = 0
    for category in categories:
        try:
            scenes = [path for path in category.iterdir() if path.is_dir()]
        except OSError as exc:
            return False, f"cannot list {category.name}: {exc}"
        if scenes:
            nonempty_categories += 1
            scene_count += len(scenes)
    if nonempty_categories < min_categories:
        return False, f"only {nonempty_categories}/{min_categories} non-empty categories"
    if scene_count < min_scenes:
        return False, f"only {scene_count}/{min_scenes} scenes"
    return True, f"{nonempty_categories} categories, {scene_count} scenes"


def data_root_candidate_ready(
    config: Mapping[str, Any],
    task_id: str,
    data_root: str,
    repo_root: Path,
) -> tuple[bool, str]:
    spec = TASK_REGISTRY[task_id]
    path = Path(data_root)
    if spec.dataset == "co3dv2":
        min_categories = int(task_value(config, task_id, "data_root_min_categories", 41))
        min_scenes = int(task_value(config, task_id, "data_root_min_scenes", 20000))
        return co3dv2_evc_root_ready(path, min_categories=min_categories, min_scenes=min_scenes)
    if not path.exists():
        return False, "path does not exist"
    return True, "exists"


def resolve_data_root_for_task(config: Mapping[str, Any], task_id: str, repo_root: Path) -> str:
    primary = task_value(config, task_id, "data_root", "")
    candidates = ensure_path_values(primary) + ensure_path_values(task_value(config, task_id, "data_root_candidates", []))
    resolved_candidates: List[str] = []
    seen = set()
    for item in candidates:
        resolved = maybe_resolve_path(item, repo_root)
        if resolved and resolved not in seen:
            resolved_candidates.append(resolved)
            seen.add(resolved)
    if not resolved_candidates:
        return ""

    # Preserve historical behavior unless the config explicitly asks for fallback candidates.
    if not ensure_path_values(task_value(config, task_id, "data_root_candidates", [])):
        return resolved_candidates[0]

    failures: List[str] = []
    for candidate in resolved_candidates:
        ok, reason = data_root_candidate_ready(config, task_id, candidate, repo_root)
        if ok:
            if failures:
                print(
                    f"[unified-eval] selected fallback data_root for {task_id}: {candidate}",
                    file=sys.stderr,
                )
            return candidate
        failures.append(f"{candidate} ({reason})")
    if bool(config.get("run", {}).get("dry_run")):
        fallback = resolved_candidates[-1]
        print(
            f"[unified-eval] dry-run could not validate data_root candidates for {task_id}; "
            f"using last candidate for command expansion: {fallback}",
            file=sys.stderr,
        )
        return fallback
    raise FileNotFoundError(
        f"No usable data_root found for {task_id}. Checked: " + "; ".join(failures)
    )


def normalize_config(raw: Dict[str, Any], repo_root: Path | None = None) -> Dict[str, Any]:
    repo_root = repo_root or REPO_ROOT
    legacy_keys = [key for key in ("families", "tasks") if key in (raw or {})]
    if legacy_keys:
        raise ValueError(
            "Legacy unified-eval config keys are no longer supported: "
            + ", ".join(legacy_keys)
            + ". Please use dataset-centric `datasets.<dataset>.{pose,depth,pointcloud}` blocks."
        )
    cfg = deep_update(copy.deepcopy(DEFAULT_CONFIG), raw or {})
    run_cfg = cfg["run"]
    model_cfg = cfg["model"]

    model_kind = str(model_cfg.get("kind", "official")).strip().lower()
    if model_kind not in {"official", "custom", "pi3"}:
        raise ValueError(f"Unsupported model.kind={model_kind}")
    model_cfg["kind"] = model_kind
    if model_kind == "custom":
        if not model_cfg.get("checkpoint"):
            raise ValueError("Custom model requires model.checkpoint")
        if not model_cfg.get("config"):
            raise ValueError("Custom model requires model.config")
        model_cfg["checkpoint"] = maybe_resolve_path(model_cfg["checkpoint"], repo_root)
        model_cfg["config"] = maybe_resolve_path(model_cfg["config"], repo_root)
    elif model_kind == "pi3":
        model_cfg["checkpoint"] = maybe_resolve_path(model_cfg.get("checkpoint", ""), repo_root) or DEFAULT_PI3_CKPT
        model_cfg["pi3_model_impl"] = str(model_cfg.get("pi3_model_impl", "")).strip()
        model_cfg["pi3_config"] = maybe_resolve_path(model_cfg.get("pi3_config", ""), repo_root)
        model_cfg["pi3_native_root"] = maybe_resolve_path(model_cfg.get("pi3_native_root", ""), repo_root)
    model_cfg["official_ckpt_root"] = maybe_resolve_path(model_cfg.get("official_ckpt_root", ""), repo_root) or DEFAULT_OFFICIAL_CKPT_ROOT
    model_cfg["pi3_root"] = maybe_resolve_path(model_cfg.get("pi3_root", ""), repo_root) or DEFAULT_PI3_ROOT

    run_cfg["output_root"] = resolve_path(run_cfg.get("output_root", ""), repo_root)
    run_cfg["python_bin"] = str(run_cfg.get("python_bin", sys.executable or "python3")).strip() or "python3"
    if not run_cfg.get("save_root"):
        run_cfg["save_root"] = DEFAULT_SAVE_ROOT
    run_cfg["save_root"] = maybe_resolve_path(run_cfg["save_root"], repo_root) or DEFAULT_SAVE_ROOT

    for task_id in TASK_REGISTRY:
        cfg["tasks"].setdefault(task_id, {})
        cfg["tasks"][task_id].setdefault("enabled", False)

    apply_dataset_centric_tasks(cfg)

    if not select_task_ids(cfg):
        raise ValueError("No dataset-centric eval tasks enabled.")
    return cfg


def pick_auto_gpu_ids() -> str:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return "0"

    rows = []
    for line in output.strip().splitlines():
        idx, mem_used, mem_total, util = [item.strip() for item in line.split(",")]
        rows.append((int(idx), int(mem_used), int(mem_total), int(util)))
    if not rows:
        raise RuntimeError("No GPUs reported by nvidia-smi")

    preferred = [row for row in rows if row[1] <= 1024 and row[3] == 0]
    pool = preferred if preferred else sorted(rows, key=lambda row: (row[1], row[3], -row[2]))
    return str(pool[0][0])


def resolve_gpu_ids(run_cfg: Mapping[str, Any]) -> str:
    gpu_ids = str(run_cfg.get("gpu_ids", "")).strip()
    if not gpu_ids or gpu_ids.lower() == "auto":
        return pick_auto_gpu_ids()
    return gpu_ids


def resolve_device(run_cfg: Mapping[str, Any], gpu_ids: str) -> str:
    device = str(run_cfg.get("device", "auto")).strip()
    if device and device.lower() != "auto":
        return device
    first_gpu = gpu_ids.split(",")[0].strip()
    if first_gpu:
        return f"cuda:{first_gpu}"
    return "cpu"


def split_gpu_ids(gpu_ids: str) -> List[str]:
    return [item.strip() for item in str(gpu_ids).split(",") if item.strip()]


def query_gpu_name_map() -> Dict[str, str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name",
                "--format=csv,noheader",
            ],
            text=True,
        )
    except Exception:
        return {}
    gpu_name_map: Dict[str, str] = {}
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        idx, name = [item.strip() for item in line.split(",", 1)]
        gpu_name_map[idx] = name
    return gpu_name_map


def assigned_gpu_indices(gpu_ids: str, device: str) -> List[str]:
    device_text = str(device).strip()
    if device_text.startswith("cuda:"):
        suffix = device_text.split(":", 1)[1].strip()
        if suffix.isdigit():
            return [suffix]
    return split_gpu_ids(gpu_ids)


def make_skipped_record(
    task_id: str,
    *,
    reason: str,
    gpu_ids: str,
    device: str,
) -> Dict[str, Any]:
    spec = TASK_REGISTRY[task_id]
    return {
        "task": task_id,
        "family": spec.family,
        "dataset": spec.dataset,
        "variant": spec.variant,
        "status": "SKIPPED",
        "skip_reason": reason,
        "assigned_gpu_ids": gpu_ids,
        "assigned_device": device,
        "summary_path": "",
        "summary": {},
        "payload": {
            "status": "SKIPPED",
            "skip_reason": reason,
            "assigned_gpu_ids": gpu_ids,
            "assigned_device": device,
        },
    }


def auto_skip_reason(config: Mapping[str, Any], task_id: str, gpu_ids: str, device: str) -> str:
    run_cfg = config["run"]
    model_cfg = config["model"]
    if task_id == "co3dv2_videodepth" and not bool(run_cfg.get("allow_co3dv2_videodepth", False)):
        return "CO3Dv2 videodepth is disabled by default; set run.allow_co3dv2_videodepth=true to opt in"
    if not bool(run_cfg.get("auto_skip_known_oom_official_videodepth_on_5090", True)):
        return ""
    if str(model_cfg.get("kind", "")).strip().lower() != "official":
        return ""
    if task_id in set(ensure_list(run_cfg.get("force_run_known_oom_tasks", []))):
        return ""
    if task_id not in KNOWN_5090_OOM_OFFICIAL_VIDEODEPTH_TASKS:
        return ""
    gpu_name_map = query_gpu_name_map()
    gpu_indices = assigned_gpu_indices(gpu_ids, device)
    if not gpu_indices:
        return ""
    gpu_names = [gpu_name_map.get(idx, "") for idx in gpu_indices]
    if not gpu_names or not all("5090" in name for name in gpu_names if name):
        return ""
    if not any(gpu_names):
        return ""
    return f"known official videodepth OOM on 5090 ({', '.join(gpu_names)})"


def assign_job_devices(run_cfg: Mapping[str, Any], num_jobs: int) -> List[tuple[str, str]]:
    if num_jobs <= 0:
        return []
    gpu_ids = resolve_gpu_ids(run_cfg)
    device = resolve_device(run_cfg, gpu_ids)
    explicit_device = str(run_cfg.get("device", "auto")).strip()
    gpu_list = split_gpu_ids(gpu_ids)
    if explicit_device and explicit_device.lower() != "auto":
        return [(gpu_ids, device) for _ in range(num_jobs)]
    if not gpu_list:
        return [("", device) for _ in range(num_jobs)]
    if num_jobs == 1:
        return [(gpu_ids, device)]
    return [(gpu_list[idx % len(gpu_list)], f"cuda:{gpu_list[idx % len(gpu_list)]}") for idx in range(num_jobs)]


def checkpoint_tag(model_cfg: Mapping[str, Any]) -> str:
    if model_cfg["kind"] == "official":
        return "official_baseline"
    return Path(str(model_cfg["checkpoint"])).stem


def build_model_env(model_cfg: Mapping[str, Any]) -> Dict[str, str]:
    if model_cfg.get("kind") != "pi3":
        return {}
    env = {
        "PI3_ROOT": str(model_cfg.get("pi3_root", "")),
    }
    for cfg_key, env_key in (
        ("pi3_model_impl", "PI3_MODEL_IMPL"),
        ("pi3_config", "PI3_CONFIG"),
        ("pi3_native_root", "PI3_NATIVE_ROOT"),
    ):
        value = str(model_cfg.get(cfg_key, "")).strip()
        if value:
            env[env_key] = value
    return env


def task_config(config: Mapping[str, Any], task_id: str) -> Dict[str, Any]:
    return dict(config["tasks"].get(task_id, {}))


def dataset_config(config: Mapping[str, Any], dataset_name: str) -> Dict[str, Any]:
    return dict(config["datasets"].get(dataset_name, {}))


def task_value(config: Mapping[str, Any], task_id: str, key: str, default: Any = None) -> Any:
    spec = TASK_REGISTRY[task_id]
    task_cfg = config["tasks"].get(task_id, {})
    dataset_cfg = config["datasets"].get(spec.dataset, {})
    if key in task_cfg:
        return task_cfg[key]
    if key in dataset_cfg:
        return dataset_cfg[key]
    return default


def ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return [str(value)]


def add_flag(cmd: List[str], flag: str, value: Any, repo_root: Path | None = None) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    text = str(value).strip()
    if not text:
        return
    if repo_root is not None and (flag.endswith("root") or flag.endswith("dir") or flag.endswith("path") or flag.endswith("file") or flag == "--output"):
        text = maybe_resolve_path(text, repo_root)
    cmd.extend([flag, text])


def select_task_ids(config: Mapping[str, Any]) -> List[str]:
    selected = []
    for task_id, spec in TASK_REGISTRY.items():
        if not bool(config["tasks"].get(task_id, {}).get("enabled", False)):
            continue
        selected.append(task_id)
    return selected


def build_re10k_model_args(model_cfg: Mapping[str, Any]) -> List[str]:
    if model_cfg["kind"] == "official":
        return [
            "--official-ckpt-root",
            str(model_cfg["official_ckpt_root"]),
            "--model-tag",
            "vggt_official",
        ]
    if model_cfg["kind"] == "pi3":
        raise ValueError("re10k_pose in unified runner only supports VGGT; use eval_pi3_re10k_pose_official.py for Pi3.")
    return [
        "--model-path",
        str(model_cfg["checkpoint"]),
        "--config",
        str(model_cfg["config"]),
        "--model-tag",
        checkpoint_tag(model_cfg),
    ]


def build_pose_model_args(model_cfg: Mapping[str, Any]) -> List[str]:
    if model_cfg["kind"] == "pi3":
        args = [
            "--model-family",
            "pi3",
            "--model-path",
            str(model_cfg["checkpoint"]),
        ]
        for key, flag in (
            ("pi3_model_impl", "--pi3-model-impl"),
            ("pi3_config", "--pi3-config"),
            ("pi3_native_root", "--pi3-native-root"),
        ):
            value = str(model_cfg.get(key, "")).strip()
            if value:
                args.extend([flag, value])
        return args
    args = ["--model-family", "vggt"]
    if model_cfg["kind"] == "official":
        args.extend(
            [
                "--vggt-model-tag",
                "official",
                "--vggt-official-ckpt-root",
                str(model_cfg["official_ckpt_root"]),
            ]
        )
    else:
        args.extend(
            [
                "--model-path",
                str(model_cfg["checkpoint"]),
                "--vggt-model-tag",
                "finetuned",
                "--vggt-config",
                str(model_cfg["config"]),
            ]
        )
    return args


def build_depth_model_args(model_cfg: Mapping[str, Any]) -> List[str]:
    if model_cfg["kind"] == "pi3":
        return [
            "--model-family",
            "pi3",
            "--ckpt",
            str(model_cfg["checkpoint"]),
        ]
    args = ["--model-family", "vggt"]
    if model_cfg["kind"] == "official":
        args.extend(
            [
                "--vggt-model-tag",
                "official",
                "--vggt-official-ckpt-root",
                str(model_cfg["official_ckpt_root"]),
            ]
        )
    else:
        args.extend(
            [
                "--vggt-model-tag",
                "pt34",
                "--vggt-pt34-ckpt",
                str(model_cfg["checkpoint"]),
                "--vggt-config",
                str(model_cfg["config"]),
            ]
        )
    return args


def make_job(*, task: str, family: str, dataset: str, variant: str, command: List[str], env: Dict[str, str], summary_path: str, summary_glob: str = "") -> Dict[str, Any]:
    job = {
        "task": task,
        "family": family,
        "dataset": dataset,
        "variant": variant,
        "command": command,
        "env": env,
        "summary_path": summary_path,
    }
    if summary_glob:
        job["summary_glob"] = summary_glob
    return job


def build_re10k_pose_job(config: Mapping[str, Any], task_id: str, repo_root: Path, gpu_ids: str, device: str) -> Dict[str, Any]:
    run_cfg = config["run"]
    model_cfg = config["model"]
    output_dir = Path(run_cfg["output_root"]) / task_id
    summary_path = str(output_dir / "re10k_pose_lightweight_summary.json")
    root = maybe_resolve_path(task_value(config, task_id, "root", DEFAULT_RE10K_ROOT), repo_root) or DEFAULT_RE10K_ROOT
    if model_cfg["kind"] == "pi3":
        command = [
            str(run_cfg["python_bin"]),
            str(repo_root / "aidi/scripts/baselines/eval_pi3_re10k_pose_official.py"),
            "--re10k-root",
            root,
            "--seed",
            str(task_value(config, task_id, "paper10_seed", 20260215)),
            "--pool-size",
            str(task_value(config, task_id, "paper10_per_scene", 10)),
            "--n-srcs",
            str(task_value(config, task_id, "val_n_srcs", 9)),
            "--limit-scenes",
            str(task_value(config, task_id, "limit_scenes", 0)),
            "--load-img-size",
            str(task_value(config, task_id, "load_img_size", 518)),
            "--device",
            device,
            "--model-path",
            str(model_cfg["checkpoint"]),
            "--model-tag",
            checkpoint_tag(model_cfg),
            "--output",
            summary_path,
        ]
    else:
        command = [
            str(run_cfg["python_bin"]),
            str(repo_root / "aidi/scripts/vggt/eval_vggt_re10k_pose_lightweight.py"),
            "--re10k-root",
            root,
            "--seed",
            str(task_value(config, task_id, "paper10_seed", 20260215)),
            "--pool-size",
            str(task_value(config, task_id, "paper10_per_scene", 10)),
            "--n-srcs",
            str(task_value(config, task_id, "val_n_srcs", 9)),
            "--limit-scenes",
            str(task_value(config, task_id, "limit_scenes", 0)),
            "--load-img-size",
            str(task_value(config, task_id, "load_img_size", 518)),
            "--output",
            summary_path,
        ]
        gpu_list = [item.strip() for item in gpu_ids.split(",") if item.strip()]
        if len(gpu_list) > 1:
            command.extend(["--devices", ",".join(f"cuda:{gpu}" for gpu in gpu_list)])
        else:
            command.extend(["--device", device])
        command.extend(build_re10k_model_args(model_cfg))
    command.extend(ensure_list(task_value(config, task_id, "extra_args", [])))
    spec = TASK_REGISTRY[task_id]
    return make_job(
        task=task_id,
        family=spec.family,
        dataset=spec.dataset,
        variant=spec.variant,
        command=command,
        env=build_model_env(model_cfg),
        summary_path=summary_path,
    )


def build_co3dv2_pose_job(config: Mapping[str, Any], task_id: str, repo_root: Path, device: str) -> Dict[str, Any]:
    run_cfg = config["run"]
    model_cfg = config["model"]
    output_dir = Path(run_cfg["output_root"]) / task_id
    summary_path = maybe_resolve_path(task_value(config, task_id, "output", ""), repo_root) or str(output_dir / "co3dv2_official_summary.json")
    if model_cfg["kind"] == "pi3":
        image_root = (
            maybe_resolve_path(task_value(config, task_id, "shared_image_root", ""), repo_root)
            or maybe_resolve_path(task_value(config, task_id, "image_root", ""), repo_root)
        )
        anno_dir = maybe_resolve_path(task_value(config, task_id, "anno_dir", ""), repo_root)
        if not image_root:
            raise ValueError(f"{task_id} requires shared_image_root or image_root for Pi3 eval")
        if not anno_dir:
            raise ValueError(f"{task_id} requires anno_dir for Pi3 eval")
        command = [
            str(run_cfg["python_bin"]),
            str(repo_root / "aidi/scripts/baselines/eval_pi3_co3d_pose_official.py"),
            "--co3d-image-root",
            image_root,
            "--co3d-anno-dir",
            anno_dir,
            "--device",
            device,
            "--load-img-size",
            str(task_value(config, task_id, "load_img_size", 518)),
            "--num-frames",
            str(task_value(config, task_id, "num_frames", 10)),
            "--seed",
            str(task_value(config, task_id, "seed", 0)),
            "--model-path",
            str(model_cfg["checkpoint"]),
            "--model-tag",
            checkpoint_tag(model_cfg),
            "--output",
            summary_path,
        ]
        if bool(task_value(config, task_id, "fast_eval", False)):
            command.append("--fast-eval")
        categories = str(task_value(config, task_id, "categories", "")).strip()
        if categories:
            command.extend(["--categories", categories])
        command.extend(ensure_list(task_value(config, task_id, "extra_args", [])))
        spec = TASK_REGISTRY[task_id]
        return make_job(
            task=task_id,
            family=spec.family,
            dataset=spec.dataset,
            variant=spec.variant,
            command=command,
            env=build_model_env(model_cfg),
            summary_path=summary_path,
        )
    env = {
        "DEVICE": device,
        "OUTPUT": summary_path,
        "LOAD_IMG_SIZE": str(task_value(config, task_id, "load_img_size", 518)),
        "SELECTION_SOURCE": str(task_value(config, task_id, "selection_source", "co3d_setlists")),
        "SUBSET": str(task_value(config, task_id, "subset", "seen41")),
        "SPLIT": str(task_value(config, task_id, "split", "test")),
        "NUM_FRAMES": str(task_value(config, task_id, "num_frames", 10)),
        "SEED": str(task_value(config, task_id, "seed", 0)),
        "OFFICIAL_CKPT_ROOT": str(model_cfg["official_ckpt_root"]),
    }
    for key, env_key in (
        ("image_root", "CO3D_IMAGE_ROOT"),
        ("setlist_root", "CO3D_SETLIST_ROOT"),
        ("anno_dir", "ANNO_DIR"),
        ("shared_image_root", "CO3D_SHARED_IMAGE_ROOT"),
    ):
        value = maybe_resolve_path(task_value(config, task_id, key, ""), repo_root)
        if value:
            env[env_key] = value
    if model_cfg["kind"] == "custom":
        env["CHECKPOINT"] = str(model_cfg["checkpoint"])
        env["CONFIG"] = str(model_cfg["config"])
    for item in ensure_list(task_value(config, task_id, "extra_env", [])):
        if "=" in item:
            key, value = item.split("=", 1)
            env[key] = value
    spec = TASK_REGISTRY[task_id]
    return make_job(
        task=task_id,
        family=spec.family,
        dataset=spec.dataset,
        variant=spec.variant,
        command=["bash", str(repo_root / "aidi/scripts/vggt/run_co3dv2_official_upstream.sh")],
        env=env,
        summary_path=summary_path,
    )


def build_da3_pose_job(config: Mapping[str, Any], task_id: str, repo_root: Path, device: str) -> Dict[str, Any]:
    run_cfg = config["run"]
    spec = TASK_REGISTRY[task_id]
    output_dir = Path(run_cfg["output_root"]) / task_id
    summary_path = str(output_dir / "metric_results" / "summary.json")
    dataset_name = spec.dataset
    command = [
        str(run_cfg["python_bin"]),
        str(repo_root / "aidi/scripts/vggt/eval_da3_pose_benchmark.py"),
        "--datasets",
        dataset_name,
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--max-frames",
        str(task_value(config, task_id, "max_frames", 100)),
    ]
    image_preprocess_style = str(task_value(config, task_id, "image_preprocess_style", "official_vggt")).strip()
    if image_preprocess_style:
        command.extend(["--image-preprocess-style", image_preprocess_style])
    process_res = task_value(config, task_id, "process_res", None)
    if process_res not in (None, ""):
        command.extend(["--process-res", str(process_res)])
    process_res_method = str(task_value(config, task_id, "process_res_method", "")).strip()
    if process_res_method:
        command.extend(["--process-res-method", process_res_method])
    benchmark_root = maybe_resolve_path(task_value(config, task_id, "benchmark_root", ""), repo_root)
    if benchmark_root:
        command.extend(["--benchmark-root", benchmark_root])
    root_value = maybe_resolve_path(task_value(config, task_id, "root", ""), repo_root)
    if root_value:
        root_flag = DEFAULT_DA3_ROOT_FLAGS.get(dataset_name)
        if root_flag:
            command.extend([root_flag, root_value])
    camera_root = maybe_resolve_path(task_value(config, task_id, "camera_root", ""), repo_root)
    if camera_root and dataset_name == "dtu64":
        command.extend(["--dtu64-camera-root", camera_root])
    command.extend(build_pose_model_args(config["model"]))
    command.extend(ensure_list(task_value(config, task_id, "extra_args", [])))
    return make_job(
        task=task_id,
        family=spec.family,
        dataset=spec.dataset,
        variant=spec.variant,
        command=command,
        env=build_model_env(config["model"]),
        summary_path=summary_path,
    )


def build_relpose_job(config: Mapping[str, Any], task_id: str, repo_root: Path, device: str) -> Dict[str, Any]:
    run_cfg = config["run"]
    spec = TASK_REGISTRY[task_id]
    output_dir = Path(run_cfg["output_root"]) / task_id
    output_path = output_dir / "metric_results" / f"{spec.dataset}_pose.json"
    command = [
        str(run_cfg["python_bin"]),
        str(repo_root / "aidi/scripts/vggt/eval_relpose_1500_benchmark.py"),
        "--dataset",
        spec.dataset,
        "--output",
        str(output_path),
        "--device",
        device,
    ]
    dataset_root = maybe_resolve_path(task_value(config, task_id, "dataset_root", ""), repo_root)
    manifest_root = maybe_resolve_path(task_value(config, task_id, "manifest_root", ""), repo_root)
    add_flag(command, "--dataset-root", dataset_root)
    add_flag(command, "--manifest-root", manifest_root)
    pair_limit = task_value(config, task_id, "pair_limit", 0)
    if int(pair_limit) > 0:
        command.extend(["--pair-limit", str(pair_limit)])
    command.extend(build_pose_model_args(config["model"]))
    command.extend(ensure_list(task_value(config, task_id, "extra_args", [])))
    return make_job(
        task=task_id,
        family=spec.family,
        dataset=spec.dataset,
        variant=spec.variant,
        command=command,
        env=build_model_env(config["model"]),
        summary_path=str(output_path),
    )


def build_relpose_distance_job(config: Mapping[str, Any], task_id: str, repo_root: Path, device: str) -> Dict[str, Any]:
    run_cfg = config["run"]
    spec = TASK_REGISTRY[task_id]
    output_dir = Path(run_cfg["output_root"]) / task_id
    summary_path = str(output_dir / "summary.json")
    protocol_dataset = spec.mode or spec.dataset
    command = [
        str(run_cfg["python_bin"]),
        str(repo_root / "aidi/scripts/baselines/eval_pi3_relpose_distance_protocol.py"),
        "--datasets",
        protocol_dataset,
        "--output-dir",
        str(output_dir),
        "--device",
        device,
    ]
    root_value = maybe_resolve_path(task_value(config, task_id, "root", ""), repo_root)
    root_flag = DEFAULT_RELPOSE_DISTANCE_ROOT_FLAGS.get(spec.dataset)
    if root_flag and root_value:
        command.extend([root_flag, root_value])
    for flag, key in (
        ("--load-img-size", "load_img_size"),
        ("--pose-eval-stride", "pose_eval_stride"),
        ("--image-load-retries", "image_load_retries"),
        ("--image-load-retry-sleep", "image_load_retry_sleep"),
        ("--limit-seqs", "limit_seqs"),
        ("--eval-frame-indices", "eval_frame_indices"),
    ):
        value = task_value(config, task_id, key, None)
        if value not in (None, "", 0):
            command.extend([flag, str(value)])
    if bool(task_value(config, task_id, "skip_plot", False)):
        command.append("--skip-plot")
    if bool(task_value(config, task_id, "require_official_layout", False)):
        command.append("--require-official-layout")
    if bool(task_value(config, task_id, "verbose", False)):
        command.append("--verbose")
    command.extend(build_pose_model_args(config["model"]))
    command.extend(ensure_list(task_value(config, task_id, "extra_args", [])))
    return make_job(
        task=task_id,
        family=spec.family,
        dataset=spec.dataset,
        variant=spec.variant,
        command=command,
        env=build_model_env(config["model"]),
        summary_path=summary_path,
    )


def build_depth_job(config: Mapping[str, Any], task_id: str, repo_root: Path, device: str) -> Dict[str, Any]:
    run_cfg = config["run"]
    spec = TASK_REGISTRY[task_id]
    output_dir = Path(run_cfg["output_root"]) / task_id
    data_root = resolve_data_root_for_task(config, task_id, repo_root)
    if not data_root:
        raise ValueError(f"{task_id} requires datasets.{spec.dataset}.data_root or tasks.{task_id}.data_root")
    summary_path = str(output_dir / f"{spec.dataset}_{spec.mode}_protocol_summary.json")
    command = [
        str(run_cfg["python_bin"]),
        str(repo_root / "aidi/scripts/baselines/eval_pi3_depth_protocol.py"),
        "--mode",
        spec.mode,
        "--dataset",
        spec.dataset,
        "--data-root",
        data_root,
        "--output-dir",
        str(output_dir),
        "--device",
        device,
    ]
    eval_device = str(task_value(config, task_id, "eval_device", "")).strip()
    if eval_device:
        command.extend(["--eval-device", eval_device])
    layout = str(task_value(config, task_id, "layout", "")).strip()
    if layout:
        command.extend(["--layout", layout])
    camera = str(task_value(config, task_id, "camera", "")).strip()
    if camera:
        command.extend(["--camera", camera])
    for flag, key in (
        ("--load-img-size", "load_img_size"),
        ("--max-size", "max_size"),
        ("--align-size", "align_size"),
        ("--safe-bound", "safe_bound"),
        ("--preprocess-style", "preprocess_style"),
        ("--eval-gt-shape", "eval_gt_shape"),
        ("--alignment", "alignment"),
        ("--max-seqs", "max_seqs"),
        ("--max-frames-per-seq", "max_frames_per_seq"),
        ("--eval-frame-indices", "eval_frame_indices"),
        ("--vggt-input-preprocess", "vggt_input_preprocess"),
    ):
        value = task_value(config, task_id, key, None)
        if (
            key == "vggt_input_preprocess"
            and value in (None, "", 0)
            and spec.mode == "videodepth"
            and config["model"]["kind"] in {"official", "custom"}
        ):
            value = "official"
        if value not in (None, "", 0):
            command.extend([flag, str(value)])
    if spec.mode == "videodepth":
        for flag, key in (
            ("--vggt-topk-override", "vggt_topk_override"),
            ("--vggt-short-topk-override", "vggt_short_topk_override"),
            ("--vggt-long-topk-override", "vggt_long_topk_override"),
            ("--vggt-long-topk-min-frames", "vggt_long_topk_min_frames"),
            ("--vggt-dtype-override", "vggt_dtype_override"),
        ):
            value = task_value(config, task_id, key, None)
            if value not in (None, "", 0):
                command.extend([flag, str(value)])
    if bool(task_value(config, task_id, "eval_resize_to_1036p", False)):
        command.append("--eval-resize-to-1036p")
    command.extend(build_depth_model_args(config["model"]))
    command.extend(ensure_list(task_value(config, task_id, "extra_args", [])))
    env = build_model_env(config["model"])
    for item in ensure_list(task_value(config, task_id, "extra_env", [])):
        if "=" in item:
            key, value = item.split("=", 1)
            env[key] = value
    return make_job(
        task=task_id,
        family=spec.family,
        dataset=spec.dataset,
        variant=spec.variant,
        command=command,
        env=env,
        summary_path=summary_path,
    )


def build_pointcloud_job(config: Mapping[str, Any], task_id: str, repo_root: Path, device: str) -> Dict[str, Any]:
    run_cfg = config["run"]
    spec = TASK_REGISTRY[task_id]
    output_dir = maybe_resolve_path(task_value(config, task_id, "output_dir", ""), repo_root) or str(Path(run_cfg["output_root"]) / task_id)
    env = {
        "DEVICE": device,
        "MODEL_FAMILY": "pi3" if config["model"]["kind"] == "pi3" else "vggt",
        "MODEL_TAG": checkpoint_tag(config["model"]) if config["model"]["kind"] == "pi3" else ("official" if config["model"]["kind"] == "official" else "custom"),
        "OUTPUT_DIR": output_dir,
        "POINT_SOURCE": str(task_value(config, task_id, "point_source", "native")),
        "LOAD_IMG_SIZE": str(task_value(config, task_id, "load_img_size", 518)),
        "MAX_SEQUENCES": str(task_value(config, task_id, "max_sequences", 0)),
        "VGGT_OFFICIAL_CKPT_ROOT": str(config["model"]["official_ckpt_root"]),
        "PI3_ROOT": str(config["model"]["pi3_root"]),
    }
    env.update(build_model_env(config["model"]))
    if spec.mode:
        env["PROTOCOL"] = spec.mode
    dataset_root = maybe_resolve_path(task_value(config, task_id, "dataset_root", ""), repo_root)
    if dataset_root:
        env["DATASET_ROOT"] = dataset_root
    if spec.dataset in {"eth3d", "7scenes", "nrgbd"}:
        seq_map = maybe_resolve_path(task_value(config, task_id, "seq_map", ""), repo_root)
        if seq_map:
            env["SEQ_MAP"] = seq_map
    if config["model"]["kind"] == "custom":
        env["VGGT_CHECKPOINT"] = str(config["model"]["checkpoint"])
        env["VGGT_CONFIG"] = str(config["model"]["config"])
    if config["model"]["kind"] == "pi3":
        env["PI3_CKPT"] = str(config["model"]["checkpoint"])
    for item in ensure_list(task_value(config, task_id, "extra_env", [])):
        if "=" in item:
            key, value = item.split("=", 1)
            env[key] = value
    script_name_map = {
        "dtu": "run_eval_dtu_mv_recon_pi3_style.sh",
        "eth3d": "run_eval_eth3d_mv_recon_pi3_style.sh",
        "7scenes": "run_eval_7scenes_mv_recon_pi3_style.sh",
        "nrgbd": "run_eval_nrgbd_mv_recon_pi3_style.sh",
    }
    try:
        script_name = script_name_map[spec.dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported pointcloud dataset={spec.dataset}") from exc
    return make_job(
        task=task_id,
        family=spec.family,
        dataset=spec.dataset,
        variant=spec.variant,
        command=["bash", str(repo_root / "aidi/scripts/vggt" / script_name)],
        env=env,
        summary_path=str(Path(output_dir) / "summary.json"),
    )


def build_job(config: Mapping[str, Any], task_id: str, repo_root: Path, gpu_ids: str, device: str) -> Dict[str, Any]:
    spec = TASK_REGISTRY[task_id]
    if spec.adapter == "re10k_pose_lightweight":
        return build_re10k_pose_job(config, task_id, repo_root, gpu_ids, device)
    if spec.adapter == "co3dv2_pose_upstream":
        return build_co3dv2_pose_job(config, task_id, repo_root, device)
    if spec.adapter == "da3_pose":
        return build_da3_pose_job(config, task_id, repo_root, device)
    if spec.adapter == "relpose1500":
        return build_relpose_job(config, task_id, repo_root, device)
    if spec.adapter == "relpose_distance":
        return build_relpose_distance_job(config, task_id, repo_root, device)
    if spec.adapter == "depth_protocol":
        return build_depth_job(config, task_id, repo_root, device)
    if spec.adapter == "pointcloud_pi3_style":
        return build_pointcloud_job(config, task_id, repo_root, device)
    raise ValueError(f"Unsupported adapter={spec.adapter} for task={task_id}")


def build_jobs(config: Mapping[str, Any], repo_root: Path | None = None) -> List[Dict[str, Any]]:
    jobs, _ = plan_jobs(config, repo_root=repo_root)
    return jobs


def plan_jobs(
    config: Mapping[str, Any], repo_root: Path | None = None
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    repo_root = repo_root or REPO_ROOT
    task_ids = select_task_ids(config)
    device_slots = assign_job_devices(config["run"], len(task_ids))
    jobs: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for task_id, (gpu_ids, device) in zip(task_ids, device_slots):
        reason = auto_skip_reason(config, task_id, gpu_ids, device)
        if reason:
            skipped_record = make_skipped_record(task_id, reason=reason, gpu_ids=gpu_ids, device=device)
            skipped.append(skipped_record)
            print(f"[unified-eval] auto-skip task={task_id} device={device} reason={reason}")
            continue
        job = build_job(config, task_id, repo_root, gpu_ids, device)
        job["assigned_gpu_ids"] = gpu_ids
        job["assigned_device"] = device
        jobs.append(job)
    return jobs, skipped


def load_yaml_config(path: Path, seen: set[Path] | None = None) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    seen = seen or set()
    if path in seen:
        chain = " -> ".join(str(item) for item in list(seen) + [path])
        raise ValueError(f"Cyclic unified-eval config extends detected: {chain}")
    seen.add(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    extends = raw.pop("extends", None)
    if not extends:
        return raw
    base_path = Path(str(extends)).expanduser()
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    base = load_yaml_config(base_path, seen=seen)
    return deep_update(base, raw)


def parse_env_overrides(value: Any) -> Dict[str, str]:
    if not value:
        return {}
    if isinstance(value, Mapping):
        return {str(key): str(item) for key, item in value.items() if str(key)}
    env: Dict[str, str] = {}
    for item in ensure_list(value):
        if "=" not in item:
            raise ValueError(f"Invalid env override {item!r}; expected KEY=VALUE")
        key, env_value = item.split("=", 1)
        key = key.strip()
        if key:
            env[key] = env_value
    return env


def read_summary(job: Mapping[str, Any]) -> Dict[str, Any]:
    summary_path = str(job["summary_path"])
    if "summary_glob" in job:
        matches = sorted(glob.glob(str(job["summary_glob"])))
        if not matches:
            raise FileNotFoundError(f"Summary not found for {job['task']}: {job['summary_glob']}")
        summary_path = matches[-1]
    with open(summary_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "task": job["task"],
        "family": job["family"],
        "dataset": job["dataset"],
        "variant": job["variant"],
        "summary_path": summary_path,
        "summary": extract_summary(job["task"], payload),
        "payload": payload,
    }


def extract_summary(task_id: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    spec = TASK_REGISTRY[task_id]
    if "summary" in payload and isinstance(payload["summary"], Mapping):
        return dict(payload["summary"])
    if spec.adapter == "da3_pose":
        datasets = payload.get("datasets", {})
        if isinstance(datasets, Mapping) and spec.dataset in datasets and isinstance(datasets[spec.dataset], Mapping):
            return dict(datasets[spec.dataset])
    if spec.adapter == "relpose_distance":
        results = payload.get("results", [])
        if isinstance(results, Sequence):
            protocol_dataset = spec.mode or spec.dataset
            for item in results:
                if not isinstance(item, Mapping):
                    continue
                if item.get("dataset") == protocol_dataset and isinstance(item.get("summary"), Mapping):
                    return dict(item["summary"])
    return dict(payload)


def make_error_record(job: Mapping[str, Any], exc: BaseException) -> Dict[str, Any]:
    message = f"{type(exc).__name__}: {exc}"
    return {
        "task": job["task"],
        "family": job["family"],
        "dataset": job["dataset"],
        "variant": job["variant"],
        "status": "ERROR",
        "error": message,
        "summary_path": str(job.get("summary_path", "")),
        "summary": {},
        "payload": {
            "status": "ERROR",
            "error": message,
            "command": [str(item) for item in job.get("command", [])],
            "assigned_gpu_ids": str(job.get("assigned_gpu_ids", "")),
            "assigned_device": str(job.get("assigned_device", "")),
        },
    }


def summary_exists(job: Mapping[str, Any]) -> bool:
    if "summary_glob" in job:
        return bool(glob.glob(str(job["summary_glob"])))
    return Path(str(job["summary_path"])).is_file()


def canonical_metric_brief(record: Mapping[str, Any]) -> str:
    status = str(record.get("status", "")).strip()
    if status and status.upper() != "SUCCESS":
        reason = str(record.get("skip_reason", "") or record.get("error", "")).strip()
        return f"{status}{(': ' + reason) if reason else ''}"
    summary = record.get("summary", {}) or {}
    wanted_groups = (
        ("Auc30", "Auc3", "AUC@20", "AUC@10", "AUC@5"),
        ("overall", "acc", "comp", "precision", "recall"),
        ("AbsRel", "RMSE", "delta1", "d1", "a1"),
        ("ATE", "RPE trans", "RPE rot", "ATE_trans", "RPE_trans", "RPE_rot"),
    )
    parts: List[str] = []
    for group in wanted_groups:
        for key in group:
            if key in summary and isinstance(summary[key], (int, float)):
                parts.append(f"{key}={float(summary[key]):.4f}")
        if parts:
            break
    if not parts:
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                parts.append(f"{key}={float(value):.4f}")
            if len(parts) >= 3:
                break
    return ", ".join(parts)


def render_summary_markdown(records: Sequence[Mapping[str, Any]]) -> str:
    lines: List[str] = ["# Unified Eval Summary", ""]
    for family in ALL_FAMILIES:
        family_records = [record for record in records if record["family"] == family]
        if not family_records:
            continue
        lines.append(f"## {family}")
        lines.append("| Task | Dataset | Variant | Summary |")
        lines.append("|---|---|---|---|")
        for record in family_records:
            lines.append(
                f"| `{record['task']}` | `{record['dataset']}` | `{record['variant']}` | {canonical_metric_brief(record)} |"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_aggregate_artifacts(*, output_root: Path, run_name: str, records: Sequence[Mapping[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": run_name,
        "output_root": str(output_root),
        "tasks": list(records),
    }
    (output_root / "aggregate_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "aggregate_summary.md").write_text(
        render_summary_markdown(records),
        encoding="utf-8",
    )


def execute_jobs(
    config: Mapping[str, Any],
    jobs: Iterable[Mapping[str, Any]],
    repo_root: Path | None = None,
    skipped_records: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    repo_root = repo_root or REPO_ROOT
    run_cfg = config["run"]
    output_root = Path(str(run_cfg["output_root"]))
    dry_run = bool(run_cfg.get("dry_run"))
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
    jobs_list = list(jobs)
    records: List[Dict[str, Any]] = [dict(item) for item in (skipped_records or [])]

    for job in jobs_list:
        print(
            f"[unified-eval] task={job['task']} family={job['family']} dataset={job['dataset']} "
            f"device={job.get('assigned_device', '')}"
        )
        print("[unified-eval] command:", " ".join(job["command"]))
    if dry_run:
        planned_records = records + [
            {
                "task": str(job["task"]),
                "family": str(job["family"]),
                "dataset": str(job["dataset"]),
                "variant": str(job["variant"]),
                "status": "PLANNED",
                "device": str(job.get("assigned_device", "")),
                "output_dir": str(job.get("output_dir", "")),
                "command": [str(item) for item in job["command"]],
            }
            for job in jobs_list
        ]
        aggregate = {
            "run_name": str(run_cfg["name"]),
            "output_root": str(output_root),
            "tasks": planned_records,
        }
        print(json.dumps(aggregate, indent=2, ensure_ascii=False))
        return aggregate

    if not jobs_list:
        write_aggregate_artifacts(output_root=output_root, run_name=str(run_cfg["name"]), records=records)
        aggregate = {
            "run_name": str(run_cfg["name"]),
            "output_root": str(output_root),
            "tasks": records,
        }
        print(json.dumps(aggregate, indent=2, ensure_ascii=False))
        return aggregate

    def run_one(job: Mapping[str, Any]) -> Dict[str, Any]:
        env = os.environ.copy()
        env.update(parse_env_overrides(run_cfg.get("env", {})))
        env.update({key: str(value) for key, value in job["env"].items() if str(value)})
        if bool(run_cfg.get("skip_existing")) and summary_exists(job):
            print(f"[unified-eval] skip existing summary for {job['task']}")
        else:
            subprocess.run(list(job["command"]), cwd=repo_root, env=env, check=True)
        return read_summary(job)

    jobs_by_device: Dict[str, List[tuple[int, Mapping[str, Any]]]] = {}
    for idx, job in enumerate(jobs_list):
        jobs_by_device.setdefault(str(job.get("assigned_device", "")), []).append((idx, job))
    worker_count = max(1, len(jobs_by_device))
    future_map: Dict[concurrent.futures.Future[List[tuple[int, Dict[str, Any]]]], str] = {}
    results_by_index: Dict[int, Dict[str, Any]] = {}
    errors: List[tuple[int, str, BaseException]] = []

    def run_device_queue(device_jobs: Sequence[tuple[int, Mapping[str, Any]]]) -> List[tuple[int, Dict[str, Any]]]:
        bucket_records: List[tuple[int, Dict[str, Any]]] = []
        for idx, job in device_jobs:
            try:
                bucket_records.append((idx, run_one(job)))
            except BaseException as exc:
                if not bool(run_cfg.get("continue_on_error", False)):
                    raise
                print(f"[unified-eval] task={job['task']} failed but continue_on_error=true: {type(exc).__name__}: {exc}", file=sys.stderr)
                bucket_records.append((idx, make_error_record(job, exc)))
        return bucket_records

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        for device, device_jobs in jobs_by_device.items():
            future = executor.submit(run_device_queue, device_jobs)
            future_map[future] = device
        for future in concurrent.futures.as_completed(future_map):
            try:
                for idx, record in future.result():
                    results_by_index[idx] = record
            except BaseException as exc:
                device = future_map[future]
                for idx, job in jobs_by_device[device]:
                    if idx not in results_by_index:
                        errors.append((idx, str(job["task"]), exc))

    if errors:
        failed_tasks = ", ".join(f"{task}: {exc}" for _, task, exc in errors)
        raise RuntimeError(f"Unified eval failed for tasks: {failed_tasks}")

    records.extend(results_by_index[idx] for idx in range(len(jobs_list)))

    write_aggregate_artifacts(output_root=output_root, run_name=str(run_cfg["name"]), records=records)
    aggregate = {
        "run_name": str(run_cfg["name"]),
        "output_root": str(output_root),
        "tasks": records,
    }
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return aggregate


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified VGGT evaluation tasks across pose/depth/pointcloud.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "aidi/configs/vggt/unified_eval_local.yaml"),
        help="Unified eval YAML config path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only without executing them.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available task ids and exit.",
    )
    parser.add_argument(
        "--gpu-ids",
        default="",
        help="Override run.gpu_ids from config, e.g. '0,1,2'.",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="Override run.output_root from config.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Override run.name from config.",
    )
    return parser.parse_args(list(argv))


def print_task_list() -> None:
    print("task,family,dataset,variant")
    for task_id, spec in TASK_REGISTRY.items():
        print(f"{task_id},{spec.family},{spec.dataset},{spec.variant}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.list_tasks:
        print_task_list()
        return 0
    config_path = Path(args.config).expanduser()
    raw_config = load_yaml_config(config_path)
    config = normalize_config(raw_config, repo_root=REPO_ROOT)
    if args.gpu_ids:
        config["run"]["gpu_ids"] = args.gpu_ids
    if args.output_root:
        config["run"]["output_root"] = resolve_path(args.output_root, REPO_ROOT)
    if args.run_name:
        config["run"]["name"] = args.run_name
    if args.dry_run:
        config["run"]["dry_run"] = True
    jobs, skipped_records = plan_jobs(config, repo_root=REPO_ROOT)
    execute_jobs(config, jobs, repo_root=REPO_ROOT, skipped_records=skipped_records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
