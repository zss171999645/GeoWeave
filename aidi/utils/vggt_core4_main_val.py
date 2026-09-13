from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
import os
import random
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from aidi.utils.core4_model_adapter import (
    PreparedCore4Model,
    looks_like_native_pi3_model,
    prepare_live_model_for_core4_eval,
    restore_live_model_after_core4_eval,
    unwrap_official_vggt_model,
    unwrap_training_model,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - allows lightweight config/unit tests without torch installed
    torch = None


CORE4_DATASET_ORDER: Tuple[str, ...] = ("re10k", "co3dv2", "dtu", "eth3d")
CO3DV2_RAW_ROOT_DEFAULT = "/horizon-bucket/saturn_v_4dlabel/009_geo/002_data/dust3r_datasets/dust3r_extracted/co3dv2"
CO3DV2_SETLIST_MIRROR_ROOTS: Tuple[str, ...] = (
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/eval_datasets/"
    "co3d_official_image_mirror_co3d_setlists_seen41_test",
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/eval_datasets/"
    "co3d_official_image_mirror_co3d_setlists_seen41_test_full",
)


def _require_torch() -> Any:
    if torch is None:
        raise ModuleNotFoundError("torch is required for core4 main validation runtime")
    return torch


def normalize_core4_main_val_cfg(raw_cfg: Optional[Dict[str, Any]], repo_root: Optional[Path] = None) -> Dict[str, Any]:
    if not raw_cfg:
        return {}

    from aidi.scripts.vggt.run_official_core4_eval import normalize_config

    cfg = normalize_config(copy.deepcopy(raw_cfg), repo_root=repo_root)
    for dataset_name in ("re10k", "co3dv2"):
        ds_cfg = cfg["datasets"].setdefault(dataset_name, {})
        ds_cfg["sample_stride"] = max(1, int(ds_cfg.get("sample_stride", 1)))

    co3dv2_cfg = cfg["datasets"].setdefault("co3dv2", {})
    image_root = str(co3dv2_cfg.get("image_root", "")).rstrip("/")
    selection_source = str(co3dv2_cfg.get("selection_source", "co3d_setlists"))
    if selection_source in {"co3d_setlists", "hf_jgz", "co3d_annotations"} and image_root in CO3DV2_SETLIST_MIRROR_ROOTS:
        co3dv2_cfg["image_root"] = CO3DV2_RAW_ROOT_DEFAULT
    return cfg


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _select_by_stride(items: Sequence[Any], stride: int) -> List[Any]:
    stride = max(1, int(stride))
    if stride <= 1:
        return list(items)
    return [item for index, item in enumerate(items) if index % stride == 0]


def _shard_by_rank(items: Sequence[Any], rank: int, world_size: int) -> List[Any]:
    world_size = max(1, int(world_size))
    rank = int(rank)
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid rank={rank} for world_size={world_size}")
    if world_size == 1:
        return list(items)
    return [item for index, item in enumerate(items) if index % world_size == rank]


def _stable_int_seed(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.sha1(payload).digest()[:4], byteorder="little", signed=False)


def _build_re10k_summary(metrics: Sequence[Dict[str, Any]]) -> Dict[str, float]:
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


def _calculate_auc_np(r_error: np.ndarray, t_error: np.ndarray, max_threshold: int = 30) -> float:
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    if num_pairs <= 0:
        return 0.0
    normalized_histogram = histogram.astype(float) / num_pairs
    return float(np.mean(np.cumsum(normalized_histogram)))


def _summarize_co3dv2_category(sequence_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    r_error = np.array([x for result in sequence_results for x in result["r_error"]], dtype=np.float64)
    t_error = np.array([x for result in sequence_results for x in result["t_error"]], dtype=np.float64)
    auc_30 = _calculate_auc_np(r_error, t_error, max_threshold=30)
    auc_20 = _calculate_auc_np(r_error, t_error, max_threshold=20)
    auc_15 = _calculate_auc_np(r_error, t_error, max_threshold=15)
    auc_10 = _calculate_auc_np(r_error, t_error, max_threshold=10)
    auc_5 = _calculate_auc_np(r_error, t_error, max_threshold=5)
    auc_3 = _calculate_auc_np(r_error, t_error, max_threshold=3)
    return {
        "num_sequences": len(sequence_results),
        "num_pairs_total": int(len(r_error)),
        "AUC_30": auc_30,
        "AUC_20": auc_20,
        "AUC_15": auc_15,
        "AUC_10": auc_10,
        "AUC_5": auc_5,
        "AUC_3": auc_3,
        "cam:pose_auc_10_mean": auc_10,
        "cam:pose_auc_20_mean": auc_20,
        "cam:pose_auc_30_mean": auc_30,
    }


def _mean_category_metric(results: Dict[str, Dict[str, Any]], key: str) -> Optional[float]:
    values = [float(result[key]) for result in results.values() if key in result]
    if not values:
        return None
    return float(np.mean(values))


def _build_co3dv2_summary(
    per_category_results: Dict[str, Dict[str, Any]],
    per_sequence_results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    return {
        "AUC_30_mean": _mean_category_metric(per_category_results, "AUC_30"),
        "AUC_20_mean": _mean_category_metric(per_category_results, "AUC_20"),
        "AUC_15_mean": _mean_category_metric(per_category_results, "AUC_15"),
        "AUC_10_mean": _mean_category_metric(per_category_results, "AUC_10"),
        "AUC_5_mean": _mean_category_metric(per_category_results, "AUC_5"),
        "AUC_3_mean": _mean_category_metric(per_category_results, "AUC_3"),
        "cam:pose_auc_10_mean": _mean_category_metric(per_category_results, "cam:pose_auc_10_mean"),
        "cam:pose_auc_20_mean": _mean_category_metric(per_category_results, "cam:pose_auc_20_mean"),
        "cam:pose_auc_30_mean": _mean_category_metric(per_category_results, "cam:pose_auc_30_mean"),
        "num_categories": len(per_category_results),
        "num_sequences": int(sum(len(results) for results in per_sequence_results.values())),
    }


def _merge_re10k_payloads(payloads: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not payloads:
        raise RuntimeError("No RE10K shard payloads to merge.")
    merged = copy.deepcopy(payloads[0])
    metrics = [item for payload in payloads for item in payload.get("metrics", [])]
    failures = [item for payload in payloads for item in payload.get("failures", [])]
    merged["metrics"] = metrics
    merged["failures"] = failures
    merged["summary"] = _build_re10k_summary(metrics)
    merged["num_shards"] = len(payloads)
    merged["shard_index"] = "merged"
    merged["worker_name"] = "distributed_merged"
    return merged


def _merge_co3dv2_payloads(payloads: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not payloads:
        raise RuntimeError("No CO3Dv2 shard payloads to merge.")
    merged = copy.deepcopy(payloads[0])
    per_sequence_results: Dict[str, List[Dict[str, Any]]] = {}
    failures: List[Dict[str, Any]] = []
    categories: List[str] = []

    for payload in payloads:
        for category in payload.get("categories", []):
            if category not in categories:
                categories.append(category)
        failures.extend(payload.get("failures", []))
        for category, sequence_results in payload.get("per_sequence_results", {}).items():
            per_sequence_results.setdefault(category, []).extend(copy.deepcopy(sequence_results))

    per_category_results = {
        category: _summarize_co3dv2_category(sequence_results)
        for category, sequence_results in per_sequence_results.items()
        if sequence_results
    }
    if not per_category_results:
        raise RuntimeError("No category produced valid CO3Dv2 results across distributed shards.")

    merged["categories"] = categories
    merged["metrics"] = [item for results in per_sequence_results.values() for item in results]
    merged["per_sequence_results"] = per_sequence_results
    merged["per_category_results"] = per_category_results
    merged["summary"] = _build_co3dv2_summary(per_category_results, per_sequence_results)
    merged["failures"] = failures
    return merged


def _merge_weighted_metric_dicts(metric_dicts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    keys = []
    for metrics in metric_dicts:
        for key in metrics.keys():
            if key not in keys:
                keys.append(key)

    merged: Dict[str, Any] = {}
    total_sequences = int(sum(int(metrics.get("num_sequences", 0)) for metrics in metric_dicts))
    for key in keys:
        if key == "num_sequences":
            merged[key] = total_sequences
            continue
        if key == "num_skipped":
            merged[key] = int(sum(int(metrics.get(key, 0)) for metrics in metric_dicts))
            continue

        weighted_sum = 0.0
        weight = 0
        saw_numeric = False
        for metrics in metric_dicts:
            if key not in metrics or not _is_number(metrics[key]):
                continue
            count = int(metrics.get("num_sequences", 0))
            value = float(metrics[key])
            if count <= 0 or math.isnan(value):
                continue
            weighted_sum += value * count
            weight += count
            saw_numeric = True
        if saw_numeric:
            merged[key] = float(weighted_sum / weight) if weight > 0 else float("nan")
        else:
            for metrics in metric_dicts:
                if key in metrics:
                    merged[key] = copy.deepcopy(metrics[key])
                    break
    return merged


def _merge_mv_recon_payloads(dataset_name: str, payloads: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not payloads:
        raise RuntimeError(f"No {dataset_name} shard payloads to merge.")
    merged = copy.deepcopy(payloads[0])
    protocol_names: List[str] = []
    for payload in payloads:
        for protocol in payload.get("protocols", {}).keys():
            if protocol not in protocol_names:
                protocol_names.append(protocol)

    merged_protocols: Dict[str, Dict[str, Any]] = {}
    for protocol in protocol_names:
        protocol_payloads = [
            payload["protocols"][protocol]
            for payload in payloads
            if protocol in payload.get("protocols", {})
        ]
        first = copy.deepcopy(protocol_payloads[0])
        first["metrics"] = _merge_weighted_metric_dicts(
            [payload.get("metrics", {}) for payload in protocol_payloads]
        )
        first["num_seq_in_map"] = int(sum(int(payload.get("num_seq_in_map", 0)) for payload in protocol_payloads))
        first["skipped"] = [
            skipped
            for payload in protocol_payloads
            for skipped in payload.get("skipped", [])
        ]
        first["num_shards"] = len(protocol_payloads)
        merged_protocols[protocol] = first
    merged["protocols"] = merged_protocols
    return merged


def _merge_dataset_payloads(dataset_name: str, payloads: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if dataset_name == "re10k":
        return _merge_re10k_payloads(payloads)
    if dataset_name == "co3dv2":
        return _merge_co3dv2_payloads(payloads)
    if dataset_name in ("dtu", "eth3d"):
        return _merge_mv_recon_payloads(dataset_name, payloads)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _write_sharded_seq_map(
    *,
    seq_map_path: Path,
    rank: int,
    world_size: int,
    max_sequences: int,
    output_dir: Path,
    protocol: str,
) -> Path:
    with seq_map_path.open("r", encoding="utf-8") as f:
        seq_map = json.load(f)

    items = list(seq_map.items())
    if int(max_sequences) > 0:
        items = items[: int(max_sequences)]
    shard_items = _shard_by_rank(items, rank=rank, world_size=world_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"{protocol}_rank{int(rank):04d}_of_{int(world_size):04d}.json"
    with shard_path.open("w", encoding="utf-8") as f:
        json.dump({key: value for key, value in shard_items}, f, indent=2, ensure_ascii=False)
    return shard_path


def _compact_payload(dataset_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if dataset_name in ("re10k", "co3dv2"):
        return {
            "summary": copy.deepcopy(payload.get("summary", {})),
            "num_metrics": len(payload.get("metrics", [])),
            "num_failures": len(payload.get("failures", [])),
        }

    protocols = {}
    for protocol, protocol_payload in payload.get("protocols", {}).items():
        protocols[protocol] = {
            "metrics": copy.deepcopy(protocol_payload.get("metrics", {})),
            "num_seq_in_map": int(protocol_payload.get("num_seq_in_map", 0)),
            "num_skipped": len(protocol_payload.get("skipped", [])),
        }
    return {"protocols": protocols}


def extract_core4_tb_scalars(dataset_name: str, payload: Dict[str, Any]) -> Dict[str, float]:
    scalars: Dict[str, float] = {}
    if dataset_name in ("re10k", "co3dv2"):
        for key, value in payload.get("summary", {}).items():
            if _is_number(value):
                scalars[key] = float(value)
        return scalars

    for protocol, protocol_payload in payload.get("protocols", {}).items():
        metrics = protocol_payload.get("metrics", {})
        for key, value in metrics.items():
            if _is_number(value):
                scalars[f"{protocol}/{key}"] = float(value)
    return scalars


def _save_rng_state() -> Dict[str, Any]:
    torch_mod = _require_torch()
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch_mod.get_rng_state(),
    }
    if torch_mod.cuda.is_available():
        state["cuda"] = torch_mod.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    torch_mod = _require_torch()
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_mod.set_rng_state(state["torch"])
    if "cuda" in state and torch_mod.cuda.is_available():
        torch_mod.cuda.set_rng_state_all(state["cuda"])


def _unwrap_training_model(model: Any) -> Any:
    return unwrap_training_model(model)


def _looks_like_native_pi3_model(model: Any) -> bool:
    return looks_like_native_pi3_model(model)


def _unwrap_official_vggt_model(model: Any) -> Any:
    return unwrap_official_vggt_model(model)


def _prepare_live_model_for_core4_eval(model: Any, global_step: int) -> Tuple[Any, Any, bool, str]:
    prepared = prepare_live_model_for_core4_eval(model, global_step)
    return prepared.live_model, prepared.inference_model, prepared.was_training, prepared.model_family


def _restore_live_model_after_core4_eval(live_model: Any, was_training: bool, global_step: int, model_family: str) -> None:
    prepared = PreparedCore4Model(
        live_model=live_model,
        inference_model=live_model,
        was_training=was_training,
        model_family=model_family,
    )
    restore_live_model_after_core4_eval(prepared, global_step)


def _resolve_model_device(model: Any) -> torch.device:
    _require_torch()
    return next(model.parameters()).device


def _default_output_root(record_dir: str, epoch: int) -> Path:
    return Path(record_dir) / "core4_main_val" / f"epoch_{epoch:04d}"


def _resolve_re10k_scene_roots(root: str, sample_stride: int, limit_scenes: int = 0) -> List[Path]:
    from aidi.scripts.baselines.eval_pi3_re10k_pose_official import resolve_scene_roots

    scene_roots = resolve_scene_roots(Path(root), scene_filter="", limit_scenes=0)
    scene_roots = _select_by_stride(scene_roots, sample_stride)
    if limit_scenes and limit_scenes > 0:
        scene_roots = scene_roots[: int(limit_scenes)]
    return scene_roots


def _load_selected_co3dv2_annotations(ds_cfg: Dict[str, Any], categories: Sequence[str]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    selection_source = str(ds_cfg.get("selection_source", "co3d_setlists"))
    split = str(ds_cfg.get("split", "test"))

    if selection_source != "co3d_setlists":
        from aidi.scripts.vggt import eval_co3d_pose_official_upstream as co3d_eval

        anno_dir = Path(str(ds_cfg["anno_dir"]))
        selected: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for category in categories:
            annotation_file = anno_dir / f"{category}_{split}.jgz"
            if not annotation_file.is_file():
                continue
            selected[category] = co3d_eval.load_annotation(annotation_file)
        return selected

    from aidi.scripts.vggt import resolve_co3dv2_official_scene_roots as resolver

    co3d_v2_dir = Path(str(ds_cfg.get("setlist_root") or ds_cfg.get("co3d_v2_dir") or CO3DV2_RAW_ROOT_DEFAULT))
    set_list_tag = str(ds_cfg.get("set_list_tag", "fewview_dev"))
    min_quality = float(ds_cfg.get("min_quality", 0.5))
    selected: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for category in categories:
        category_dir = co3d_v2_dir / category
        if not category_dir.is_dir():
            continue
        subset_lists_data = resolver._load_json(str(category_dir / "set_lists" / f"set_lists_{set_list_tag}.json"))
        frame_data = resolver._load_jgz(str(category_dir / "frame_annotations.jgz"))
        sequence_data = resolver._load_jgz(str(category_dir / "sequence_annotations.jgz"))

        frame_data_processed: Dict[str, Dict[int, Dict[str, Any]]] = {}
        for frame_item in frame_data:
            sequence_name = frame_item["sequence_name"]
            frame_data_processed.setdefault(sequence_name, {})[int(frame_item["frame_number"])] = frame_item

        good_quality_sequences = {
            seq_item["sequence_name"]
            for seq_item in sequence_data
            if float(seq_item["viewpoint_quality_score"]) > min_quality
        }

        selected_frames_by_scene: Dict[str, List[Dict[str, Any]]] = {}
        for sequence_name, frame_number, filepath in subset_lists_data[split]:
            if sequence_name not in good_quality_sequences:
                continue
            frame_map = frame_data_processed.get(sequence_name, {})
            if int(frame_number) not in frame_map:
                continue
            frame_item = frame_map[int(frame_number)]
            selected_frames_by_scene.setdefault(sequence_name, []).append(
                {
                    "filepath": filepath,
                    "frame_number": int(frame_number),
                    "R": frame_item["viewpoint"]["R"],
                    "T": frame_item["viewpoint"]["T"],
                    "focal_length": frame_item["viewpoint"]["focal_length"],
                    "principal_point": frame_item["viewpoint"]["principal_point"],
                }
            )
        selected[category] = selected_frames_by_scene

    return selected


def _filter_co3dv2_seq_names_with_image_dirs(
    seq_names: Sequence[str],
    *,
    image_root: str,
    category: str,
) -> List[str]:
    root = Path(str(image_root))
    filtered: List[str] = []
    for seq_name in seq_names:
        if (root / category / seq_name).is_dir():
            filtered.append(seq_name)
    return filtered


def _filter_co3dv2_frames_with_existing_images(
    seq_data: Sequence[Dict[str, Any]],
    *,
    image_root: str,
) -> List[Dict[str, Any]]:
    root = Path(str(image_root))
    filtered: List[Dict[str, Any]] = []
    for item in seq_data:
        filepath = item.get("filepath")
        if filepath and (root / str(filepath)).is_file():
            filtered.append(item)
    return filtered


def run_re10k_core4_main_val(
    *,
    vggt_model: Any,
    ds_cfg: Dict[str, Any],
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, Any]:
    from aidi.scripts.vggt import eval_vggt_re10k_pose_lightweight as re10k_eval

    world_size = max(1, int(world_size))
    rank = int(rank)
    args = SimpleNamespace(
        model_tag="train_main_val",
        model_path="",
        config="",
        official_ckpt_root="",
        re10k_root=str(ds_cfg["root"]),
        seed=int(ds_cfg.get("paper10_seed", 20260215)),
        pool_size=int(ds_cfg.get("paper10_per_scene", 10)),
        n_srcs=int(ds_cfg.get("val_n_srcs", 9)),
        load_img_size=int(ds_cfg.get("load_img_size", 518)),
        image_mode=str(ds_cfg.get("image_mode", "crop")),
        image_load_retries=int(ds_cfg.get("image_load_retries", 8)),
        image_load_retry_sleep=float(ds_cfg.get("image_load_retry_sleep", 0.5)),
        devices=[],
        num_shards=world_size,
        shard_index=rank,
        worker_name=f"rank{rank:04d}",
        limit_scenes=0,
        scene_filter="",
        device=str(device),
    )

    dtype = re10k_eval.resolve_autocast_dtype(device)
    scene_roots = _resolve_re10k_scene_roots(
        args.re10k_root,
        int(ds_cfg.get("sample_stride", 1)),
        int(ds_cfg.get("limit_scenes", 0)),
    )
    scene_roots = _shard_by_rank(scene_roots, rank=rank, world_size=world_size)

    metrics: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for scene_dir in scene_roots:
        try:
            metric = re10k_eval.process_scene(
                model=vggt_model,
                scene_dir=scene_dir,
                args=args,
                device=device,
                dtype=dtype,
            )
            metrics.append(metric)
        except Exception as exc:
            failures.append(
                {
                    "path": re10k_eval.scene_key_from_dir(scene_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return re10k_eval.build_result_payload(
        args=args,
        loaded_model="live_model",
        device=device,
        dtype=dtype,
        metrics=metrics,
        failures=failures,
    )


def run_re10k_pi3_core4_main_val(
    *,
    pi3_model: Any,
    ds_cfg: Dict[str, Any],
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, Any]:
    from aidi.scripts.baselines import eval_pi3_re10k_pose_official as pi3_eval

    world_size = max(1, int(world_size))
    rank = int(rank)
    args = SimpleNamespace(
        model_tag="train_main_val",
        model_path="live_model",
        re10k_root=str(ds_cfg["root"]),
        seed=int(ds_cfg.get("paper10_seed", 20260215)),
        pool_size=int(ds_cfg.get("paper10_per_scene", 10)),
        n_srcs=int(ds_cfg.get("val_n_srcs", 9)),
        load_img_size=int(ds_cfg.get("load_img_size", 518)),
        image_load_retries=int(ds_cfg.get("image_load_retries", 8)),
        image_load_retry_sleep=float(ds_cfg.get("image_load_retry_sleep", 0.5)),
        limit_scenes=0,
        scene_filter="",
        device=str(device),
    )

    dtype = pi3_eval.resolve_autocast_dtype(device)
    scene_roots = _resolve_re10k_scene_roots(
        args.re10k_root,
        int(ds_cfg.get("sample_stride", 1)),
        int(ds_cfg.get("limit_scenes", 0)),
    )
    scene_roots = _shard_by_rank(scene_roots, rank=rank, world_size=world_size)

    metrics: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for scene_dir in scene_roots:
        try:
            metric = pi3_eval.process_scene(
                model=pi3_model,
                scene_dir=scene_dir,
                args=args,
                device=device,
                dtype=dtype,
            )
            metrics.append(metric)
        except Exception as exc:
            failures.append(
                {
                    "path": pi3_eval.scene_key_from_dir(scene_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "implementation": "pi3_aligned_re10k_live_main_val",
        "model_tag": args.model_tag,
        "model_path": "live_model",
        "loaded_model": "live_model",
        "re10k_root": str(args.re10k_root),
        "seed": int(args.seed),
        "pool_size": int(args.pool_size),
        "n_srcs": int(args.n_srcs),
        "load_img_size": int(args.load_img_size),
        "limit_scenes": int(ds_cfg.get("limit_scenes", 0)),
        "scene_filter": str(args.scene_filter),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "summary": pi3_eval.build_summary(metrics),
        "metrics": metrics,
        "failures": failures,
    }


def run_co3dv2_core4_main_val(
    *,
    vggt_model: Any,
    ds_cfg: Dict[str, Any],
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, Any]:
    from aidi.scripts.vggt import eval_co3d_pose_official_upstream as co3d_eval
    torch_mod = _require_torch()

    world_size = max(1, int(world_size))
    rank = int(rank)
    args = SimpleNamespace(
        debug=False,
        debug_category="",
        categories=str(ds_cfg.get("categories", "")),
        fast_eval=False,
        seed=int(ds_cfg.get("seed", 0)),
        num_frames=int(ds_cfg.get("num_frames", 10)),
        min_num_images=int(ds_cfg.get("min_num_images", 50)),
        co3d_image_root=str(ds_cfg.get("image_root") or CO3DV2_RAW_ROOT_DEFAULT),
        co3d_anno_dir=str(ds_cfg.get("anno_dir", "")),
        image_mode=str(ds_cfg.get("image_mode", "crop")),
        load_img_size=int(ds_cfg.get("load_img_size", 518)),
        image_load_retries=int(ds_cfg.get("image_load_retries", 8)),
        image_load_retry_sleep=float(ds_cfg.get("image_load_retry_sleep", 0.5)),
        model_tag="train_main_val",
        model_path="",
        official_ckpt_root="",
    )

    co3d_eval.set_random_seeds(args.seed)
    categories = co3d_eval.parse_category_list(args)
    dtype = torch_mod.bfloat16 if device.type == "cuda" and torch_mod.cuda.get_device_capability(device=device)[0] >= 8 else torch_mod.float16

    per_category_results: Dict[str, Dict[str, Any]] = {}
    per_sequence_results: Dict[str, List[Dict[str, Any]]] = {}
    failures: List[Dict[str, Any]] = []
    sample_stride = int(ds_cfg.get("sample_stride", 1))
    selected_annotations = _load_selected_co3dv2_annotations(ds_cfg, categories)

    for category in categories:
        annotation = selected_annotations.get(category)
        if not annotation:
            if rank == 0:
                failures.append({"category": category, "error": "missing_selected_annotation"})
            continue

        seq_names = _filter_co3dv2_seq_names_with_image_dirs(
            sorted(annotation.keys()),
            image_root=args.co3d_image_root,
            category=category,
        )
        if not seq_names:
            if rank == 0:
                failures.append({"category": category, "error": "no_available_image_sequence"})
            continue
        seq_names = _select_by_stride(seq_names, sample_stride)
        if int(ds_cfg.get("max_sequences", 0)) > 0:
            seq_names = seq_names[: int(ds_cfg["max_sequences"])]
        seq_names = _shard_by_rank(seq_names, rank=rank, world_size=world_size)

        sequence_results: List[Dict[str, Any]] = []
        for seq_name in seq_names:
            seq_data = _filter_co3dv2_frames_with_existing_images(annotation[seq_name], image_root=args.co3d_image_root)
            if len(seq_data) < max(args.min_num_images, args.num_frames):
                failures.append({"path": f"{category}/{seq_name}", "error": "insufficient_available_images"})
                continue
            try:
                # Make CO3Dv2 frame sampling independent of rank count and shard layout.
                per_sequence_seed = _stable_int_seed(args.seed, category, seq_name)
                co3d_eval.set_random_seeds(per_sequence_seed)
                seq_result = co3d_eval.process_sequence(
                    model=vggt_model,
                    seq_name=seq_name,
                    seq_data=seq_data,
                    category=category,
                    co3d_dir=args.co3d_image_root,
                    min_num_images=args.min_num_images,
                    num_frames=args.num_frames,
                    device=device,
                    dtype=dtype,
                    image_mode=args.image_mode,
                    load_img_size=args.load_img_size,
                    image_load_retries=args.image_load_retries,
                    image_load_retry_sleep=args.image_load_retry_sleep,
                )
            except Exception as exc:
                failures.append({"path": f"{category}/{seq_name}", "error": f"{type(exc).__name__}: {exc}"})
                continue

            if seq_result is not None:
                sequence_results.append(seq_result)

        if not sequence_results:
            continue

        per_sequence_results[category] = sequence_results
        per_category_results[category] = _summarize_co3dv2_category(sequence_results)

    if not per_category_results and world_size <= 1:
        raise RuntimeError("No category produced valid CO3Dv2 results.")

    overall_summary = _build_co3dv2_summary(per_category_results, per_sequence_results)

    metrics = [item for results in per_sequence_results.values() for item in results]
    return {
        "implementation": "vendored_upstream_test_co3d",
        "model_tag": args.model_tag,
        "co3d_dir": args.co3d_image_root,
        "co3d_anno_dir": args.co3d_anno_dir,
        "categories": categories,
        "num_frames": args.num_frames,
        "seed": args.seed,
        "image_mode": args.image_mode,
        "load_img_size": args.load_img_size,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "metrics": metrics,
        "per_category_results": per_category_results,
        "per_sequence_results": per_sequence_results,
        "summary": overall_summary,
        "failures": failures,
    }


def _extract_pi3_pred_extrinsics(predictions: Dict[str, Any]) -> np.ndarray:
    if "camera_poses" not in predictions:
        raise KeyError("camera_poses")
    camera_poses = predictions["camera_poses"]
    if hasattr(camera_poses, "detach"):
        camera_poses = camera_poses.detach().float().cpu().numpy()
    poses = np.asarray(camera_poses, dtype=np.float64)
    if poses.ndim == 4:
        if poses.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for camera_poses, got {poses.shape}")
        poses = poses[0]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected (N,4,4) camera_poses, got {poses.shape}")
    return np.linalg.inv(poses)[:, :3, :4]


def _run_pi3_co3dv2_sequence(
    *,
    pi3_model: Any,
    seq_name: str,
    seq_data: Sequence[Dict[str, Any]],
    category: str,
    args: SimpleNamespace,
    device: torch.device,
    dtype: torch.dtype,
    co3d_eval: Any,
) -> Optional[Dict[str, Any]]:
    torch_mod = _require_torch()

    if len(seq_data) < args.min_num_images:
        return None

    metadata = []
    for data in seq_data:
        if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
            return None
        extri_opencv = co3d_eval.convert_pt3d_RT_to_opencv(data["R"], data["T"])
        metadata.append({"filepath": data["filepath"], "extri": extri_opencv})

    ids = np.random.choice(len(metadata), args.num_frames, replace=False)
    image_names = [os.path.join(args.co3d_image_root, metadata[i]["filepath"]) for i in ids]
    gt_extri = np.stack([np.array(metadata[i]["extri"]) for i in ids], axis=0)

    from aidi.scripts.baselines import eval_pi3_re10k_pose_official as pi3_eval

    images = pi3_eval.load_images_with_retry(
        image_names=image_names,
        new_width=args.load_img_size,
        device=str(device),
        retries=args.image_load_retries,
        retry_sleep=args.image_load_retry_sleep,
    )

    with torch_mod.no_grad():
        if getattr(device, "type", "") == "cuda":
            with torch_mod.amp.autocast(device_type="cuda", dtype=dtype):
                predictions = pi3_model(images)
        else:
            predictions = pi3_model(images)

    pred_extrinsic = torch_mod.from_numpy(_extract_pi3_pred_extrinsics(predictions)).to(device=device, dtype=torch_mod.float64)
    gt_extrinsic = torch_mod.from_numpy(gt_extri).to(device=device, dtype=torch_mod.float64)

    add_row = torch_mod.tensor([0, 0, 0, 1], device=device, dtype=torch_mod.float64).expand(pred_extrinsic.size(0), 1, 4)
    pred_se3 = torch_mod.cat((pred_extrinsic, add_row), dim=1)
    gt_se3 = torch_mod.cat((gt_extrinsic, add_row), dim=1)

    rel_rangle_deg, rel_tangle_deg = co3d_eval.se3_to_relative_pose_error(pred_se3, gt_se3, args.num_frames)
    r_error = rel_rangle_deg.cpu().numpy()
    t_error = rel_tangle_deg.cpu().numpy()

    racc_5 = float((rel_rangle_deg < 5).float().mean().item())
    tacc_5 = float((rel_tangle_deg < 5).float().mean().item())
    auc_30, _ = co3d_eval.calculate_auc_np(r_error, t_error, max_threshold=30)
    auc_20, _ = co3d_eval.calculate_auc_np(r_error, t_error, max_threshold=20)
    auc_15, _ = co3d_eval.calculate_auc_np(r_error, t_error, max_threshold=15)
    auc_10, _ = co3d_eval.calculate_auc_np(r_error, t_error, max_threshold=10)
    auc_5, _ = co3d_eval.calculate_auc_np(r_error, t_error, max_threshold=5)
    auc_3, _ = co3d_eval.calculate_auc_np(r_error, t_error, max_threshold=3)

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


def run_co3dv2_pi3_core4_main_val(
    *,
    pi3_model: Any,
    ds_cfg: Dict[str, Any],
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, Any]:
    from aidi.scripts.vggt import eval_co3d_pose_official_upstream as co3d_eval
    torch_mod = _require_torch()

    world_size = max(1, int(world_size))
    rank = int(rank)
    args = SimpleNamespace(
        debug=False,
        debug_category="",
        categories=str(ds_cfg.get("categories", "")),
        seed=int(ds_cfg.get("seed", 0)),
        num_frames=int(ds_cfg.get("num_frames", 10)),
        min_num_images=int(ds_cfg.get("min_num_images", 50)),
        co3d_image_root=str(ds_cfg.get("image_root") or CO3DV2_RAW_ROOT_DEFAULT),
        co3d_anno_dir=str(ds_cfg.get("anno_dir", "")),
        load_img_size=int(ds_cfg.get("load_img_size", 518)),
        image_load_retries=int(ds_cfg.get("image_load_retries", 8)),
        image_load_retry_sleep=float(ds_cfg.get("image_load_retry_sleep", 0.5)),
        model_tag="train_main_val",
        model_path="live_model",
    )

    co3d_eval.set_random_seeds(args.seed)
    categories = co3d_eval.parse_category_list(args)
    dtype = torch_mod.bfloat16 if device.type == "cuda" and torch_mod.cuda.get_device_capability(device=device)[0] >= 8 else torch_mod.float16

    per_category_results: Dict[str, Dict[str, Any]] = {}
    per_sequence_results: Dict[str, List[Dict[str, Any]]] = {}
    failures: List[Dict[str, Any]] = []
    sample_stride = int(ds_cfg.get("sample_stride", 1))
    selected_annotations = _load_selected_co3dv2_annotations(ds_cfg, categories)

    for category in categories:
        annotation = selected_annotations.get(category)
        if not annotation:
            if rank == 0:
                failures.append({"category": category, "error": "missing_selected_annotation"})
            continue

        seq_names = _filter_co3dv2_seq_names_with_image_dirs(
            sorted(annotation.keys()),
            image_root=args.co3d_image_root,
            category=category,
        )
        if not seq_names:
            if rank == 0:
                failures.append({"category": category, "error": "no_available_image_sequence"})
            continue
        seq_names = _select_by_stride(seq_names, sample_stride)
        if int(ds_cfg.get("max_sequences", 0)) > 0:
            seq_names = seq_names[: int(ds_cfg["max_sequences"])]
        seq_names = _shard_by_rank(seq_names, rank=rank, world_size=world_size)

        sequence_results: List[Dict[str, Any]] = []
        for seq_name in seq_names:
            seq_data = _filter_co3dv2_frames_with_existing_images(annotation[seq_name], image_root=args.co3d_image_root)
            if len(seq_data) < max(args.min_num_images, args.num_frames):
                failures.append({"path": f"{category}/{seq_name}", "error": "insufficient_available_images"})
                continue
            try:
                per_sequence_seed = _stable_int_seed(args.seed, category, seq_name)
                co3d_eval.set_random_seeds(per_sequence_seed)
                seq_result = _run_pi3_co3dv2_sequence(
                    pi3_model=pi3_model,
                    seq_name=seq_name,
                    seq_data=seq_data,
                    category=category,
                    args=args,
                    device=device,
                    dtype=dtype,
                    co3d_eval=co3d_eval,
                )
            except Exception as exc:
                failures.append({"path": f"{category}/{seq_name}", "error": f"{type(exc).__name__}: {exc}"})
                continue

            if seq_result is not None:
                sequence_results.append(seq_result)

        if not sequence_results:
            continue

        per_sequence_results[category] = sequence_results
        per_category_results[category] = _summarize_co3dv2_category(sequence_results)

    if not per_category_results and world_size <= 1:
        raise RuntimeError("No category produced valid CO3Dv2 results.")

    overall_summary = _build_co3dv2_summary(per_category_results, per_sequence_results)
    metrics = [item for results in per_sequence_results.values() for item in results]
    return {
        "implementation": "pi3_aligned_official_co3d_live_main_val",
        "model_tag": args.model_tag,
        "model_path": "live_model",
        "loaded_model": "live_model",
        "co3d_dir": args.co3d_image_root,
        "co3d_anno_dir": args.co3d_anno_dir,
        "categories": categories,
        "num_frames": args.num_frames,
        "seed": args.seed,
        "load_img_size": args.load_img_size,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "metrics": metrics,
        "per_category_results": per_category_results,
        "per_sequence_results": per_sequence_results,
        "summary": overall_summary,
        "failures": failures,
    }


def _run_mv_recon_core4_dataset(
    *,
    dataset_name: str,
    official_model: Any,
    model_family: str,
    ds_cfg: Dict[str, Any],
    device: torch.device,
    temp_output_dir: Path,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, Any]:
    _require_torch()
    from aidi.scripts.baselines import eval_pi3_mv_recon_core as mvrecon

    rank = int(rank)
    world_size = max(1, int(world_size))
    dataset_root = Path(str(ds_cfg.get("dataset_root") or mvrecon.default_dataset_root(dataset_name))).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"{dataset_name} dataset root not found: {dataset_root}")

    umeyama, accuracy, completion = mvrecon.import_local_pi3_metric_utils()
    protocols = mvrecon.resolve_protocols(dataset_name, str(ds_cfg.get("protocol", "auto")))
    pi3_root = Path(str(ds_cfg.get("pi3_root", mvrecon.DEFAULT_PI3_ROOT))).resolve()
    seq_dirs = mvrecon.discover_sequences(dataset_root)
    if not seq_dirs:
        raise RuntimeError(f"No scene folders found under {dataset_root}")

    infer_cfg = SimpleNamespace(
        load_img_size=int(ds_cfg.get("load_img_size", 518)),
        device=str(device),
        verbose=bool(ds_cfg.get("verbose", False)),
        point_source=str(ds_cfg.get("point_source", "native")),
        vggt_input_style=str(ds_cfg.get("vggt_input_style", "official_crop")),
    )

    summary = {
        "dataset": dataset_name,
        "model_family": model_family,
        "vggt_model_tag": "live_main_val" if model_family == "vggt" else "",
        "dataset_root": str(dataset_root),
        "pi3_root": str(pi3_root),
        "ckpt": "live_model",
        "device": str(device),
        "point_source": infer_cfg.point_source,
        "vggt_input_style": infer_cfg.vggt_input_style,
        "load_img_size": infer_cfg.load_img_size,
        "dtu_unit_scale": float(ds_cfg.get("dtu_unit_scale", 1.0)),
        "dtu_center_crop_height": int(ds_cfg.get("dtu_center_crop_height", 0)),
        "dtu_data_format": str(ds_cfg.get("dtu_data_format", "auto")),
        "protocols": {},
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for protocol in protocols:
        seq_map = mvrecon.resolve_seq_map(
            pi3_root,
            dataset_name,
            protocol,
            str(ds_cfg.get("seq_map", "")),
        )
        eval_seq_map = seq_map
        eval_max_sequences = int(ds_cfg.get("max_sequences", 0))
        if world_size > 1:
            eval_seq_map = _write_sharded_seq_map(
                seq_map_path=seq_map,
                rank=rank,
                world_size=world_size,
                max_sequences=eval_max_sequences,
                output_dir=temp_output_dir / "seq_maps",
                protocol=protocol,
            )
            eval_max_sequences = 0
        infer_mv_pointclouds = (
            mvrecon.infer_pi3_mv_pointclouds
            if model_family == "pi3"
            else mvrecon.infer_vggt_mv_pointclouds
        )
        metric, skipped, seq_total = mvrecon.eval_protocol(
            dataset=dataset_name,
            protocol=protocol,
            seq_map_path=eval_seq_map,
            seq_dirs=seq_dirs,
            model=official_model,
            infer_cfg=infer_cfg,
            load_img_size=infer_cfg.load_img_size,
            max_depth=float(ds_cfg.get("max_depth", 10.0)),
            min_depth=float(ds_cfg.get("min_depth", 1e-3)),
            dtu_mask_erode=int(ds_cfg.get("dtu_mask_erode", 10)),
            dtu_unit_scale=float(ds_cfg.get("dtu_unit_scale", 1.0)),
            dtu_center_crop_height=int(ds_cfg.get("dtu_center_crop_height", 0)),
            dtu_data_format=str(ds_cfg.get("dtu_data_format", "auto")),
            eth3d_extri_c2w=bool(ds_cfg.get("eth3d_extri_c2w", False)),
            output_dir=temp_output_dir,
            infer_mv_pointclouds=infer_mv_pointclouds,
            umeyama=umeyama,
            accuracy=accuracy,
            completion=completion,
            icp_threshold=float(
                ds_cfg.get(
                    "icp_threshold_dtu" if dataset_name == "dtu" else "icp_threshold",
                    100.0 if dataset_name == "dtu" else 0.1,
                )
            ),
            max_sequences=eval_max_sequences,
            verbose=bool(ds_cfg.get("verbose", False)),
        )
        summary["protocols"][protocol] = {
            "seq_map": str(seq_map),
            "rank_seq_map": str(eval_seq_map),
            "rank": rank,
            "world_size": world_size,
            "metrics": metric,
            "num_seq_in_map": int(seq_total),
            "num_seq_evaluated_cap": int(ds_cfg.get("max_sequences", 0)),
            "icp_threshold": float(
                ds_cfg.get(
                    "icp_threshold_dtu" if dataset_name == "dtu" else "icp_threshold",
                    100.0 if dataset_name == "dtu" else 0.1,
                )
            ),
            "skipped": [{"seq": seq, "reason": reason} for seq, reason in skipped],
        }

    return summary


def run_dtu_core4_main_val(
    *,
    official_model: Any,
    model_family: str,
    ds_cfg: Dict[str, Any],
    device: torch.device,
    temp_output_dir: Path,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, Any]:
    return _run_mv_recon_core4_dataset(
        dataset_name="dtu",
        official_model=official_model,
        model_family=model_family,
        ds_cfg=ds_cfg,
        device=device,
        temp_output_dir=temp_output_dir,
        rank=rank,
        world_size=world_size,
    )


def run_eth3d_core4_main_val(
    *,
    official_model: Any,
    model_family: str,
    ds_cfg: Dict[str, Any],
    device: torch.device,
    temp_output_dir: Path,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, Any]:
    return _run_mv_recon_core4_dataset(
        dataset_name="eth3d",
        official_model=official_model,
        model_family=model_family,
        ds_cfg=ds_cfg,
        device=device,
        temp_output_dir=temp_output_dir,
        rank=rank,
        world_size=world_size,
    )


def _distributed_rank_world(distributed_sharding: bool) -> Tuple[bool, int, int]:
    torch_mod = _require_torch()
    dist = getattr(torch_mod, "distributed", None)
    if (
        not distributed_sharding
        or dist is None
        or not dist.is_available()
        or not dist.is_initialized()
    ):
        return False, 0, 1
    return True, int(dist.get_rank()), int(dist.get_world_size())


def _all_gather_rank_result(rank_result: Dict[str, Any], distributed: bool) -> List[Dict[str, Any]]:
    if not distributed:
        return [rank_result]
    torch_mod = _require_torch()
    gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(torch_mod.distributed.get_world_size())]
    torch_mod.distributed.all_gather_object(gathered, rank_result)
    return [item for item in gathered if item is not None]


def run_core4_main_val(
    *,
    model: Any,
    raw_cfg: Dict[str, Any],
    epoch: int,
    global_step: int,
    record_dir: str,
    repo_root: Optional[Path] = None,
    distributed_sharding: bool = False,
) -> Dict[str, Any]:
    _require_torch()
    cfg = normalize_core4_main_val_cfg(raw_cfg, repo_root=repo_root)
    if not cfg:
        raise ValueError("main_val_core4_cfg is empty")

    distributed, rank, world_size = _distributed_rank_world(distributed_sharding)
    rng_state = _save_rng_state()
    live_model, inference_model, was_training, model_family = _prepare_live_model_for_core4_eval(model, global_step)
    device = _resolve_model_device(live_model)
    output_root = _default_output_root(record_dir, epoch)
    if rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)

    aggregate = {
        "run_name": cfg["run"]["name"],
        "epoch": int(epoch),
        "global_step": int(global_step),
        "distributed_sharding": bool(distributed),
        "rank": int(rank),
        "world_size": int(world_size),
        "model_family": model_family,
        "datasets": [],
    }

    try:
        with tempfile.TemporaryDirectory(prefix=f"core4_main_val_epoch{epoch:04d}_rank{rank:04d}_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            for dataset_name in CORE4_DATASET_ORDER:
                ds_cfg = cfg["datasets"].get(dataset_name, {})
                if not ds_cfg.get("enabled"):
                    continue

                started = time.time()
                rank_result: Dict[str, Any]
                try:
                    if dataset_name == "re10k":
                        if model_family == "pi3":
                            payload = run_re10k_pi3_core4_main_val(
                                pi3_model=inference_model,
                                ds_cfg=ds_cfg,
                                device=device,
                                rank=rank,
                                world_size=world_size,
                            )
                        else:
                            payload = run_re10k_core4_main_val(
                                vggt_model=inference_model,
                                ds_cfg=ds_cfg,
                                device=device,
                                rank=rank,
                                world_size=world_size,
                            )
                    elif dataset_name == "co3dv2":
                        if model_family == "pi3":
                            payload = run_co3dv2_pi3_core4_main_val(
                                pi3_model=inference_model,
                                ds_cfg=ds_cfg,
                                device=device,
                                rank=rank,
                                world_size=world_size,
                            )
                        else:
                            payload = run_co3dv2_core4_main_val(
                                vggt_model=inference_model,
                                ds_cfg=ds_cfg,
                                device=device,
                                rank=rank,
                                world_size=world_size,
                            )
                    elif dataset_name == "dtu":
                        payload = run_dtu_core4_main_val(
                            official_model=live_model,
                            model_family=model_family,
                            ds_cfg=ds_cfg,
                            device=device,
                            temp_output_dir=tmp_root / "dtu",
                            rank=rank,
                            world_size=world_size,
                        )
                    elif dataset_name == "eth3d":
                        payload = run_eth3d_core4_main_val(
                            official_model=live_model,
                            model_family=model_family,
                            ds_cfg=ds_cfg,
                            device=device,
                            temp_output_dir=tmp_root / "eth3d",
                            rank=rank,
                            world_size=world_size,
                        )
                    else:
                        raise ValueError(f"Unsupported dataset: {dataset_name}")
                    rank_result = {
                        "rank": rank,
                        "runtime_sec": float(time.time() - started),
                        "payload": payload,
                    }
                except Exception as exc:
                    rank_result = {
                        "rank": rank,
                        "runtime_sec": float(time.time() - started),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }

                rank_results = _all_gather_rank_result(rank_result, distributed)
                errors = [item for item in rank_results if item.get("error")]
                if errors:
                    error_text = "; ".join(f"rank{item['rank']}: {item['error']}" for item in errors)
                    raise RuntimeError(f"Core4 main validation failed for {dataset_name}: {error_text}")

                merged_payload = _merge_dataset_payloads(
                    dataset_name,
                    [item["payload"] for item in rank_results if "payload" in item],
                )
                if rank == 0:
                    aggregate["datasets"].append(
                        {
                            "name": dataset_name,
                            "runtime_sec": float(max(item.get("runtime_sec", 0.0) for item in rank_results)),
                            "payload": _compact_payload(dataset_name, merged_payload),
                            "num_shards": int(world_size),
                        }
                    )
    finally:
        _restore_live_model_after_core4_eval(live_model, was_training, global_step, model_family)
        _restore_rng_state(rng_state)

    if rank == 0:
        summary_path = output_root / "aggregate_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2, ensure_ascii=False)
        aggregate["summary_path"] = str(summary_path)
    return aggregate
