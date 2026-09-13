#!/usr/bin/env python3
"""Run FastVGGT on the BlendedMVG rebuttal protocol.

This is intentionally a bridge script, not a modification of FastVGGT.  It
loads the external FastVGGT implementation, reuses our protocol image folders,
and writes `points.npz` files that can be summarized against the BlendedMVG GT
prefix point clouds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image


DEFAULT_VARIANTS = ("clean", "tail", "xscene")
DEFAULT_PREFIX_SIZE = 6


@dataclass(frozen=True)
class RowSim3:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    target_rms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run FastVGGT inference for protocol jobs.")
    run.add_argument("--protocol", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--fastvggt-root", required=True)
    run.add_argument("--official-ckpt-root", required=True)
    run.add_argument("--model-name", default="fastvggt_m0_r090")
    run.add_argument("--variants", nargs="*", default=list(DEFAULT_VARIANTS))
    run.add_argument("--sample-limit", type=int, default=0)
    run.add_argument("--merging", default="0", help="FastVGGT merging block index, or 'none' for vanilla VGGT.")
    run.add_argument("--merge-ratio", type=float, default=0.9)
    run.add_argument("--load-img-size", type=int, default=518)
    run.add_argument("--depth-conf-thresh", type=float, default=3.0)
    run.add_argument("--overwrite", action="store_true")

    summ = sub.add_parser("summarize", help="Summarize FastVGGT outputs against GT prefix point clouds.")
    summ.add_argument("--protocol", required=True)
    summ.add_argument("--output-root", required=True)
    summ.add_argument("--model-name", default="fastvggt_m0_r090")
    summ.add_argument("--variants", nargs="*", default=list(DEFAULT_VARIANTS))
    summ.add_argument("--sample-limit", type=int, default=0)
    summ.add_argument("--prefix-size", type=int, default=DEFAULT_PREFIX_SIZE)
    summ.add_argument("--pred-stride", type=int, default=16)
    summ.add_argument("--gt-stride", type=int, default=16)
    summ.add_argument("--max-points", type=int, default=120000)
    summ.add_argument("--summary-name", default="fastvggt_prefix_pointcloud_metrics")

    jobs = sub.add_parser("emit-jobs", help="Print the jobs implied by a protocol.")
    jobs.add_argument("--protocol", required=True)
    jobs.add_argument("--variants", nargs="*", default=list(DEFAULT_VARIANTS))
    jobs.add_argument("--sample-limit", type=int, default=0)

    return parser.parse_args()


def load_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_protocol_jobs(
    protocol: dict[str, Any],
    *,
    variants: Sequence[str] | None = None,
    sample_limit: int = 0,
) -> Iterator[tuple[str, str, dict[str, Any], dict[str, Any]]]:
    selected_variants = tuple(variants or DEFAULT_VARIANTS)
    samples = protocol.get("samples", [])
    if sample_limit > 0:
        samples = samples[:sample_limit]
    for sample in samples:
        sample_id = str(sample["sample_id"])
        sample_variants = sample.get("variants", {})
        for variant in selected_variants:
            if variant not in sample_variants:
                continue
            yield sample_id, variant, sample_variants[variant], sample


def remap_aggregator_special_token_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    remapped = dict(state_dict)

    def remap_suffix(old_suffix: str, new_suffix: str) -> None:
        for key in list(remapped):
            if not key.endswith(old_suffix) or key.endswith(new_suffix):
                continue
            new_key = key[: -len(old_suffix)] + new_suffix
            if new_key in remapped:
                continue
            remapped[new_key] = remapped.pop(key)

    remap_suffix("camera_token", "special_tokens.camera_token")
    remap_suffix("register_token", "special_tokens.register_token")
    remap_suffix("patch_embed.cls_token", "patch_embed.special_tokens.cls_token")
    remap_suffix("patch_embed.pos_embed", "patch_embed.special_tokens.pos_embed")
    remap_suffix("patch_embed.register_tokens", "patch_embed.special_tokens.register_tokens")
    remap_suffix("patch_embed.mask_token", "patch_embed.special_tokens.mask_token")
    return remapped


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    stripped: dict[str, Any] = {}
    for key, value in state_dict.items():
        while key.startswith("module."):
            key = key[len("module.") :]
        stripped[key] = value
    return stripped


def choose_state_dict_candidate(
    target_keys: Iterable[str],
    candidates: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[str], list[str]]:
    target = set(target_keys)
    best: tuple[int, int, int, dict[str, Any], list[str], list[str]] | None = None
    for index, candidate in enumerate(candidates):
        source = set(candidate)
        missing = sorted(target - source)
        unexpected = sorted(source - target)
        score = len(missing) + len(unexpected)
        current = (score, len(missing), index, candidate, missing, unexpected)
        if best is None or current[:3] < best[:3]:
            best = current
    if best is None:
        raise ValueError("No state_dict candidates supplied")
    _score, _missing_count, _index, candidate, missing, unexpected = best
    return candidate, missing, unexpected


def estimate_row_sim3(source: np.ndarray, target: np.ndarray) -> RowSim3:
    """Estimate a Sim(3) transform for row-vector points: target ~= scale * source @ R + t."""

    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if src.shape != tgt.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 arrays, got {src.shape} and {tgt.shape}")
    if src.shape[0] < 3:
        raise ValueError(f"Need at least 3 points, got {src.shape[0]}")

    src_mu = src.mean(axis=0)
    tgt_mu = tgt.mean(axis=0)
    src_c = src - src_mu
    tgt_c = tgt - tgt_mu
    cov = src_c.T @ tgt_c / float(src.shape[0])
    u, singular, vt = np.linalg.svd(cov)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        singular[-1] *= -1.0
        rotation = u @ vt

    var_src = float(np.mean(np.sum(src_c * src_c, axis=1)))
    scale = float(np.sum(singular) / max(var_src, 1.0e-12))
    translation = tgt_mu - scale * (src_mu @ rotation)
    target_rms = float(np.sqrt(np.mean(np.sum(tgt_c * tgt_c, axis=1))))
    return RowSim3(scale=scale, rotation=rotation, translation=translation, target_rms=max(target_rms, 1.0e-12))


def apply_row_sim3(points: np.ndarray, transform: RowSim3) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    return (pts @ transform.rotation) * float(transform.scale) + transform.translation


def finite_points(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1)


def flatten_points(
    points: np.ndarray,
    *,
    stride: int,
    max_points: int,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(points)
    if arr.ndim == 4:
        arr = arr[:, ::stride, ::stride, :]
        mask = finite_points(arr)
        if valid is not None:
            mask &= np.asarray(valid)[:, ::stride, ::stride]
        flat = arr[mask]
    elif arr.ndim == 2 and arr.shape[1] == 3:
        mask = finite_points(arr)
        flat = arr[mask]
    else:
        raise ValueError(f"Unsupported point array shape: {arr.shape}")

    if max_points > 0 and flat.shape[0] > max_points:
        idx = np.linspace(0, flat.shape[0] - 1, max_points).round().astype(np.int64)
        flat = flat[idx]
    return flat.astype(np.float64, copy=False)


def nearest_neighbors(reference: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree as KDTree

    tree = KDTree(np.asarray(reference, dtype=np.float64))
    distances, indices = tree.query(np.asarray(query, dtype=np.float64), workers=-1)
    return distances.astype(np.float64, copy=False), indices.astype(np.int64, copy=False)


def mean_or_nan(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.mean(vals)) if vals else float("nan")


def median_or_nan(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(vals)) if vals else float("nan")


def cache_depth_to_world(cache_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(cache_path)
    depth = np.asarray(data["depth"], dtype=np.float64)
    intrinsic = np.asarray(data["intrinsic"], dtype=np.float64)
    c2w = np.asarray(data["pose"], dtype=np.float64)
    valid = depth > 1.0e-4

    h, w = depth.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    z = depth
    x = (xx.astype(np.float64) - intrinsic[0, 2]) * z / max(float(intrinsic[0, 0]), 1.0e-12)
    y = (yy.astype(np.float64) - intrinsic[1, 2]) * z / max(float(intrinsic[1, 1]), 1.0e-12)
    cam = np.stack([x, y, z], axis=-1)
    world = cam @ c2w[:3, :3].T + c2w[:3, 3]
    return world.astype(np.float32), valid


def load_gt_prefix(sample: dict[str, Any], variant: str, prefix_size: int) -> tuple[np.ndarray, np.ndarray]:
    frames = sample["variants"][variant]["frames"][:prefix_size]
    points = []
    valid = []
    for frame in frames:
        pts, mask = cache_depth_to_world(frame["cache_path"])
        points.append(pts)
        valid.append(mask)
    return np.stack(points, axis=0), np.stack(valid, axis=0)


def load_split_checkpoint_module(module: Any, ckpt_path: Path, *, name: str) -> tuple[list[str], list[str]]:
    import torch

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = state.get("model", state)
    state_dict = strip_module_prefix(state_dict)
    candidates = [state_dict]
    if name == "aggregator":
        remapped = remap_aggregator_special_token_keys(state_dict)
        if set(remapped) != set(state_dict):
            candidates.append(remapped)
    state_dict, _key_missing, _key_unexpected = choose_state_dict_candidate(module.state_dict().keys(), candidates)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    return list(missing), list(unexpected)


def add_fastvggt_to_path(root: str | Path) -> None:
    fastvggt_root = str(Path(root).resolve())
    if fastvggt_root not in sys.path:
        sys.path.insert(0, fastvggt_root)


def checkpoint_import_roots(script_path: str | Path) -> list[Path]:
    path = Path(script_path).resolve()
    if path.parent.name == "tools":
        return [path.parent.parent]
    return [path.parent]


def add_checkpoint_import_roots() -> None:
    for root in checkpoint_import_roots(__file__):
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.append(root_str)


def build_model(args: argparse.Namespace) -> Any:
    import torch

    add_fastvggt_to_path(args.fastvggt_root)
    add_checkpoint_import_roots()
    from vggt.models.vggt import VGGT

    merging = None if str(args.merging).lower() in {"none", "null", "-1"} else int(args.merging)
    model = VGGT(merging=merging, merge_ratio=float(args.merge_ratio), enable_point=False, enable_track=False)
    ckpt_root = Path(args.official_ckpt_root)
    load_plan = [
        ("aggregator", model.aggregator, ckpt_root / "aggregator.pt"),
        ("camera", model.camera_head, ckpt_root / "camera.pt"),
        ("depth", model.depth_head, ckpt_root / "depth.pt"),
    ]
    for name, module, ckpt_path in load_plan:
        if module is None:
            continue
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing {name} checkpoint: {ckpt_path}")
        missing, unexpected = load_split_checkpoint_module(module, ckpt_path, name=name)
        optional = ("track_head", "point_head")
        missing = [key for key in missing if not any(token in key for token in optional)]
        unexpected = [key for key in unexpected if not any(token in key for token in optional)]
        print(f"[fastvggt] loaded {name}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if missing:
            print(f"[fastvggt] {name} missing preview: {missing[:5]}", flush=True)
        if unexpected:
            print(f"[fastvggt] {name} unexpected preview: {unexpected[:5]}", flush=True)
    model = model.cuda().eval().to(torch.bfloat16)
    return model


def sorted_image_paths(input_dir: str | Path) -> list[Path]:
    paths: list[Path] = []
    root = Path(input_dir)
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(sorted(root.glob(pattern)))
    return sorted(paths)


def load_images_for_vggt(image_paths: Sequence[Path], load_img_size: int) -> tuple[Any, np.ndarray, int, int]:
    import torch
    from torchvision import transforms as tv_transforms

    to_tensor = tv_transforms.ToTensor()
    tensors = []
    colors = []
    final_width = int(load_img_size)
    final_height = int(load_img_size)
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        new_width = int(load_img_size)
        new_height = int(round(height * (new_width / width) / 14.0) * 14)
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        tensor = to_tensor(image)
        if new_height > load_img_size:
            start_y = (new_height - load_img_size) // 2
            tensor = tensor[:, start_y : start_y + load_img_size, :]
            final_height = int(load_img_size)
        else:
            final_height = int(new_height)
        final_width = int(new_width)
        tensors.append(tensor)
        color = (np.transpose(tensor.numpy(), (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
        colors.append(color)
    if not tensors:
        raise ValueError("No images found for FastVGGT inference")
    return torch.stack(tensors), np.stack(colors, axis=0), final_width // 14, final_height // 14


def unproject_depth_to_world(depth: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    seq, height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    world_points = np.empty((seq, height, width, 3), dtype=np.float32)
    for i in range(seq):
        z = depth[i]
        fx = max(float(intrinsic[i, 0, 0]), 1.0e-12)
        fy = max(float(intrinsic[i, 1, 1]), 1.0e-12)
        cx = float(intrinsic[i, 0, 2])
        cy = float(intrinsic[i, 1, 2])
        cam = np.stack([(xx - cx) * z / fx, (yy - cy) * z / fy, z], axis=-1)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :4] = extrinsic[i]
        c2w = np.linalg.inv(w2c)
        world_points[i] = (cam @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32)
    return world_points


def run_single_inference(
    model: Any,
    input_dir: Path,
    output_dir: Path,
    *,
    load_img_size: int,
    depth_conf_thresh: float,
) -> None:
    import torch

    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    image_paths = sorted_image_paths(input_dir)
    if len(image_paths) < 3:
        raise ValueError(f"FastVGGT expects at least 3 images, got {len(image_paths)} in {input_dir}")
    images, colors, patch_width, patch_height = load_images_for_vggt(image_paths, load_img_size=load_img_size)
    if hasattr(model, "update_patch_dimensions"):
        model.update_patch_dimensions(patch_width, patch_height)

    torch.cuda.synchronize()
    start = time.time()
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        predictions = model(images.cuda().to(torch.bfloat16), image_paths=[str(path) for path in image_paths])
    torch.cuda.synchronize()
    inference_ms = (time.time() - start) * 1000.0

    height = int(images.shape[-2])
    width = int(images.shape[-1])
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], (height, width))
    depth = predictions["depth"][0].detach().float().cpu().numpy().squeeze(-1)
    depth_conf = predictions["depth_conf"][0].detach().float().cpu().numpy()
    depth = depth.astype(np.float32, copy=False)
    depth[depth_conf < float(depth_conf_thresh)] = np.nan
    points = unproject_depth_to_world(
        depth.astype(np.float64, copy=False),
        extrinsic[0].detach().float().cpu().numpy(),
        intrinsic[0].detach().float().cpu().numpy(),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "points.npz",
        points=points,
        colors=colors,
        image_paths=np.asarray([str(path) for path in image_paths]),
        depth=depth,
        depth_conf=depth_conf.astype(np.float32, copy=False),
        extrinsic=extrinsic[0].detach().float().cpu().numpy(),
        intrinsic=intrinsic[0].detach().float().cpu().numpy(),
        inference_ms=np.asarray(inference_ms, dtype=np.float32),
    )
    meta = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_images": len(image_paths),
        "height": height,
        "width": width,
        "patch_width": patch_width,
        "patch_height": patch_height,
        "depth_conf_thresh": float(depth_conf_thresh),
        "inference_ms": float(inference_ms),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_protocol(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    model = build_model(args)
    output_root = Path(args.output_root) / args.model_name
    jobs = list(iter_protocol_jobs(protocol, variants=args.variants, sample_limit=args.sample_limit))
    print(f"[fastvggt] jobs={len(jobs)} output_root={output_root}", flush=True)
    for index, (sample_id, variant, variant_payload, _sample) in enumerate(jobs, start=1):
        input_dir = Path(variant_payload["input_dir"])
        output_dir = output_root / sample_id / variant
        if (output_dir / "points.npz").exists() and not args.overwrite:
            print(f"[fastvggt] skip existing {index}/{len(jobs)} {sample_id} {variant}", flush=True)
            continue
        print(f"[fastvggt] run {index}/{len(jobs)} {sample_id} {variant}", flush=True)
        run_single_inference(
            model,
            input_dir,
            output_dir,
            load_img_size=int(args.load_img_size),
            depth_conf_thresh=float(args.depth_conf_thresh),
        )


def summarize_pair(pred_points: np.ndarray, gt_points: np.ndarray) -> dict[str, Any]:
    if pred_points.shape[0] < 8 or gt_points.shape[0] < 8:
        return {"status": f"failed:not_enough_points:pred={pred_points.shape[0]}:gt={gt_points.shape[0]}"}
    n_align = min(pred_points.shape[0], gt_points.shape[0], 50000)
    pred_align = pred_points[np.linspace(0, pred_points.shape[0] - 1, n_align).round().astype(np.int64)]
    gt_align = gt_points[np.linspace(0, gt_points.shape[0] - 1, n_align).round().astype(np.int64)]
    try:
        transform = estimate_row_sim3(pred_align, gt_align)
        pred_aligned = apply_row_sim3(pred_points, transform)
        acc_dist, _ = nearest_neighbors(gt_points, pred_aligned)
        comp_dist, _ = nearest_neighbors(pred_aligned, gt_points)
    except Exception as exc:
        return {"status": f"failed:{type(exc).__name__}:{exc}"}
    norm = float(transform.target_rms)
    return {
        "status": "ok",
        "n_pred": int(pred_points.shape[0]),
        "n_gt": int(gt_points.shape[0]),
        "n_align": int(n_align),
        "align_scale": float(transform.scale),
        "target_rms": norm,
        "acc_mean_norm": float(np.mean(acc_dist / norm)),
        "acc_median_norm": float(np.median(acc_dist / norm)),
        "acc_p90_norm": float(np.percentile(acc_dist / norm, 90.0)),
        "comp_mean_norm": float(np.mean(comp_dist / norm)),
        "comp_median_norm": float(np.median(comp_dist / norm)),
        "comp_p90_norm": float(np.percentile(comp_dist / norm, 90.0)),
    }


def summarize_protocol(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    output_root = Path(args.output_root) / args.model_name
    rows: list[dict[str, Any]] = []
    for sample_id, variant, _variant_payload, sample in iter_protocol_jobs(
        protocol, variants=args.variants, sample_limit=args.sample_limit
    ):
        row: dict[str, Any] = {"sample_id": sample_id, "variant": variant, "model": args.model_name}
        out_file = output_root / sample_id / variant / "points.npz"
        try:
            pred_npz = np.load(out_file)
            pred = np.asarray(pred_npz["points"][: int(args.prefix_size)], dtype=np.float64)
            gt, gt_valid = load_gt_prefix(sample, variant, int(args.prefix_size))
            pred_flat = flatten_points(pred, stride=int(args.pred_stride), max_points=int(args.max_points))
            gt_flat = flatten_points(
                gt,
                stride=int(args.gt_stride),
                max_points=int(args.max_points),
                valid=gt_valid,
            )
            row.update(summarize_pair(pred_flat, gt_flat))
            if "inference_ms" in pred_npz:
                row["inference_ms"] = float(np.asarray(pred_npz["inference_ms"]).reshape(()))
        except Exception as exc:
            row["status"] = f"failed:{type(exc).__name__}:{exc}"
        rows.append(row)

    summary_dir = Path(args.output_root) / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / f"{args.summary_name}.csv"
    json_path = summary_dir / f"{args.summary_name}.json"
    md_path = summary_dir / f"{args.summary_name}.md"
    write_csv(csv_path, rows)
    summary = aggregate_rows(rows)
    json_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    md_path.write_text(format_summary_markdown(summary, rows), encoding="utf-8")
    print(f"[fastvggt] wrote {csv_path}", flush=True)
    print(f"[fastvggt] wrote {json_path}", flush=True)
    print(f"[fastvggt] wrote {md_path}", flush=True)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    variants = sorted({str(row.get("variant")) for row in rows})
    for variant in variants:
        subset = [row for row in rows if row.get("variant") == variant and row.get("status") == "ok"]
        summary[variant] = {
            "rows_ok": len(subset),
            "acc_mean_norm": mean_or_nan(row.get("acc_mean_norm", float("nan")) for row in subset),
            "comp_mean_norm": mean_or_nan(row.get("comp_mean_norm", float("nan")) for row in subset),
            "acc_median_norm": median_or_nan(row.get("acc_mean_norm", float("nan")) for row in subset),
            "comp_median_norm": median_or_nan(row.get("comp_mean_norm", float("nan")) for row in subset),
            "inference_ms_mean": mean_or_nan(row.get("inference_ms", float("nan")) for row in subset),
        }
    return summary


def format_summary_markdown(summary: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# FastVGGT BlendedMVG Prefix Point-Cloud Metrics",
        "",
        "This is an external VGGT token-merging baseline on the rebuttal protocol. It is not a Pi3/GeoWeave same-backbone ablation.",
        "",
        "| Variant | OK rows | Acc mean norm | Comp mean norm | Mean inference ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant, item in summary.items():
        lines.append(
            f"| {variant} | {item['rows_ok']} | {item['acc_mean_norm']:.6f} | "
            f"{item['comp_mean_norm']:.6f} | {item['inference_ms_mean']:.1f} |"
        )
    failed = [row for row in rows if row.get("status") != "ok"]
    if failed:
        lines.extend(["", "## Failed Rows", ""])
        for row in failed[:20]:
            lines.append(f"- {row.get('sample_id')} {row.get('variant')}: {row.get('status')}")
    lines.append("")
    return "\n".join(lines)


def emit_jobs(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    for sample_id, variant, variant_payload, _sample in iter_protocol_jobs(
        protocol, variants=args.variants, sample_limit=args.sample_limit
    ):
        print(f"{sample_id}\t{variant}\t{variant_payload['input_dir']}")


def main() -> None:
    args = parse_args()
    if args.cmd == "run":
        run_protocol(args)
    elif args.cmd == "summarize":
        summarize_protocol(args)
    elif args.cmd == "emit-jobs":
        emit_jobs(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
