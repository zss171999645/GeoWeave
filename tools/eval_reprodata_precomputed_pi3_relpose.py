#!/usr/bin/env python3
"""Evaluate cached Pi3 camera poses for a multi-variant protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


METRIC_KEYS = ("ATE", "RPE trans", "RPE rot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def iter_protocol_jobs(protocol: Mapping[str, Any], *, sample_limit: int = 0):
    samples = list(protocol.get("samples", []))
    if sample_limit > 0:
        samples = samples[:sample_limit]
    for sample in samples:
        for variant in protocol.get("variants", []):
            yield str(sample["sample_id"]), str(variant), sample["variants"][variant]


def resolve_points_npz(root: Path, model: str, sample_id: str, variant: str) -> Path:
    path = root / model / sample_id / variant / "points.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_pred_c2w(path: str | Path) -> np.ndarray:
    with np.load(path) as data:
        if "camera_poses" not in data:
            raise KeyError(f"camera_poses missing from {path}; keys={list(data.keys())}")
        poses = np.asarray(data["camera_poses"], dtype=np.float64)
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected camera poses shaped (N,4,4), got {poses.shape}")
    return poses


def load_gt_c2w(variant_payload: Mapping[str, Any]) -> np.ndarray:
    sequence_dir = Path(str(variant_payload["sequence_dir"]))
    poses = np.loadtxt(sequence_dir / "pose_90.txt", dtype=np.float64)
    return poses.reshape(-1, 4, 4)


def parse_eval_indices(variant_payload: Mapping[str, Any]) -> list[int]:
    meta = load_json(variant_payload["tuple_meta_path"])
    indices = [int(item) for item in meta.get("eval_frame_indices", [])]
    if not indices:
        raise ValueError(f"Missing eval_frame_indices in {variant_payload['tuple_meta_path']}")
    return indices


def validate_image_order(points_npz: Path, variant_payload: Mapping[str, Any]) -> None:
    with np.load(points_npz) as data:
        if "image_paths" not in data:
            raise KeyError(f"image_paths missing from {points_npz}")
        predicted = [Path(str(item)).name for item in data["image_paths"].tolist()]
    expected = [Path(str(frame["image_path"])).name for frame in variant_payload["frames"]]
    if predicted != expected:
        raise RuntimeError(f"image order mismatch for {points_npz}: predicted={predicted}, expected={expected}")


def metric_mean(rows: Iterable[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else float("nan")


def variant_distractor_count(variant: str) -> int | None:
    variant = str(variant)
    if not variant.startswith("noise"):
        return None
    suffix = variant.removeprefix("noise")
    return int(suffix) if suffix.isdigit() else None


def build_variant_summary(
    rows: list[dict[str, Any]],
    variant_order: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    present = {str(row["variant"]) for row in rows}
    variants = [str(item) for item in variant_order if str(item) in present] if variant_order is not None else sorted(present)
    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant]
        result[variant] = {
            "num_sequences": len(subset),
            **{key: metric_mean(subset, key) for key in METRIC_KEYS},
        }
    return result


def build_clean_delta_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_sample_variant = {(str(row["sample_id"]), str(row["variant"])): row for row in rows}
    variants = sorted(
        {str(row["variant"]) for row in rows},
        key=lambda item: variant_distractor_count(item) if variant_distractor_count(item) is not None else item,
    )
    sample_ids = sorted({str(row["sample_id"]) for row in rows})
    result: dict[str, dict[str, Any]] = {}
    for variant in variants:
        pairs: list[dict[str, Any]] = []
        for sample_id in sample_ids:
            clean = by_sample_variant.get((sample_id, "noise0"))
            current = by_sample_variant.get((sample_id, variant))
            if clean is None or current is None:
                continue
            pair = {"sample_id": sample_id}
            for key in METRIC_KEYS:
                pair[f"delta_{key}"] = float(current[key]) - float(clean[key])
            pairs.append(pair)
        result[variant] = {
            "num_pairs": len(pairs),
            **{f"mean_delta_{key}": metric_mean(pairs, f"delta_{key}") for key in METRIC_KEYS},
            "pairs": pairs,
        }
    return result


def evaluate_job(
    *,
    model: str,
    sample_id: str,
    variant: str,
    variant_payload: Mapping[str, Any],
    prediction_root: Path,
    output_root: Path,
    evo_utils: Any,
    skip_plot: bool,
    verbose: bool,
) -> dict[str, Any]:
    from aidi.scripts.baselines import eval_pi3_relpose_distance_protocol as relpose_eval

    points_npz = resolve_points_npz(prediction_root, model, sample_id, variant)
    validate_image_order(points_npz, variant_payload)
    pred_full = load_pred_c2w(points_npz)
    gt_full = load_gt_c2w(variant_payload)
    eval_indices = parse_eval_indices(variant_payload)
    pred = relpose_eval.subset_pose_array_for_eval(pred_full, eval_indices)
    gt = relpose_eval.subset_pose_array_for_eval(gt_full, eval_indices)
    sequence_name = f"{sample_id}__{variant}"
    sequence_root = output_root / sequence_name
    sequence_root.mkdir(parents=True, exist_ok=True)
    pred_traj = evo_utils.get_tum_poses(pred)
    gt_traj = evo_utils.get_tum_poses(gt)
    evo_utils.save_tum_poses(pred_traj, str(sequence_root / "pred_traj.txt"), verbose=verbose)
    np.save(sequence_root / "pred_poses.npy", pred)
    np.save(sequence_root / "pred_poses_full.npy", pred_full)
    ate, rpe_trans, rpe_rot = evo_utils.eval_metrics(
        pred_traj=pred_traj,
        gt_traj=gt_traj,
        seq=sequence_name,
        filename=str(sequence_root / "eval_metric.txt"),
        verbose=verbose,
    )
    if not skip_plot:
        evo_utils.plot_trajectory(
            pred_traj=pred_traj,
            gt_traj=gt_traj,
            title=sequence_name,
            filename=str(sequence_root / "vis.png"),
            align=True,
            correct_scale=True,
            verbose=verbose,
        )
    row = {
        "model": model,
        "sample_id": sample_id,
        "variant": variant,
        "ATE": float(ate),
        "RPE trans": float(rpe_trans),
        "RPE rot": float(rpe_rot),
        "num_input_frames": int(pred_full.shape[0]),
        "num_eval_frames": int(pred.shape[0]),
        "points_npz": str(points_npz),
    }
    distractor_count = variant_distractor_count(variant)
    if distractor_count is not None:
        row["distractor_count"] = int(distractor_count)
    if "cross_overlap_mean" in variant_payload:
        row["cross_overlap_mean"] = float(variant_payload["cross_overlap_mean"])
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    from aidi.scripts.baselines import eval_pi3_relpose_distance_protocol as relpose_eval

    protocol = load_json(args.protocol)
    prediction_root = Path(args.prediction_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    evo_utils = relpose_eval.load_evo_utils_runtime()
    rows = [
        evaluate_job(
            model=str(args.model),
            sample_id=sample_id,
            variant=variant,
            variant_payload=payload,
            prediction_root=prediction_root,
            output_root=output_root / "sequences",
            evo_utils=evo_utils,
            skip_plot=bool(args.skip_plot),
            verbose=bool(args.verbose),
        )
        for sample_id, variant, payload in iter_protocol_jobs(protocol, sample_limit=int(args.sample_limit))
    ]
    result = {
        "model": str(args.model),
        "protocol": str(Path(args.protocol).expanduser().resolve()),
        "num_rows": len(rows),
        "rows": rows,
        "variant_summary": build_variant_summary(rows, variant_order=protocol.get("variants", [])),
        "clean_delta_summary": build_clean_delta_summary(rows) if "noise0" in protocol.get("variants", []) else {},
    }
    write_csv(output_root / "pose_rows.csv", rows)
    write_json(output_root / "pose_rows.json", rows)
    write_json(output_root / "pose_summary.json", result)
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
