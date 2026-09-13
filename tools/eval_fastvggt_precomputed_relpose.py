#!/usr/bin/env python3
"""Evaluate precomputed FastVGGT poses on rebuttal relpose protocols.

FastVGGT inference is produced by ``tools/fastvggt_blendedmvg_protocol.py`` as
``points.npz`` files. Those files contain VGGT camera extrinsics. This script
does not rerun inference and does not define a new metric: it converts the
stored extrinsics to camera-to-world poses, applies the protocol's official
``eval_frame_indices``, and calls the same evo-backed ATE/RPE evaluator used by
the Pi3/GeoWeave reproduction scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_repo_root_on_syspath() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


ensure_repo_root_on_syspath()

from aidi.scripts.baselines import eval_pi3_relpose_distance_protocol as relpose_eval  # noqa: E402


DEFAULT_PROTOCOL_ROOT = Path("/mnt/cfs/zhoufeng/geoweave_repro_inputs/rebuttal_allsettings_20260702")
DEFAULT_FASTVGGT_ROOT = Path(
    "/mnt/cfs/zhoufeng/geoweave_repro_outputs/rebuttal_allsettings_fastvggt_internal_20260702"
)

SETTING_DEFAULTS: dict[str, dict[str, str]] = {
    "scannetpp_weak": {
        "protocol": "scannetpp_weak_protocol.json",
        "dataset": "scannetv2",
    },
    "waymo_weak": {
        "protocol": "waymo_weak_protocol.json",
        "dataset": "vkitti2",
    },
    "waymo_plausible": {
        "protocol": "waymo_plausible_protocol.json",
        "dataset": "vkitti2",
    },
}

METRIC_KEYS = ("ATE", "RPE trans", "RPE rot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting-name", choices=sorted(SETTING_DEFAULTS), required=True)
    parser.add_argument("--protocol", default="")
    parser.add_argument("--protocol-root", default=str(DEFAULT_PROTOCOL_ROOT))
    parser.add_argument("--fastvggt-root", default=str(DEFAULT_FASTVGGT_ROOT))
    parser.add_argument("--model-name", default="fastvggt_m0_r090")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_protocol_path(args: argparse.Namespace) -> Path:
    if args.protocol:
        return Path(args.protocol).expanduser().resolve()
    defaults = SETTING_DEFAULTS[str(args.setting_name)]
    return (Path(args.protocol_root).expanduser() / defaults["protocol"]).resolve()


def resolve_dataset_name(args: argparse.Namespace) -> str:
    if args.dataset:
        return str(args.dataset)
    return SETTING_DEFAULTS[str(args.setting_name)]["dataset"]


def iter_protocol_jobs(
    protocol: Mapping[str, Any],
    *,
    sample_limit: int = 0,
) -> Iterable[tuple[str, str, Mapping[str, Any], Mapping[str, Any]]]:
    samples = list(protocol.get("samples", []))
    if sample_limit > 0:
        samples = samples[:sample_limit]
    for sample in samples:
        sample_id = str(sample["sample_id"])
        variants = sample.get("variants", {})
        for variant in protocol.get("variants", variants.keys()):
            if variant not in variants:
                continue
            yield sample_id, str(variant), variants[variant], sample


def protocol_variant_count(protocol: Mapping[str, Any]) -> int:
    variants = protocol.get("variants", [])
    if variants:
        return len(variants)
    return max((len(sample.get("variants", {})) for sample in protocol.get("samples", [])), default=0)


def output_sequence_name(protocol: Mapping[str, Any], sample_id: str, variant: str) -> str:
    if protocol_variant_count(protocol) <= 1:
        return sample_id
    return f"{sample_id}__{variant}"


def resolve_points_npz(
    fastvggt_root: str | Path,
    setting_name: str,
    model_name: str,
    sample_id: str,
    variant: str,
) -> Path:
    root = Path(fastvggt_root).expanduser().resolve()
    candidates = (
        root / setting_name / model_name / sample_id / variant / "points.npz",
        root / model_name / sample_id / variant / "points.npz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing FastVGGT points.npz for {setting_name}/{sample_id}/{variant}: {candidates}")


def parse_eval_indices(variant_payload: Mapping[str, Any]) -> list[int]:
    meta_path = Path(str(variant_payload["tuple_meta_path"]))
    meta = load_json(meta_path)
    return relpose_eval.parse_eval_frame_indices(",".join(str(item) for item in meta.get("eval_frame_indices", [])))


def load_gt_c2w(variant_payload: Mapping[str, Any]) -> np.ndarray:
    sequence_dir = Path(str(variant_payload["sequence_dir"]))
    pose_path = sequence_dir / "pose_90.txt"
    return relpose_eval.load_replica_pose_file(pose_path)


def load_fastvggt_pred_c2w(points_npz: str | Path) -> np.ndarray:
    data = np.load(points_npz)
    if "extrinsic" not in data:
        raise KeyError(f"FastVGGT output lacks 'extrinsic': {points_npz}")
    return relpose_eval.vggt_extrinsics_to_c2w(np.asarray(data["extrinsic"], dtype=np.float64))


def validate_image_order(points_npz: str | Path, variant_payload: Mapping[str, Any]) -> None:
    data = np.load(points_npz)
    if "image_paths" not in data:
        raise KeyError(f"FastVGGT output lacks 'image_paths': {points_npz}")
    actual = [Path(str(path)).name for path in data["image_paths"].tolist()]
    expected = [Path(str(frame["image_path"])).name for frame in variant_payload.get("frames", [])]
    if actual != expected:
        raise RuntimeError(
            f"FastVGGT image order mismatch for {points_npz}: actual={actual[:5]} expected={expected[:5]}"
        )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else float("nan")


def build_dataset_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "ATE": metric_mean(rows, "ATE"),
        "RPE trans": metric_mean(rows, "RPE trans"),
        "RPE rot": metric_mean(rows, "RPE rot"),
        "num_sequences": int(len(rows)),
    }


def build_variant_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for variant in sorted({str(row["variant"]) for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        summary[variant] = build_dataset_summary(subset)
    return summary


def build_pair_delta_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_sample_variant: dict[tuple[str, str], Mapping[str, Any]] = {
        (str(row["sample_id"]), str(row["variant"])): row for row in rows
    }
    pairs = []
    for sample_id in sorted({str(row["sample_id"]) for row in rows}):
        clean = by_sample_variant.get((sample_id, "clean_tail"))
        noise = by_sample_variant.get((sample_id, "plausible_noise_tail"))
        if clean is None or noise is None:
            continue
        row: dict[str, Any] = {"sample_id": sample_id}
        for key in METRIC_KEYS:
            row[f"clean_{key}"] = float(clean[key])
            row[f"distractor_{key}"] = float(noise[key])
            row[f"delta_{key}"] = float(noise[key]) - float(clean[key])
        pairs.append(row)
    if not pairs:
        return {"num_pairs": 0}
    summary: dict[str, Any] = {"num_pairs": len(pairs), "pairs": pairs}
    for key in METRIC_KEYS:
        summary[f"mean_delta_{key}"] = float(np.mean([pair[f"delta_{key}"] for pair in pairs]))
    return summary


def evaluate_job(
    *,
    protocol: Mapping[str, Any],
    setting_name: str,
    dataset_name: str,
    model_name: str,
    fastvggt_root: Path,
    dataset_root: Path,
    sample_id: str,
    variant: str,
    variant_payload: Mapping[str, Any],
    evo_utils: Any,
    skip_plot: bool,
    verbose: bool,
) -> dict[str, Any]:
    points_npz = resolve_points_npz(fastvggt_root, setting_name, model_name, sample_id, variant)
    validate_image_order(points_npz, variant_payload)
    pred_c2w_full = load_fastvggt_pred_c2w(points_npz)
    gt_c2w_full = load_gt_c2w(variant_payload)
    eval_indices = parse_eval_indices(variant_payload)
    pred_c2w = relpose_eval.subset_pose_array_for_eval(pred_c2w_full, eval_indices)
    gt_c2w = relpose_eval.subset_pose_array_for_eval(gt_c2w_full, eval_indices)

    seq = output_sequence_name(protocol, sample_id, variant)
    seq_root = dataset_root / seq
    seq_root.mkdir(parents=True, exist_ok=True)
    pred_traj = evo_utils.get_tum_poses(pred_c2w)
    gt_traj = evo_utils.get_tum_poses(gt_c2w)
    evo_utils.save_tum_poses(pred_traj, str(seq_root / "pred_traj.txt"), verbose=verbose)
    np.save(seq_root / "pred_poses.npy", pred_c2w)
    np.save(seq_root / "pred_poses_full.npy", pred_c2w_full)
    write_json(seq_root / "eval_frame_indices.json", {"eval_frame_indices": eval_indices})
    ate, rpe_trans, rpe_rot = evo_utils.eval_metrics(
        pred_traj=pred_traj,
        gt_traj=gt_traj,
        seq=seq,
        filename=str(seq_root / "eval_metric.txt"),
        verbose=verbose,
    )
    if not skip_plot:
        evo_utils.plot_trajectory(
            pred_traj=pred_traj,
            gt_traj=gt_traj,
            title=seq,
            filename=str(seq_root / "vis.png"),
            align=True,
            correct_scale=True,
            verbose=verbose,
        )
    return {
        "dataset": dataset_name,
        "seq": seq,
        "sample_id": sample_id,
        "variant": variant,
        "ATE": float(ate),
        "RPE trans": float(rpe_trans),
        "RPE rot": float(rpe_rot),
        "points_npz": str(points_npz),
        "num_input_frames": int(pred_c2w_full.shape[0]),
        "num_eval_frames": int(pred_c2w.shape[0]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = resolve_protocol_path(args)
    protocol = load_json(protocol_path)
    setting_name = str(args.setting_name)
    dataset_name = resolve_dataset_name(args)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_root = output_root / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)
    evo_utils = relpose_eval.load_evo_utils_runtime()

    rows = [
        evaluate_job(
            protocol=protocol,
            setting_name=setting_name,
            dataset_name=dataset_name,
            model_name=str(args.model_name),
            fastvggt_root=Path(args.fastvggt_root),
            dataset_root=dataset_root,
            sample_id=sample_id,
            variant=variant,
            variant_payload=variant_payload,
            evo_utils=evo_utils,
            skip_plot=bool(args.skip_plot),
            verbose=bool(args.verbose),
        )
        for sample_id, variant, variant_payload, _sample in iter_protocol_jobs(
            protocol,
            sample_limit=int(args.sample_limit),
        )
    ]
    write_csv(dataset_root / "seq_metrics.csv", rows)
    dataset_summary = build_dataset_summary(rows)
    write_csv(
        output_root / f"{dataset_name}-metric.csv",
        [
            {
                "dataset": dataset_name,
                "ATE": dataset_summary["ATE"],
                "RPE trans": dataset_summary["RPE trans"],
                "RPE rot": dataset_summary["RPE rot"],
                "num_sequences": dataset_summary["num_sequences"],
            }
        ],
    )
    summary = {
        "setting_name": setting_name,
        "datasets": [dataset_name],
        "model": str(args.model_name),
        "protocol": str(protocol_path),
        "fastvggt_root": str(Path(args.fastvggt_root).expanduser().resolve()),
        "output_root": str(output_root),
        "results": [
            {
                "dataset": dataset_name,
                "root": str(protocol.get("source_root", "")),
                "layout": "protocol_precomputed",
                "summary": dataset_summary,
                "variant_summary": build_variant_summary(rows),
                "paired_delta_summary": build_pair_delta_summary(rows),
                "num_sequences": len(rows),
                "model": str(args.model_name),
            }
        ],
    }
    write_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
