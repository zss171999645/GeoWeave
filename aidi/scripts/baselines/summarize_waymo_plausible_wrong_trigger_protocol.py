#!/usr/bin/env python3
"""Summarize Pi3 metrics for Waymo plausible-wrong trigger protocol."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


METRICS = ("ATE", "RPE trans", "RPE rot")
BASELINE_VARIANTS = ("prefix6_only", "clean_tail", "interleaved_clean_tail")
KEY_VARIANTS = (
    "plausible_wrong_tail",
    "plausible_wrong_reversed",
    "plausible_wrong_tail_clean_gt_tail",
    "plausible_wrong_blurred_tail",
    "gray_tail",
    "repeat_eval_last",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--model-tags", default="official,sparse79")
    parser.add_argument("--dataset-name", default="vkitti2")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def parse_csv_list(text: str) -> List[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def read_variant_index(protocol_root: Path) -> Dict[str, Dict[str, str]]:
    path = protocol_root / "variant_index.csv"
    rows: Dict[str, Dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["seq"]] = dict(row)
    return rows


def read_metrics(result_root: Path, model_tag: str, dataset_name: str) -> Dict[str, Dict[str, float]]:
    path = result_root / model_tag / dataset_name / "seq_metrics.csv"
    rows: Dict[str, Dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["seq"]] = {metric: float(row[metric]) for metric in METRICS}
    return rows


def mean(values: Sequence[float]) -> float:
    return float(np.mean(list(values))) if values else float("nan")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(protocol_root: Path, result_root: Path, model_tags: Sequence[str], dataset_name: str) -> Dict[str, Any]:
    variant_index = read_variant_index(protocol_root)
    by_sample: Dict[str, Dict[str, str]] = defaultdict(dict)
    for seq, row in variant_index.items():
        by_sample[row["sample_id"]][row["variant"]] = seq

    per_sample_rows: List[Dict[str, Any]] = []
    per_variant_rows: List[Dict[str, Any]] = []
    metrics_by_model = {tag: read_metrics(result_root, tag, dataset_name) for tag in model_tags}

    for sample_id, variants in sorted(by_sample.items()):
        for model_tag, metrics in metrics_by_model.items():
            base_values = {
                baseline: metrics[variants[baseline]]
                for baseline in BASELINE_VARIANTS
                if baseline in variants and variants[baseline] in metrics
            }
            for variant, seq in sorted(variants.items()):
                if seq not in metrics:
                    continue
                row: Dict[str, Any] = {
                    "sample_id": sample_id,
                    "model": model_tag,
                    "variant": variant,
                    "seq": seq,
                }
                for metric in METRICS:
                    row[metric] = float(metrics[seq][metric])
                    for baseline, baseline_values in base_values.items():
                        row[f"{metric}_delta_vs_{baseline}"] = float(metrics[seq][metric]) - float(
                            baseline_values[metric]
                        )
                per_sample_rows.append(row)

    for model_tag in model_tags:
        for variant in sorted({row["variant"] for row in per_sample_rows}):
            rows = [row for row in per_sample_rows if row["model"] == model_tag and row["variant"] == variant]
            if not rows:
                continue
            out: Dict[str, Any] = {"model": model_tag, "variant": variant, "num_samples": len(rows)}
            for metric in METRICS:
                out[f"{metric}_mean"] = mean([float(row[metric]) for row in rows])
                for baseline in BASELINE_VARIANTS:
                    key = f"{metric}_delta_vs_{baseline}"
                    values = [float(row[key]) for row in rows if key in row]
                    if values:
                        out[f"{key}_mean"] = mean(values)
            per_variant_rows.append(out)

    checks: Dict[str, Any] = {}
    if set(model_tags) >= {"official", "sparse79"}:
        for variant in KEY_VARIANTS:
            official = [
                row
                for row in per_sample_rows
                if row["model"] == "official" and row["variant"] == variant and "ATE_delta_vs_clean_tail" in row
            ]
            sparse = [
                row
                for row in per_sample_rows
                if row["model"] == "sparse79" and row["variant"] == variant and "ATE_delta_vs_clean_tail" in row
            ]
            sparse_by_sample = {row["sample_id"]: row for row in sparse}
            paired = [(row, sparse_by_sample[row["sample_id"]]) for row in official if row["sample_id"] in sparse_by_sample]
            checks[variant] = {
                "num_pairs": len(paired),
                "official_ATE_delta_vs_clean_mean": mean([float(row["ATE_delta_vs_clean_tail"]) for row, _ in paired]),
                "sparse79_ATE_delta_vs_clean_mean": mean([float(row["ATE_delta_vs_clean_tail"]) for _, row in paired]),
                "sparse79_smaller_degradation_count": int(
                    sum(
                        float(sp["ATE_delta_vs_clean_tail"]) < float(off["ATE_delta_vs_clean_tail"])
                        for off, sp in paired
                    )
                ),
            }

        wrong = [
            row for row in per_sample_rows if row["model"] == "official" and row["variant"] == "plausible_wrong_tail"
        ]
        pose_ctrl = [
            row
            for row in per_sample_rows
            if row["model"] == "official" and row["variant"] == "plausible_wrong_tail_clean_gt_tail"
        ]
        ctrl_by_sample = {row["sample_id"]: row for row in pose_ctrl}
        diffs = [
            abs(float(row["ATE"]) - float(ctrl_by_sample[row["sample_id"]]["ATE"]))
            for row in wrong
            if row["sample_id"] in ctrl_by_sample
        ]
        checks["tail_gt_pose_control_max_abs_ATE_diff_official"] = max(diffs) if diffs else float("nan")

    return {
        "protocol_root": str(protocol_root),
        "result_root": str(result_root),
        "model_tags": list(model_tags),
        "dataset_name": dataset_name,
        "num_samples": len(by_sample),
        "per_sample": per_sample_rows,
        "per_variant": per_variant_rows,
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    protocol_root = Path(args.protocol_root).expanduser().resolve()
    result_root = Path(args.result_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else result_root / "trigger_protocol_summary.json"
    summary = build_summary(
        protocol_root=protocol_root,
        result_root=result_root,
        model_tags=parse_csv_list(args.model_tags),
        dataset_name=str(args.dataset_name),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output.with_suffix(".per_sample.csv"), summary["per_sample"])
    write_csv(output.with_suffix(".per_variant.csv"), summary["per_variant"])
    print(json.dumps({k: v for k, v in summary.items() if k not in {"per_sample", "per_variant"}}, indent=2))


if __name__ == "__main__":
    main()
