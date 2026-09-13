#!/usr/bin/env python3
"""Evaluate predeclared stopping gates for the expanded distractor sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


METRICS = ("ATE", "RPE trans")
ENDPOINT_VARIANTS = ("noise0", "noise4")
FULL_VARIANTS = tuple(f"noise{index}" for index in range(5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    endpoint = subparsers.add_parser("endpoint")
    endpoint.add_argument("--new-pi3", required=True)
    endpoint.add_argument("--new-geoweave", required=True)
    endpoint.add_argument("--old-pi3", required=True)
    endpoint.add_argument("--old-geoweave", required=True)
    endpoint.add_argument("--output", required=True)

    full = subparsers.add_parser("full")
    full.add_argument("--pi3", required=True)
    full.add_argument("--geoweave", required=True)
    full.add_argument("--output", required=True)
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalized_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        payload["sample_id"] = str(row["sample_id"])
        payload["variant"] = str(row["variant"])
        for metric in METRICS:
            payload[metric] = float(row[metric])
            if not np.isfinite(payload[metric]):
                raise ValueError(f"Non-finite {metric} for {payload['sample_id']} {payload['variant']}")
        output.append(payload)
    return output


def validate_complete(rows: Sequence[dict[str, Any]], variants: Sequence[str]) -> list[str]:
    by_sample: dict[str, set[str]] = {}
    for row in normalized_rows(rows):
        by_sample.setdefault(str(row["sample_id"]), set()).add(str(row["variant"]))
    required = set(variants)
    incomplete = {sample: sorted(required - present) for sample, present in by_sample.items() if required - present}
    if incomplete:
        raise ValueError(f"Incomplete variants: {incomplete}")
    return sorted(by_sample)


def means_by_variant(rows: Sequence[dict[str, Any]], variants: Sequence[str]) -> dict[str, dict[str, float]]:
    normalized = normalized_rows(rows)
    validate_complete(normalized, variants)
    output: dict[str, dict[str, float]] = {}
    for variant in variants:
        subset = [row for row in normalized if row["variant"] == variant]
        if not subset:
            raise ValueError(f"Missing variant {variant}")
        output[variant] = {metric: float(np.mean([row[metric] for row in subset])) for metric in METRICS}
    return output


def ensure_paired_samples(pi3_rows: Sequence[dict[str, Any]], geo_rows: Sequence[dict[str, Any]], variants: Sequence[str]) -> None:
    pi3_samples = validate_complete(pi3_rows, variants)
    geo_samples = validate_complete(geo_rows, variants)
    if pi3_samples != geo_samples:
        raise ValueError("Pi3 and GeoWeave sample IDs do not match")


def summarize_endpoints(pi3_rows: Sequence[dict[str, Any]], geo_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ensure_paired_samples(pi3_rows, geo_rows, ENDPOINT_VARIANTS)
    pi3 = means_by_variant(pi3_rows, ENDPOINT_VARIANTS)
    geo = means_by_variant(geo_rows, ENDPOINT_VARIANTS)
    payload: dict[str, Any] = {
        "num_samples": len(validate_complete(pi3_rows, ENDPOINT_VARIANTS)),
        "pi3": pi3,
        "geoweave": geo,
    }
    for metric in METRICS:
        pi3_degradation = pi3["noise4"][metric] - pi3["noise0"][metric]
        geo_degradation = geo["noise4"][metric] - geo["noise0"][metric]
        payload[f"pi3_degradation_{metric}"] = pi3_degradation
        payload[f"geoweave_degradation_{metric}"] = geo_degradation
        payload[f"advantage_{metric}"] = pi3_degradation - geo_degradation
    return payload


def endpoint_gate(
    new_pi3_rows: Sequence[dict[str, Any]],
    new_geo_rows: Sequence[dict[str, Any]],
    old_pi3_rows: Sequence[dict[str, Any]],
    old_geo_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    new_summary = summarize_endpoints(new_pi3_rows, new_geo_rows)
    old_summary = summarize_endpoints(old_pi3_rows, old_geo_rows)
    combined_summary = summarize_endpoints(
        list(new_pi3_rows) + list(old_pi3_rows),
        list(new_geo_rows) + list(old_geo_rows),
    )
    reasons: list[str] = []
    if all(new_summary[f"advantage_{metric}"] <= 0.0 for metric in METRICS):
        reasons.append("new20_both_metrics_nonpositive")
    for metric in METRICS:
        if combined_summary[f"advantage_{metric}"] <= 0.0:
            reasons.append(f"combined40_{metric}_nonpositive")
    return {
        "decision": "stop" if reasons else "proceed",
        "reasons": reasons,
        "sets": {"old20": old_summary, "new20": new_summary, "combined40": combined_summary},
    }


def full_sweep_gate(pi3_rows: Sequence[dict[str, Any]], geo_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ensure_paired_samples(pi3_rows, geo_rows, FULL_VARIANTS)
    pi3 = means_by_variant(pi3_rows, FULL_VARIANTS)
    geo = means_by_variant(geo_rows, FULL_VARIANTS)
    reasons: list[str] = []
    payload: dict[str, Any] = {
        "num_samples": len(validate_complete(pi3_rows, FULL_VARIANTS)),
        "pi3": pi3,
        "geoweave": geo,
    }
    for metric in METRICS:
        endpoint_advantage = (
            pi3["noise4"][metric]
            - pi3["noise0"][metric]
            - geo["noise4"][metric]
            + geo["noise0"][metric]
        )
        pi3_area = float(np.mean([pi3[f"noise{k}"][metric] - pi3["noise0"][metric] for k in range(1, 5)]))
        geo_area = float(np.mean([geo[f"noise{k}"][metric] - geo["noise0"][metric] for k in range(1, 5)]))
        area_advantage = pi3_area - geo_area
        payload[f"endpoint_advantage_{metric}"] = endpoint_advantage
        payload[f"degradation_area_advantage_{metric}"] = area_advantage
        if geo["noise4"][metric] >= pi3["noise4"][metric]:
            reasons.append(f"noise4_absolute_{metric}_not_lower")
        if endpoint_advantage <= 0.0:
            reasons.append(f"noise4_degradation_{metric}_nonpositive")
        if area_advantage <= 0.0:
            reasons.append(f"degradation_area_{metric}_nonpositive")
    payload["decision"] = "stop" if reasons else "proceed"
    payload["reasons"] = reasons
    return payload


def write_result(path: str | Path, result: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.command == "endpoint":
        result = endpoint_gate(
            read_csv(args.new_pi3),
            read_csv(args.new_geoweave),
            read_csv(args.old_pi3),
            read_csv(args.old_geoweave),
        )
    else:
        result = full_sweep_gate(read_csv(args.pi3), read_csv(args.geoweave))
    write_result(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
