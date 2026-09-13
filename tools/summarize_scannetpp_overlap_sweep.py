#!/usr/bin/env python3
"""Aggregate paired Pi3/GeoWeave ScanNet++ overlap-sweep pose results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_LEVELS = ("high", "medium", "low", "near_zero")
METRICS = {
    "ATE": "ATE",
    "RPE_trans": "RPE trans",
    "RPE_rot": "RPE rot",
}


def parse_levels(raw: str) -> tuple[str, ...]:
    levels = tuple(item.strip() for item in str(raw).split(",") if item.strip())
    if not levels or len(set(levels)) != len(levels):
        raise ValueError(f"Expected distinct comma-separated levels, got {raw!r}")
    return levels


def read_csv(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _row_map(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["sample_id"]), str(row["variant"]))
        if key in result:
            raise ValueError(f"Duplicate pose row for {key}")
        result[key] = row
    return result


def build_paired_rows(
    pi3_rows: Sequence[Mapping[str, Any]],
    geoweave_rows: Sequence[Mapping[str, Any]],
    levels: Sequence[str] = DEFAULT_LEVELS,
) -> list[dict[str, Any]]:
    pi3 = _row_map(pi3_rows)
    geoweave = _row_map(geoweave_rows)
    if set(pi3) != set(geoweave):
        missing_geoweave = sorted(set(pi3) - set(geoweave))
        missing_pi3 = sorted(set(geoweave) - set(pi3))
        raise ValueError(
            f"paired key mismatch: missing_geoweave={missing_geoweave[:5]} missing_pi3={missing_pi3[:5]}"
        )
    level_index = {str(level): index for index, level in enumerate(levels)}
    unknown = sorted({variant for _, variant in pi3 if variant not in level_index})
    if unknown:
        raise ValueError(f"Unexpected overlap levels: {unknown}")

    paired: list[dict[str, Any]] = []
    for sample_id, level in sorted(pi3, key=lambda key: (key[0], level_index[key[1]])):
        base = pi3[(sample_id, level)]
        geo = geoweave[(sample_id, level)]
        base_overlap = float(base["cross_overlap_mean"])
        geo_overlap = float(geo["cross_overlap_mean"])
        if not np.isclose(base_overlap, geo_overlap, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Measured overlap differs between models for {(sample_id, level)}")
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "overlap_level": level,
            "cross_overlap_mean": base_overlap,
        }
        for short_name, source_name in METRICS.items():
            base_value = float(base[source_name])
            geo_value = float(geo[source_name])
            if not np.isfinite(base_value) or not np.isfinite(geo_value):
                raise ValueError(f"Non-finite {source_name} for {(sample_id, level)}")
            row[f"pi3_{short_name}"] = base_value
            row[f"geoweave_{short_name}"] = geo_value
            row[f"delta_{short_name}"] = base_value - geo_value
        paired.append(row)

    sample_ids = sorted({str(row["sample_id"]) for row in paired})
    expected_levels = set(str(item) for item in levels)
    for sample_id in sample_ids:
        actual = {str(row["overlap_level"]) for row in paired if row["sample_id"] == sample_id}
        if actual != expected_levels:
            raise ValueError(f"Sample {sample_id} has incomplete levels: {sorted(actual)}")
    return paired


def _bootstrap_means(
    values_by_sample: Mapping[str, float],
    sampled_indices: np.ndarray,
    sample_ids: Sequence[str],
) -> np.ndarray:
    values = np.asarray([float(values_by_sample[sample_id]) for sample_id in sample_ids], dtype=np.float64)
    return values[sampled_indices].mean(axis=1)


def summarize_bands(
    paired_rows: Sequence[Mapping[str, Any]],
    levels: Sequence[str] = DEFAULT_LEVELS,
    bootstrap_repeats: int = 10000,
    seed: int = 20260826,
) -> list[dict[str, Any]]:
    sample_ids = sorted({str(row["sample_id"]) for row in paired_rows})
    if not sample_ids:
        raise ValueError("No paired rows")
    if int(bootstrap_repeats) <= 0:
        raise ValueError(f"bootstrap_repeats must be positive, got {bootstrap_repeats}")
    by_key = {(str(row["sample_id"]), str(row["overlap_level"])): row for row in paired_rows}
    expected = {(sample_id, str(level)) for sample_id in sample_ids for level in levels}
    if set(by_key) != expected:
        raise ValueError("Paired rows do not form a complete anchor-by-level grid")

    rng = np.random.default_rng(int(seed))
    sampled_indices = rng.integers(0, len(sample_ids), size=(int(bootstrap_repeats), len(sample_ids)))
    summaries: list[dict[str, Any]] = []
    for level in levels:
        level_rows = [by_key[(sample_id, str(level))] for sample_id in sample_ids]
        summary: dict[str, Any] = {
            "overlap_level": str(level),
            "num_anchors": int(len(sample_ids)),
            "mean_cross_overlap": float(np.mean([float(row["cross_overlap_mean"]) for row in level_rows])),
        }
        for prefix in ("pi3", "geoweave", "delta"):
            for metric in METRICS:
                key = f"{prefix}_{metric}"
                values_by_sample = {str(row["sample_id"]): float(row[key]) for row in level_rows}
                values = np.asarray(list(values_by_sample.values()), dtype=np.float64)
                bootstrap = _bootstrap_means(values_by_sample, sampled_indices, sample_ids)
                summary[f"mean_{key}"] = float(values.mean())
                summary[f"median_{key}"] = float(np.median(values))
                summary[f"p90_{key}"] = float(np.quantile(values, 0.9))
                summary[f"max_{key}"] = float(values.max())
                summary[f"{key}_ci_low"] = float(np.quantile(bootstrap, 0.025))
                summary[f"{key}_ci_high"] = float(np.quantile(bootstrap, 0.975))
        summaries.append(summary)
    return summaries


def plot_summaries(summary_rows: Sequence[Mapping[str, Any]], output_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(summary_rows, key=lambda row: float(row["mean_cross_overlap"]))
    x = np.asarray([float(row["mean_cross_overlap"]) for row in rows])
    output_root.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 3.5), constrained_layout=True)
    for prefix, label, color, marker in (
        ("pi3", "Pi3", "#6B7280", "o"),
        ("geoweave", "GeoWeave", "#2563EB", "s"),
    ):
        y = np.asarray([float(row[f"mean_{prefix}_ATE"]) for row in rows])
        low = np.asarray([float(row[f"{prefix}_ATE_ci_low"]) for row in rows])
        high = np.asarray([float(row[f"{prefix}_ATE_ci_high"]) for row in rows])
        ax.errorbar(x, y, yerr=np.stack([y - low, high - y]), label=label, color=color, marker=marker, capsize=3)
    ax.set_xlabel("Measured cross-group overlap")
    ax.set_ylabel("ATE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(output_root / f"ate_vs_overlap.{suffix}", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.5), constrained_layout=True)
    y = np.asarray([float(row["mean_delta_ATE"]) for row in rows])
    low = np.asarray([float(row["delta_ATE_ci_low"]) for row in rows])
    high = np.asarray([float(row["delta_ATE_ci_high"]) for row in rows])
    ax.errorbar(x, y, yerr=np.stack([y - low, high - y]), color="#2563EB", marker="o", capsize=3)
    ax.axhline(0.0, color="#111827", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Measured cross-group overlap")
    ax.set_ylabel(r"$\Delta$ATE (Pi3 $-$ GeoWeave)")
    ax.grid(alpha=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(output_root / f"delta_ate_vs_overlap.{suffix}", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi3-pose-rows", required=True)
    parser.add_argument("--geoweave-pose-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--levels", default=",".join(DEFAULT_LEVELS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir).expanduser().resolve()
    levels = parse_levels(args.levels)
    paired = build_paired_rows(
        read_csv(Path(args.pi3_pose_rows)),
        read_csv(Path(args.geoweave_pose_rows)),
        levels=levels,
    )
    summary_rows = summarize_bands(
        paired,
        levels=levels,
        bootstrap_repeats=int(args.bootstrap_repeats),
        seed=int(args.seed),
    )
    write_csv(output_root / "per_anchor_metrics.csv", paired)
    write_csv(output_root / "band_summary.csv", summary_rows)
    write_json(
        output_root / "summary.json",
        {
            "levels": list(levels),
            "num_anchors": len({row["sample_id"] for row in paired}),
            "num_rows": len(paired),
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "seed": int(args.seed),
            "bands": summary_rows,
        },
    )
    plot_summaries(summary_rows, output_root)
    print(f"[overlap-summary] anchors={len({row['sample_id'] for row in paired})} rows={len(paired)} output={output_root}")


if __name__ == "__main__":
    main()
