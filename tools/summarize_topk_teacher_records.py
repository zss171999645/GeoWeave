#!/usr/bin/env python3
"""Summarize GeoWeave dense-teacher Top-K records.

This reads the reusable ``.npz`` records written by
``reprodata_pi3_topk_mechanism_experiments.py selection-change --teacher-only``
and computes:

1. Top-K set overlap between ``geoweave_dense_teacher_topk`` and
   ``geoweave_scorer_topk``.
2. Merged role-ratio rows/aggregates from all shards.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REFERENCE_KEY = "geoweave_dense_teacher_topk"
CANDIDATE_KEY = "geoweave_scorer_topk"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Result root containing per-setting shard_* dirs.")
    parser.add_argument("--settings", nargs="+", required=True)
    parser.add_argument("--reference-key", default=REFERENCE_KEY)
    parser.add_argument("--candidate-key", default=CANDIDATE_KEY)
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def aggregate_numeric_rows(
    rows: Sequence[dict[str, Any]],
    *,
    group_keys: Sequence[str],
    metric_keys: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key, "") for key in group_keys), []).append(row)

    out: list[dict[str, Any]] = []
    for key_tuple, group in sorted(groups.items()):
        row = {key: value for key, value in zip(group_keys, key_tuple)}
        row["rows"] = len(group)
        for metric in metric_keys:
            values = [float(item[metric]) for item in group if metric in item and item[metric] != ""]
            if values:
                row[metric] = finite_mean(values)
        out.append(row)
    return out


def topk_overlap_sums(reference: np.ndarray, candidate: np.ndarray, *, key_count: int) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.int64)
    cand = np.asarray(candidate, dtype=np.int64)
    if ref.ndim != 2 or cand.ndim != 2 or ref.shape != cand.shape:
        raise ValueError(f"Expected matching [Q,K] arrays, got {ref.shape} and {cand.shape}")
    if ref.size == 0:
        return {
            "queries": 0,
            "topk": int(ref.shape[1]),
            "sum_overlap_count": 0.0,
            "sum_recall_at_k": 0.0,
            "sum_jaccard_at_k": 0.0,
        }

    max_index = int(max(ref.max(initial=0), cand.max(initial=0)))
    dense_key_count = max(int(key_count), max_index + 1)
    if ref.min(initial=0) < 0 or cand.min(initial=0) < 0:
        raise ValueError("Top-K arrays contain negative indices.")

    q_count, topk = ref.shape
    marker = np.zeros((q_count, dense_key_count), dtype=np.bool_)
    row_ids = np.arange(q_count)[:, None]
    marker[row_ids, ref] = True
    overlap = marker[row_ids, cand].sum(axis=1).astype(np.float64)
    union = (2.0 * float(topk)) - overlap
    return {
        "queries": int(q_count),
        "topk": int(topk),
        "sum_overlap_count": float(overlap.sum()),
        "sum_recall_at_k": float((overlap / float(max(topk, 1))).sum()),
        "sum_jaccard_at_k": float((overlap / np.maximum(union, 1.0)).sum()),
    }


def aggregate_overlap_rows(rows: Sequence[dict[str, Any]], *, group_keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key, "") for key in group_keys), []).append(row)

    out: list[dict[str, Any]] = []
    for key_tuple, group in sorted(groups.items()):
        query_total = sum(int(item["queries"]) for item in group)
        overlap_total = sum(float(item["sum_overlap_count"]) for item in group)
        recall_total = sum(float(item["sum_recall_at_k"]) for item in group)
        jaccard_total = sum(float(item["sum_jaccard_at_k"]) for item in group)
        row = {key: value for key, value in zip(group_keys, key_tuple)}
        row.update(
            {
                "records": len(group),
                "queries": int(query_total),
                "topk": int(group[0]["topk"]) if group else 0,
                "key_count": finite_mean([float(item["key_count"]) for item in group]),
                "mean_overlap_count": float(overlap_total / query_total) if query_total else float("nan"),
                "mean_recall_at_k": float(recall_total / query_total) if query_total else float("nan"),
                "mean_jaccard_at_k": float(jaccard_total / query_total) if query_total else float("nan"),
            }
        )
        out.append(row)
    return out


def iter_record_manifest(setting_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    for manifest in sorted(setting_dir.glob("shard_*/records/topk_records_manifest.jsonl")):
        record_root = manifest.parent
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            out.append((record_root / row["file"], row))
    return out


def summarize_overlap(setting_dir: Path, *, reference_key: str, candidate_key: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    merged_manifest_rows: list[dict[str, Any]] = []
    for npz_path, manifest_row in iter_record_manifest(setting_dir):
        data = np.load(npz_path)
        if reference_key not in data.files or candidate_key not in data.files:
            raise KeyError(f"{npz_path} missing {reference_key} or {candidate_key}; keys={data.files}")
        key_count = int(manifest_row["tokens_per_view"]) * int(manifest_row["num_views"])
        sums = topk_overlap_sums(data[reference_key], data[candidate_key], key_count=key_count)
        row = {
            "setting": str(manifest_row["setting"]),
            "sample_id": str(manifest_row["sample_id"]),
            "variant": str(manifest_row["variant"]),
            "layer": str(manifest_row["layer"]),
            "query_scope": str(manifest_row["query_scope"]),
            "batch_index": int(manifest_row["batch_index"]),
            "reference_model": reference_key,
            "candidate_model": candidate_key,
            "key_count": int(key_count),
        }
        row.update(sums)
        rows.append(row)

        merged_manifest_row = dict(manifest_row)
        merged_manifest_row["record_file_absolute"] = str(npz_path)
        merged_manifest_rows.append(merged_manifest_row)

    aggregate = {
        "by_setting": aggregate_overlap_rows(rows, group_keys=("setting", "variant", "query_scope")),
        "by_layer": aggregate_overlap_rows(rows, group_keys=("setting", "variant", "query_scope", "layer")),
    }
    merged_dir = setting_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    (merged_dir / "topk_records_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged_manifest_rows),
        encoding="utf-8",
    )
    return rows, aggregate


def merge_role_rows(setting_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(setting_dir.glob("shard_*/output/topk_role_rows.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    aggregate = {
        "by_setting": aggregate_numeric_rows(
            rows,
            group_keys=(
                "experiment",
                "setting",
                "variant",
                "query_scope",
                "model",
                "target_name",
                "target_definition",
                "query_view_group",
            ),
            metric_keys=(
                "queries",
                "mean_same_view_group_ratio",
                "mean_cross_view_group_ratio",
                "mean_target_role_ratio",
            ),
        )
    }
    return rows, aggregate


def write_summary_markdown(path: Path, overlap_aggregate: dict[str, Any], role_aggregate: dict[str, Any]) -> None:
    lines = [
        "# GeoWeave Dense-Teacher Top-K vs Scorer Top-K",
        "",
        "## Selection Overlap",
        "",
        "| Setting | Variant | Query scope | Records | Queries | Recall@K | Jaccard@K | Overlap count |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in overlap_aggregate["by_setting"]:
        lines.append(
            "| {setting} | {variant} | {scope} | {records} | {queries} | {recall:.6f} | {jaccard:.6f} | {overlap:.3f} |".format(
                setting=row.get("setting", ""),
                variant=row.get("variant", ""),
                scope=row.get("query_scope", ""),
                records=int(row.get("records", 0)),
                queries=int(row.get("queries", 0)),
                recall=float(row.get("mean_recall_at_k", float("nan"))),
                jaccard=float(row.get("mean_jaccard_at_k", float("nan"))),
                overlap=float(row.get("mean_overlap_count", float("nan"))),
            )
        )
    lines.extend(
        [
            "",
            "## Role Ratios",
            "",
            "| Experiment | Setting | Variant | Query scope | Model | Target | Query group | Rows | Ratio |",
            "|---|---|---|---|---|---|---|---:|---:|",
        ]
    )
    for row in role_aggregate["by_setting"]:
        ratio = row.get("mean_same_view_group_ratio", row.get("mean_target_role_ratio", float("nan")))
        lines.append(
            "| {experiment} | {setting} | {variant} | {scope} | {model} | {target} | {group} | {rows} | {ratio:.6f} |".format(
                experiment=row.get("experiment", ""),
                setting=row.get("setting", ""),
                variant=row.get("variant", ""),
                scope=row.get("query_scope", ""),
                model=row.get("model", ""),
                target=row.get("target_definition", ""),
                group=row.get("query_view_group", ""),
                rows=int(row.get("rows", 0)),
                ratio=float(ratio),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_setting(root: Path, setting: str, *, reference_key: str, candidate_key: str) -> dict[str, Any]:
    setting_dir = root / setting
    merged_dir = setting_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    overlap_rows, overlap_aggregate = summarize_overlap(
        setting_dir,
        reference_key=reference_key,
        candidate_key=candidate_key,
    )
    role_rows, role_aggregate = merge_role_rows(setting_dir)

    (merged_dir / "teacher_student_topk_overlap_rows.json").write_text(
        json.dumps(overlap_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(merged_dir / "teacher_student_topk_overlap_rows.csv", overlap_rows)
    (merged_dir / "teacher_student_topk_overlap_aggregate.json").write_text(
        json.dumps(overlap_aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (merged_dir / "topk_role_rows.json").write_text(
        json.dumps(role_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(merged_dir / "topk_role_rows.csv", role_rows)
    (merged_dir / "topk_role_aggregate.json").write_text(
        json.dumps(role_aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_summary_markdown(merged_dir / "teacher_student_summary.md", overlap_aggregate, role_aggregate)
    return {
        "setting": setting,
        "overlap_rows": len(overlap_rows),
        "role_rows": len(role_rows),
        "merged_dir": str(merged_dir),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    summaries = [
        summarize_setting(root, setting, reference_key=args.reference_key, candidate_key=args.candidate_key)
        for setting in args.settings
    ]
    print(json.dumps({"root": str(root), "settings": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
