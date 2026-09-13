#!/usr/bin/env python3
"""Build a paired ScanNet++ 5+5 sweep across fixed cross-group overlap bands."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_repo_root_on_syspath() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


ensure_repo_root_on_syspath()

from aidi.scripts.baselines.build_evc_same_scene_candidate_pool_benchmark import (  # noqa: E402
    generic_scene_slug,
    parse_scene_specs,
    resolve_scene_roots,
)
from aidi.scripts.baselines.build_same_scene_low_overlap_5plus5_benchmark import (  # noqa: E402
    bidirectional_overlap,
    choose_local_cluster,
    cross_group_stats,
    frame_ids,
    internal_overlap_mean,
    load_scene_record_for_protocol,
)
from aidi.scripts.baselines.overlap_noise_seq_map_utils import (  # noqa: E402
    build_scene_overlap_table,
    subsample_scene_record,
)


BAND_SPECS = OrderedDict(
    [
        ("high", (0.10, 0.20)),
        ("medium", (0.03, 0.10)),
        ("low", (0.005, 0.03)),
        ("near_zero", (0.0, 0.005)),
    ]
)
FIXED_QUANTILE_LEVELS = OrderedDict(
    [
        ("highest", 0.85),
        ("medium", 0.60),
        ("low", 0.30),
        ("lowest", 0.05),
    ]
)

DEFAULT_SCANNETPP_ROOT = "/mnt/cfs/datasets/scannetpp/scannetpplus/Scannetpp/data"


def classify_overlap_band(value: float) -> str | None:
    value = float(value)
    for name, (lower, upper) in BAND_SPECS.items():
        if float(lower) <= value < float(upper):
            return name
    return None


def uniform_frame_indices(total: int, target: int) -> list[int]:
    total = int(total)
    target = int(target)
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if target <= 0 or target >= total:
        return list(range(total))
    return [int(item) for item in np.rint(np.linspace(0, total - 1, target)).astype(np.int64).tolist()]


def merge_uniform_with_required_frames(total: int, target: int, required: Sequence[int]) -> list[int]:
    required_ids = [int(item) for item in required]
    if any(item < 0 or item >= int(total) for item in required_ids):
        raise IndexError(f"Required frame outside [0, {int(total)}): {required_ids}")
    return sorted(set(uniform_frame_indices(int(total), int(target))).union(required_ids))


def combined_nerfstudio_frames(transforms: Mapping[str, Any]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for frame in [*list(transforms.get("frames", [])), *list(transforms.get("test_frames", []))]:
        image_name = Path(str(frame["file_path"])).name
        if image_name in seen_names:
            continue
        seen_names.add(image_name)
        frames.append(dict(frame))
    return frames


def nerfstudio_c2w_to_opencv(c2w_gl: np.ndarray) -> np.ndarray:
    c2w_gl = np.asarray(c2w_gl, dtype=np.float64)
    if c2w_gl.shape != (4, 4):
        raise ValueError(f"Expected 4x4 Nerfstudio pose, got {c2w_gl.shape}")
    return c2w_gl @ np.diag([1.0, -1.0, -1.0, 1.0])


def select_group_b_by_band(
    group_a_ids: Sequence[int],
    candidates: Sequence[Mapping[str, Any]],
) -> OrderedDict[str, Dict[str, Any]] | None:
    group_a_set = {int(item) for item in group_a_ids}
    buckets: Dict[str, list[Dict[str, Any]]] = {name: [] for name in BAND_SPECS}
    for source in candidates:
        row = dict(source)
        group_b_ids = [int(item) for item in row["group_b_ids"]]
        if group_a_set.intersection(group_b_ids):
            continue
        band = classify_overlap_band(float(row["cross_overlap_mean"]))
        if band is not None:
            row["group_b_ids"] = group_b_ids
            buckets[band].append(row)

    selected: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    for band, (lower, upper) in BAND_SPECS.items():
        rows = buckets[band]
        if not rows:
            return None
        midpoint = 0.5 * (float(lower) + float(upper))
        rows.sort(
            key=lambda row: (
                round(abs(float(row["cross_overlap_mean"]) - midpoint), 12),
                -float(row["group_b_internal_overlap_mean"]),
                int(row["anchor_b"]),
                tuple(int(item) for item in row["group_b_ids"]),
            )
        )
        selected[band] = rows[0]
    return selected


def select_group_b_by_quantiles(
    candidates: Sequence[Mapping[str, Any]],
) -> OrderedDict[str, Dict[str, Any]]:
    if len(candidates) < len(FIXED_QUANTILE_LEVELS):
        raise ValueError(
            f"Need at least {len(FIXED_QUANTILE_LEVELS)} candidates for quantile levels, got {len(candidates)}"
        )
    overlaps = np.asarray([float(row["cross_overlap_mean"]) for row in candidates], dtype=np.float64)
    unique_overlaps = np.unique(overlaps)
    if len(unique_overlaps) < len(FIXED_QUANTILE_LEVELS):
        raise ValueError(f"Need at least four distinct overlap values, got {unique_overlaps.tolist()}")
    used: set[int] = set()
    used_values: set[float] = set()
    chosen_rows: list[Dict[str, Any]] = []
    for _level, quantile in FIXED_QUANTILE_LEVELS.items():
        target = float(np.quantile(unique_overlaps, float(quantile)))
        available = [
            index
            for index in range(len(candidates))
            if index not in used and float(candidates[index]["cross_overlap_mean"]) not in used_values
        ]
        chosen = min(
            available,
            key=lambda index: (
                abs(float(candidates[index]["cross_overlap_mean"]) - target),
                -float(candidates[index]["group_b_internal_overlap_mean"]),
                int(candidates[index]["anchor_b"]),
            ),
        )
        used.add(chosen)
        used_values.add(float(candidates[chosen]["cross_overlap_mean"]))
        row = dict(candidates[chosen])
        row["quantile_target"] = float(quantile)
        row["target_overlap"] = target
        chosen_rows.append(row)
    chosen_rows.sort(key=lambda row: float(row["cross_overlap_mean"]), reverse=True)
    selected: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    for (level, quantile), row in zip(FIXED_QUANTILE_LEVELS.items(), chosen_rows):
        row["quantile_target"] = float(quantile)
        row["target_overlap"] = float(np.quantile(unique_overlaps, float(quantile)))
        selected[level] = row
    return selected


def selection_rows_for_anchor(
    scene_name: str,
    anchor_a: int,
    group_a_ids: Sequence[int],
    selected_by_band: Mapping[str, Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    fixed_group_a = [int(item) for item in group_a_ids]
    rows: list[Dict[str, Any]] = []
    for band in BAND_SPECS:
        candidate = dict(selected_by_band[band])
        rows.append(
            {
                "scene_name": str(scene_name),
                "anchor_a": int(anchor_a),
                "overlap_level": band,
                "group_a_ids": list(fixed_group_a),
                **candidate,
            }
        )
    return rows


def _rows_by_anchor(rows: Sequence[Mapping[str, Any]]) -> Dict[tuple[str, int], list[Dict[str, Any]]]:
    grouped: Dict[tuple[str, int], list[Dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        grouped[(str(row["scene_name"]), int(row["anchor_a"]))].append(row)
    return dict(grouped)


def select_balanced_anchor_rows(
    rows: Sequence[Mapping[str, Any]],
    target_anchors: int,
    max_anchors_per_scene: int,
) -> list[Dict[str, Any]]:
    if int(target_anchors) <= 0:
        raise ValueError(f"target_anchors must be positive, got {target_anchors}")
    if int(max_anchors_per_scene) <= 0:
        raise ValueError(f"max_anchors_per_scene must be positive, got {max_anchors_per_scene}")

    grouped = _rows_by_anchor(rows)
    anchors_by_scene: Dict[str, list[tuple[int, list[Dict[str, Any]]]]] = defaultdict(list)
    for (scene_name, anchor_a), anchor_rows in grouped.items():
        if {str(row["overlap_level"]) for row in anchor_rows} != set(BAND_SPECS):
            continue
        anchors_by_scene[scene_name].append((anchor_a, anchor_rows))
    for scene_name in anchors_by_scene:
        anchors_by_scene[scene_name].sort(key=lambda item: item[0])

    scene_names = sorted(anchors_by_scene)
    allocation = {scene_name: 0 for scene_name in scene_names}
    remaining = int(target_anchors)
    for _ in range(int(max_anchors_per_scene)):
        for scene_name in scene_names:
            if remaining == 0:
                break
            if allocation[scene_name] >= min(int(max_anchors_per_scene), len(anchors_by_scene[scene_name])):
                continue
            allocation[scene_name] += 1
            remaining -= 1
        if remaining == 0:
            break

    quantile_groups_by_scene: Dict[str, list[tuple[int, list[Dict[str, Any]]]]] = {}
    for scene_name, scene_groups in anchors_by_scene.items():
        keep = allocation[scene_name]
        if keep == 0:
            quantile_groups_by_scene[scene_name] = []
        elif keep == 1:
            quantile_groups_by_scene[scene_name] = [scene_groups[int(round((len(scene_groups) - 1) * 0.5))]]
        else:
            indices = uniform_frame_indices(len(scene_groups), keep)
            quantile_groups_by_scene[scene_name] = [scene_groups[index] for index in indices]

    selected_groups: list[list[Dict[str, Any]]] = []
    for round_index in range(int(max_anchors_per_scene)):
        for scene_name in scene_names:
            scene_groups = quantile_groups_by_scene[scene_name]
            if round_index >= len(scene_groups):
                continue
            selected_groups.append(scene_groups[round_index][1])
            if len(selected_groups) == int(target_anchors):
                break
        if len(selected_groups) == int(target_anchors):
            break

    ordered_rows: list[Dict[str, Any]] = []
    level_index = {level: index for index, level in enumerate(BAND_SPECS)}
    for anchor_rows in selected_groups:
        ordered_rows.extend(sorted(anchor_rows, key=lambda row: level_index[str(row["overlap_level"])]))
    return ordered_rows


def validate_selection_rows(
    rows: Sequence[Mapping[str, Any]],
    min_anchors: int,
    local_overlap_threshold: float,
) -> Dict[str, Any]:
    grouped = _rows_by_anchor(rows)
    if len(grouped) < int(min_anchors):
        raise ValueError(f"Expected at least {int(min_anchors)} balanced anchors, got {len(grouped)}")

    for key, anchor_rows in sorted(grouped.items()):
        levels = [str(row["overlap_level"]) for row in anchor_rows]
        if set(levels) != set(BAND_SPECS) or len(levels) != len(BAND_SPECS):
            raise ValueError(f"Anchor {key} does not contain exactly one row for every overlap level: {levels}")
        fixed_group_a = [int(item) for item in anchor_rows[0]["group_a_ids"]]
        if len(fixed_group_a) != 5 or len(set(fixed_group_a)) != 5:
            raise ValueError(f"Anchor {key} must have five unique Group A frames: {fixed_group_a}")
        for row in anchor_rows:
            group_a = [int(item) for item in row["group_a_ids"]]
            group_b = [int(item) for item in row["group_b_ids"]]
            if group_a != fixed_group_a:
                raise ValueError(f"Anchor {key} does not keep Group A fixed across levels")
            if len(group_b) != 5 or len(set(group_b)) != 5:
                raise ValueError(f"Anchor {key} level {row['overlap_level']} must have five unique Group B frames")
            if set(group_a).intersection(group_b):
                raise ValueError(f"Anchor {key} level {row['overlap_level']} has overlap between Group A and Group B frame IDs")
            if min(
                float(row["group_a_anchor_support_min"]),
                float(row["group_b_anchor_support_min"]),
            ) < float(local_overlap_threshold):
                raise ValueError(f"Anchor {key} level {row['overlap_level']} violates the local-overlap threshold")
            actual_band = classify_overlap_band(float(row["cross_overlap_mean"]))
            if actual_band != str(row["overlap_level"]):
                raise ValueError(
                    f"Anchor {key} level {row['overlap_level']} has mean overlap {row['cross_overlap_mean']} outside declared band"
                )

    return {
        "num_anchors": int(len(grouped)),
        "num_rows": int(len(rows)),
        "num_scenes": int(len({scene_name for scene_name, _ in grouped})),
        "levels": list(BAND_SPECS),
    }


def validate_fixed_quantile_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_anchors: int,
    local_overlap_threshold: float,
) -> Dict[str, Any]:
    grouped: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["anchor_name"])].append(row)
    if len(grouped) != int(expected_anchors):
        raise ValueError(f"Expected {int(expected_anchors)} fixed anchors, got {len(grouped)}")
    for anchor_name, anchor_rows in grouped.items():
        by_level = {str(row["overlap_level"]): row for row in anchor_rows}
        if set(by_level) != set(FIXED_QUANTILE_LEVELS) or len(anchor_rows) != len(FIXED_QUANTILE_LEVELS):
            raise ValueError(f"Anchor {anchor_name} does not contain all quantile levels")
        ordered = [by_level[level] for level in FIXED_QUANTILE_LEVELS]
        fixed_group_a = [int(item) for item in ordered[0]["group_a_ids"]]
        for row in ordered:
            group_a = [int(item) for item in row["group_a_ids"]]
            group_b = [int(item) for item in row["group_b_ids"]]
            if group_a != fixed_group_a:
                raise ValueError(f"Anchor {anchor_name} changes Group A across levels")
            if len(group_b) != 5 or len(set(group_b)) != 5 or set(group_a).intersection(group_b):
                raise ValueError(f"Anchor {anchor_name} has invalid Group B at {row['overlap_level']}")
            if float(row["group_b_anchor_support_min"]) < float(local_overlap_threshold):
                raise ValueError(f"Anchor {anchor_name} violates Group B local-overlap threshold")
        means = [float(row["cross_overlap_mean"]) for row in ordered]
        if not all(lhs > rhs for lhs, rhs in zip(means, means[1:])):
            raise ValueError(f"Anchor {anchor_name} quantile levels are not strictly ordered: {means}")
    return {
        "num_anchors": int(len(grouped)),
        "num_rows": int(len(rows)),
        "levels": list(FIXED_QUANTILE_LEVELS),
    }


def anchor_support_min(
    group_ids: Sequence[int],
    anchor_id: int,
    overlap_table: Mapping[int, Mapping[int, float]],
) -> float:
    support_values = [
        bidirectional_overlap(overlap_table, int(anchor_id), int(frame_id))
        for frame_id in group_ids
        if int(frame_id) != int(anchor_id)
    ]
    return float(min(support_values)) if support_values else 0.0


def enumerate_scene_candidates(
    scene_name: str,
    scene_record: Mapping[str, Any],
    overlap_table: Mapping[int, Mapping[int, float]],
    local_overlap_threshold: float,
    anchor_stride: int,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    if int(anchor_stride) <= 0:
        raise ValueError(f"anchor_stride must be positive, got {anchor_stride}")
    ids = sorted(frame_ids(scene_record))
    candidate_rows: list[Dict[str, Any]] = []
    complete_rows: list[Dict[str, Any]] = []

    for anchor_a in ids[:: int(anchor_stride)]:
        group_a = choose_local_cluster(
            anchor_id=anchor_a,
            overlap_table=overlap_table,
            all_ids=ids,
            count=5,
            local_overlap_threshold=float(local_overlap_threshold),
        )
        if group_a is None:
            continue
        group_a_min = anchor_support_min(group_a, anchor_a, overlap_table)
        group_a_internal = internal_overlap_mean(group_a, overlap_table)
        per_anchor_candidates: list[Dict[str, Any]] = []
        group_a_set = set(group_a)
        for anchor_b in ids:
            if int(anchor_b) in group_a_set:
                continue
            group_b = choose_local_cluster(
                anchor_id=anchor_b,
                overlap_table=overlap_table,
                all_ids=ids,
                count=5,
                local_overlap_threshold=float(local_overlap_threshold),
                excluded_ids=group_a_set,
            )
            if group_b is None:
                continue
            stats = cross_group_stats(group_a, group_b, overlap_table)
            band = classify_overlap_band(float(stats["cross_overlap_mean"]))
            if band is None:
                continue
            row = {
                "scene_name": str(scene_name),
                "anchor_a": int(anchor_a),
                "anchor_b": int(anchor_b),
                "overlap_level": band,
                "group_a_ids": [int(item) for item in group_a],
                "group_b_ids": [int(item) for item in group_b],
                "group_a_anchor_support_min": float(group_a_min),
                "group_b_anchor_support_min": anchor_support_min(group_b, anchor_b, overlap_table),
                "group_a_internal_overlap_mean": float(group_a_internal),
                "group_b_internal_overlap_mean": internal_overlap_mean(group_b, overlap_table),
                "anchor_pair_overlap": bidirectional_overlap(overlap_table, anchor_a, anchor_b),
                **stats,
            }
            candidate_rows.append(row)
            per_anchor_candidates.append(row)

        selected = select_group_b_by_band(group_a, per_anchor_candidates)
        if selected is not None:
            complete_rows.extend(
                selection_rows_for_anchor(
                    scene_name=scene_name,
                    anchor_a=anchor_a,
                    group_a_ids=group_a,
                    selected_by_band=selected,
                )
            )
    return candidate_rows, complete_rows


def enumerate_fixed_group_a_candidates(
    scene_name: str,
    fixed_anchor: Mapping[str, Any],
    scene_record: Mapping[str, Any],
    overlap_table: Mapping[int, Mapping[int, float]],
    local_overlap_threshold: float,
    level_mode: str = "absolute",
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    ids = sorted(frame_ids(scene_record))
    group_a = [int(item) for item in fixed_anchor["group_a_ids"]]
    if len(group_a) != 5 or len(set(group_a)) != 5:
        raise ValueError(f"Fixed Group A must contain five unique views: {group_a}")
    if not set(group_a).issubset(ids):
        raise ValueError(f"Fixed Group A is outside the scanned scene record: {group_a}")
    anchor_a = int(group_a[0])
    group_a_min = anchor_support_min(group_a, anchor_a, overlap_table)
    group_a_internal = internal_overlap_mean(group_a, overlap_table)
    group_a_set = set(group_a)
    candidates: list[Dict[str, Any]] = []
    for anchor_b in ids:
        if int(anchor_b) in group_a_set:
            continue
        group_b = choose_local_cluster(
            anchor_id=anchor_b,
            overlap_table=overlap_table,
            all_ids=ids,
            count=5,
            local_overlap_threshold=float(local_overlap_threshold),
            excluded_ids=group_a_set,
        )
        if group_b is None:
            continue
        stats = cross_group_stats(group_a, group_b, overlap_table)
        band = classify_overlap_band(float(stats["cross_overlap_mean"]))
        if str(level_mode) == "absolute" and band is None:
            continue
        candidates.append(
            {
                "scene_name": str(scene_name),
                "anchor_name": str(fixed_anchor["anchor_name"]),
                "legacy_tuple_name": str(fixed_anchor["legacy_tuple_name"]),
                "legacy_anchor_a": int(fixed_anchor["legacy_anchor_a"]),
                "anchor_a": anchor_a,
                "anchor_b": int(anchor_b),
                "overlap_level": band or "candidate",
                "group_a_ids": list(group_a),
                "group_b_ids": [int(item) for item in group_b],
                "group_a_anchor_support_min": float(group_a_min),
                "group_b_anchor_support_min": anchor_support_min(group_b, anchor_b, overlap_table),
                "group_a_internal_overlap_mean": float(group_a_internal),
                "group_b_internal_overlap_mean": internal_overlap_mean(group_b, overlap_table),
                "anchor_pair_overlap": bidirectional_overlap(overlap_table, anchor_a, anchor_b),
                **stats,
            }
        )
    if str(level_mode) == "absolute":
        selected = select_group_b_by_band(group_a, candidates)
        if selected is None:
            return candidates, []
        rows = selection_rows_for_anchor(
            scene_name=scene_name,
            anchor_a=anchor_a,
            group_a_ids=group_a,
            selected_by_band=selected,
        )
    elif str(level_mode) == "quantile":
        if len(candidates) < len(FIXED_QUANTILE_LEVELS):
            return candidates, []
        selected = select_group_b_by_quantiles(candidates)
        rows = []
        for level in FIXED_QUANTILE_LEVELS:
            candidate = dict(selected[level])
            candidate.pop("overlap_level", None)
            rows.append(
                {
                    "scene_name": str(scene_name),
                    "anchor_a": int(anchor_a),
                    "overlap_level": level,
                    "group_a_ids": list(group_a),
                    **candidate,
                }
            )
    else:
        raise ValueError(f"Unsupported fixed Group A level_mode={level_mode!r}")
    for row in rows:
        row.update(
            {
                "anchor_name": str(fixed_anchor["anchor_name"]),
                "legacy_tuple_name": str(fixed_anchor["legacy_tuple_name"]),
                "legacy_anchor_a": int(fixed_anchor["legacy_anchor_a"]),
            }
        )
    return candidates, rows


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=isinstance(value, dict))
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def decode_complete_anchor_csv_row(source: Mapping[str, str]) -> Dict[str, Any]:
    row: Dict[str, Any] = dict(source)
    for key in ("anchor_a", "anchor_b"):
        if key in row and row[key] != "":
            row[key] = int(row[key])
    for key in ("group_a_ids", "group_b_ids"):
        if key in row and row[key] != "":
            row[key] = [int(item) for item in json.loads(row[key])]
    for key in (
        "group_a_anchor_support_min",
        "group_b_anchor_support_min",
        "group_a_internal_overlap_mean",
        "group_b_internal_overlap_mean",
        "anchor_pair_overlap",
        "cross_overlap_mean",
        "cross_overlap_max",
        "cross_overlap_min",
    ):
        if key in row and row[key] != "":
            row[key] = float(row[key])
    return row


def reselect_from_preflight(
    preflight_root: Path,
    target_anchors: int,
    min_anchors: int,
    max_anchors_per_scene: int,
    local_overlap_threshold: float,
) -> Dict[str, Any]:
    preflight_root = Path(preflight_root).expanduser().resolve()
    complete_rows_path = preflight_root / "complete_anchor_rows.csv"
    with complete_rows_path.open("r", newline="", encoding="utf-8") as handle:
        complete_rows = [decode_complete_anchor_csv_row(row) for row in csv.DictReader(handle)]
    selected_rows = select_balanced_anchor_rows(
        complete_rows,
        target_anchors=int(target_anchors),
        max_anchors_per_scene=int(max_anchors_per_scene),
    )
    validation = validate_selection_rows(
        selected_rows,
        min_anchors=int(min_anchors),
        local_overlap_threshold=float(local_overlap_threshold),
    )
    if validation["num_anchors"] != int(target_anchors):
        raise ValueError(f"Expected exactly {int(target_anchors)} reselected anchors, got {validation['num_anchors']}")

    manifest_path = preflight_root / "selection_manifest.json"
    summary_path = preflight_root / "preflight_summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy2(manifest_path, preflight_root / f"selection_manifest.before-quantile-{timestamp}.json")
    manifest["rows"] = selected_rows
    manifest["selection_strategy"] = "per_scene_temporal_quantiles_then_round_robin_v2"
    manifest["reselected_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summary["num_selected_anchors"] = int(validation["num_anchors"])
    summary["num_selected_rows"] = int(validation["num_rows"])
    summary["selection_strategy"] = manifest["selection_strategy"]
    summary["validation"] = validation
    summary["reselected_at"] = manifest["reselected_at"]
    write_json(manifest_path, manifest)
    write_json(summary_path, summary)
    return {
        "preflight_root": str(preflight_root),
        "selection_strategy": manifest["selection_strategy"],
        "num_anchors": int(validation["num_anchors"]),
        "num_rows": int(validation["num_rows"]),
        "num_scenes": int(validation["num_scenes"]),
        "anchors": sorted({(str(row["scene_name"]), int(row["anchor_a"])) for row in selected_rows}),
    }


def run_fixed_group_a_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.fixed_group_a_manifest:
        raise ValueError("--fixed-group-a-manifest is required for --mode fixed-preflight")
    fixed_manifest_path = Path(args.fixed_group_a_manifest).expanduser().resolve()
    fixed_manifest = json.loads(fixed_manifest_path.read_text(encoding="utf-8"))
    anchors = [dict(anchor) for anchor in fixed_manifest["anchors"]]
    level_names = list(FIXED_QUANTILE_LEVELS) if str(args.fixed_level_mode) == "quantile" else list(BAND_SPECS)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    anchors_by_scene: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for anchor in anchors:
        anchors_by_scene[str(anchor["scene_name"])].append(anchor)

    all_candidates: list[Dict[str, Any]] = []
    all_complete_rows: list[Dict[str, Any]] = []
    scene_summaries: list[Dict[str, Any]] = []
    anchor_summaries: list[Dict[str, Any]] = []
    for scene_name in sorted(anchors_by_scene):
        scene_root = dataset_root / scene_name
        required_ids = sorted(
            {
                int(frame_id)
                for anchor in anchors_by_scene[scene_name]
                for frame_id in anchor["group_a_ids"]
            }
        )
        started = time.time()
        scene_record = load_official_scannetpp_scene_record(
            scene_root=scene_root,
            target_num_frames=int(args.target_num_frames),
            render_height=int(args.official_render_height),
            render_cache_root=output_root / "rendered_depth",
            required_frame_ids=required_ids,
        )
        overlap_table = build_scene_overlap_table(
            scene_record=scene_record,
            sample_stride=int(args.overlap_sample_stride),
            depth_rel_tol=float(args.depth_rel_tol),
        )
        scene_candidate_count = 0
        scene_complete_count = 0
        for anchor in sorted(anchors_by_scene[scene_name], key=lambda item: str(item["anchor_name"])):
            candidates, complete_rows = enumerate_fixed_group_a_candidates(
                scene_name=scene_name,
                fixed_anchor=anchor,
                scene_record=scene_record,
                overlap_table=overlap_table,
                local_overlap_threshold=float(args.local_overlap_threshold),
                level_mode=str(args.fixed_level_mode),
            )
            all_candidates.extend(candidates)
            all_complete_rows.extend(complete_rows)
            scene_candidate_count += len(candidates)
            scene_complete_count += int(len(complete_rows) == len(level_names))
            band_counts = {
                level: int(sum(str(row["overlap_level"]) == level for row in candidates))
                for level in BAND_SPECS
            }
            anchor_summaries.append(
                {
                    "scene_name": scene_name,
                    "anchor_name": str(anchor["anchor_name"]),
                    "legacy_tuple_name": str(anchor["legacy_tuple_name"]),
                    "group_a_ids": [int(item) for item in anchor["group_a_ids"]],
                    "group_a_anchor_support_min": (
                        float(candidates[0]["group_a_anchor_support_min"]) if candidates else float("nan")
                    ),
                    "candidate_band_counts": band_counts,
                    "candidate_overlap_min": (
                        float(min(row["cross_overlap_mean"] for row in candidates)) if candidates else float("nan")
                    ),
                    "candidate_overlap_max": (
                        float(max(row["cross_overlap_mean"] for row in candidates)) if candidates else float("nan")
                    ),
                    "selected_overlaps": {
                        str(row["overlap_level"]): float(row["cross_overlap_mean"])
                        for row in complete_rows
                    },
                    "complete": bool(len(complete_rows) == len(level_names)),
                }
            )
        scene_summaries.append(
            {
                "scene_name": scene_name,
                "source_frame_count": int(scene_record["source_frame_count"]),
                "scanned_frame_count": int(len(frame_ids(scene_record))),
                "required_group_a_frames": required_ids,
                "num_fixed_anchors": int(len(anchors_by_scene[scene_name])),
                "num_complete_anchors": int(scene_complete_count),
                "num_candidate_rows": int(scene_candidate_count),
                "elapsed_seconds": float(time.time() - started),
            }
        )
        print(
            f"[fixed-overlap-preflight] scene={scene_name} anchors={len(anchors_by_scene[scene_name])} "
            f"complete={scene_complete_count} candidates={scene_candidate_count}",
            flush=True,
        )

    complete_anchor_names = {str(row["anchor_name"]) for row in all_complete_rows}
    expected_anchor_names = {str(anchor["anchor_name"]) for anchor in anchors}
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": "original_group_a_paired_overlap_sweep_v1",
        "levels": level_names,
        "level_mode": str(args.fixed_level_mode),
        "fixed_group_a_manifest": str(fixed_manifest_path),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "config": vars(args),
        "num_scenes": int(len(anchors_by_scene)),
        "num_fixed_anchors": int(len(anchors)),
        "num_complete_anchors": int(len(complete_anchor_names)),
        "num_candidate_rows": int(len(all_candidates)),
        "num_selected_rows": int(len(all_complete_rows)),
        "missing_anchor_names": sorted(expected_anchor_names - complete_anchor_names),
        "scenes": scene_summaries,
        "anchors": anchor_summaries,
    }
    selection_manifest = {
        "protocol": "original_group_a_paired_overlap_sweep_v1",
        "selection_strategy": "fixed_original_group_a_viewpoints_v1",
        "levels": level_names,
        "level_mode": str(args.fixed_level_mode),
        "fixed_group_a_manifest": str(fixed_manifest_path),
        "dataset_root": str(dataset_root),
        "config": vars(args),
        "rows": all_complete_rows,
    }
    write_csv(output_root / "candidate_rows.csv", all_candidates)
    write_csv(output_root / "complete_anchor_rows.csv", all_complete_rows)
    write_json(output_root / "selection_manifest.json", selection_manifest)
    write_json(output_root / "preflight_summary.json", summary)
    if complete_anchor_names != expected_anchor_names or len(all_complete_rows) != len(anchors) * len(level_names):
        raise RuntimeError(
            f"Fixed Group A preflight retained {len(complete_anchor_names)}/{len(anchors)} anchors; "
            f"diagnostics at {output_root}"
        )
    validation = (
        validate_fixed_quantile_rows(
            all_complete_rows,
            expected_anchors=len(anchors),
            local_overlap_threshold=float(args.local_overlap_threshold),
        )
        if str(args.fixed_level_mode) == "quantile"
        else validate_selection_rows(
            all_complete_rows,
            min_anchors=len(anchors),
            local_overlap_threshold=float(args.local_overlap_threshold),
        )
    )
    summary["validation"] = validation
    write_json(output_root / "preflight_summary.json", summary)
    return summary


def discover_official_scannetpp_scenes(
    dataset_root: Path,
    scene_specs: Sequence[str],
    limit_scenes: int,
) -> list[Path]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    if scene_specs:
        roots = [
            Path(spec).expanduser().resolve() if Path(spec).is_absolute() else (dataset_root / spec).resolve()
            for spec in scene_specs
        ]
    else:
        roots = sorted(
            path
            for path in dataset_root.iterdir()
            if path.is_dir()
            and (path / "dslr" / "nerfstudio" / "transforms.json").is_file()
            and (path / "scans" / "mesh_aligned_0.05.ply").is_file()
        )
    for root in roots:
        if not (root / "dslr" / "nerfstudio" / "transforms.json").is_file():
            raise FileNotFoundError(f"Missing Nerfstudio transforms under {root}")
        if not (root / "scans" / "mesh_aligned_0.05.ply").is_file():
            raise FileNotFoundError(f"Missing aligned mesh under {root}")
    return roots[: int(limit_scenes)] if int(limit_scenes) > 0 else roots


def render_pinhole_depth_cache(
    mesh_path: Path,
    poses_c2w: Sequence[np.ndarray],
    intrinsic: np.ndarray,
    frame_ids_to_render: Sequence[int],
    cache_root: Path,
    render_height: int,
    render_width: int,
) -> list[Path]:
    import open3d as o3d

    cache_root.mkdir(parents=True, exist_ok=True)
    depth_paths = [cache_root / f"frame_{int(frame_id):06d}.npy" for frame_id in frame_ids_to_render]
    pending = [index for index, path in enumerate(depth_paths) if not path.is_file()]
    if not pending:
        return depth_paths

    legacy_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(legacy_mesh.triangles) == 0:
        raise ValueError(f"Mesh has no triangles: {mesh_path}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh))

    ys, xs = np.meshgrid(
        np.arange(int(render_height), dtype=np.float32),
        np.arange(int(render_width), dtype=np.float32),
        indexing="ij",
    )
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    camera_directions = np.stack(
        [(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)],
        axis=-1,
    ).reshape(-1, 3)

    for index in pending:
        c2w = np.asarray(poses_c2w[index], dtype=np.float32)
        directions_world = camera_directions @ c2w[:3, :3].T
        origins_world = np.broadcast_to(c2w[:3, 3], directions_world.shape)
        rays = np.concatenate([origins_world, directions_world], axis=1).astype(np.float32, copy=False)
        hits = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(int(render_height), int(render_width))
        depth = np.asarray(hits, dtype=np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth <= 1.0e-4] = 0.0
        np.save(depth_paths[index], depth)
    return depth_paths


def load_official_scannetpp_scene_record(
    scene_root: Path,
    target_num_frames: int,
    render_height: int,
    render_cache_root: Path,
    required_frame_ids: Sequence[int] = (),
) -> Dict[str, Any]:
    scene_root = Path(scene_root).expanduser().resolve()
    transforms_path = scene_root / "dslr" / "nerfstudio" / "transforms.json"
    image_root = scene_root / "dslr" / "resized_images"
    mesh_path = scene_root / "scans" / "mesh_aligned_0.05.ply"
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = combined_nerfstudio_frames(transforms)
    selected_indices = merge_uniform_with_required_frames(
        len(frames),
        int(target_num_frames),
        required_frame_ids,
    )
    selected_frames = [frames[index] for index in selected_indices]
    color_paths = [image_root / Path(str(frame["file_path"])).name for frame in selected_frames]
    missing_images = [str(path) for path in color_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} official ScanNet++ DSLR images; first={missing_images[0]}")

    source_width = int(transforms["w"])
    source_height = int(transforms["h"])
    render_height = int(render_height)
    if render_height <= 0:
        raise ValueError(f"render_height must be positive, got {render_height}")
    render_width = int(round(float(render_height) * float(source_width) / float(source_height)))
    scale_x = float(render_width) / float(source_width)
    scale_y = float(render_height) / float(source_height)
    intrinsic = np.asarray(
        [
            [float(transforms["fl_x"]) * scale_x, 0.0, float(transforms["cx"]) * scale_x],
            [0.0, float(transforms["fl_y"]) * scale_y, float(transforms["cy"]) * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    poses = [
        nerfstudio_c2w_to_opencv(np.asarray(frame["transform_matrix"], dtype=np.float64)).astype(np.float32)
        for frame in selected_frames
    ]
    scene_cache_root = Path(render_cache_root) / scene_root.name / f"h{render_height}_w{render_width}"
    depth_paths = render_pinhole_depth_cache(
        mesh_path=mesh_path,
        poses_c2w=poses,
        intrinsic=intrinsic,
        frame_ids_to_render=selected_indices,
        cache_root=scene_cache_root,
        render_height=render_height,
        render_width=render_width,
    )
    cache_manifest = {
        "scene_name": scene_root.name,
        "source_scene_root": str(scene_root),
        "source_transforms": str(transforms_path),
        "source_mesh": str(mesh_path),
        "source_camera_model": transforms.get("camera_model", "unknown"),
        "virtual_camera_model": "PINHOLE",
        "source_frame_count": int(len(frames)),
        "selected_frame_indices": selected_indices,
        "render_height": render_height,
        "render_width": render_width,
        "intrinsic": intrinsic.astype(float).tolist(),
    }
    write_json(scene_cache_root / "render_manifest.json", cache_manifest)
    return {
        "scene_name": scene_root.name,
        "seq_root": scene_root,
        "frame_names": [Path(str(frame["file_path"])).name for frame in selected_frames],
        "frame_ids": selected_indices,
        "color_paths": color_paths,
        "depth_paths": depth_paths,
        "poses": np.stack(poses, axis=0).astype(np.float32),
        "intrinsics": [intrinsic.copy() for _ in selected_indices],
        "source_frame_count": int(len(frames)),
        "source_camera_model": transforms.get("camera_model", "unknown"),
        "virtual_camera_model": "PINHOLE",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tuple_name_for_row(row: Mapping[str, Any]) -> str:
    return f"{row['scene_name']}__a{int(row['anchor_a']):06d}__{row['overlap_level']}"


def materialize_from_manifest(
    manifest_path: Path,
    output_root: Path,
    dataset_name: str,
    overwrite: bool,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not bool(overwrite):
            raise FileExistsError(f"Materialization output already exists and is non-empty: {output_root}")
        backup = output_root.with_name(f"{output_root.name}.previous-{time.strftime('%Y%m%d-%H%M%S')}")
        output_root.rename(backup)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_root = Path(manifest["dataset_root"]).expanduser().resolve()
    level_names = [str(item) for item in manifest.get("levels", list(BAND_SPECS))]
    level_index = {level: index for index, level in enumerate(level_names)}
    rows = sorted(
        (dict(row) for row in manifest["rows"]),
        key=lambda row: (str(row["scene_name"]), int(row["anchor_a"]), level_index[str(row["overlap_level"])]),
    )
    dataset_output_root = output_root / str(dataset_name)
    dataset_output_root.mkdir(parents=True, exist_ok=True)
    scene_cache: Dict[str, Dict[str, Any]] = {}
    tuple_records: list[Dict[str, Any]] = []

    for row in rows:
        scene_name = str(row["scene_name"])
        if scene_name not in scene_cache:
            scene_root = dataset_root / scene_name
            transforms_path = scene_root / "dslr" / "nerfstudio" / "transforms.json"
            transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
            scene_cache[scene_name] = {
                "scene_root": scene_root,
                "frames": combined_nerfstudio_frames(transforms),
            }
        scene_payload = scene_cache[scene_name]
        scene_root = Path(scene_payload["scene_root"])
        frames = scene_payload["frames"]
        ordered_ids = [int(item) for item in row["group_a_ids"]] + [int(item) for item in row["group_b_ids"]]
        if len(ordered_ids) != 10 or len(set(ordered_ids)) != 10:
            raise ValueError(f"Expected ten unique frame IDs for {tuple_name_for_row(row)}: {ordered_ids}")

        tuple_name = tuple_name_for_row(row)
        tuple_root = dataset_output_root / tuple_name
        color_root = tuple_root / "color_90"
        color_root.mkdir(parents=True, exist_ok=False)
        poses: list[np.ndarray] = []
        source_image_paths: list[str] = []
        image_hashes: list[str] = []
        frame_rows: list[Dict[str, Any]] = []
        for out_index, frame_id in enumerate(ordered_ids):
            if frame_id < 0 or frame_id >= len(frames):
                raise IndexError(f"Frame ID {frame_id} outside transforms length {len(frames)} for {scene_name}")
            frame = frames[frame_id]
            source_image = scene_root / "dslr" / "resized_images" / Path(str(frame["file_path"])).name
            if not source_image.is_file():
                raise FileNotFoundError(source_image)
            target_image = color_root / f"frame_{out_index:04d}.jpg"
            target_image.symlink_to(source_image.resolve())
            image_hash = sha256_file(source_image)
            source_image_paths.append(str(source_image))
            image_hashes.append(image_hash)
            poses.append(nerfstudio_c2w_to_opencv(np.asarray(frame["transform_matrix"], dtype=np.float64)))
            frame_rows.append(
                {
                    "out_index": int(out_index),
                    "frame_id": int(frame_id),
                    "group": "A" if out_index < 5 else "B",
                    "role": "anchor" if frame_id in {int(row["anchor_a"]), int(row["anchor_b"])} else "support",
                    "source_image": str(source_image),
                    "sha256": image_hash,
                }
            )
        np.savetxt(tuple_root / "pose_90.txt", np.stack(poses, axis=0).reshape(10, 16), fmt="%.9f")
        tuple_meta = {
            "protocol": "scannetpp_paired_overlap_sweep_v1",
            "dataset_name": str(dataset_name),
            "scene_name": scene_name,
            "source_scene_root": str(scene_root),
            "tuple_name": tuple_name,
            "anchor_a": int(row["anchor_a"]),
            "anchor_b": int(row["anchor_b"]),
            "overlap_level": str(row["overlap_level"]),
            "group_a_ids": [int(item) for item in row["group_a_ids"]],
            "group_b_ids": [int(item) for item in row["group_b_ids"]],
            "ordered_frame_ids": ordered_ids,
            "eval_frame_indices": list(range(10)),
            "selection_metrics": {
                key: value
                for key, value in row.items()
                if key not in {"scene_name", "group_a_ids", "group_b_ids", "overlap_level"}
            },
            "frames": frame_rows,
        }
        write_json(tuple_root / "tuple_meta.json", tuple_meta)
        tuple_records.append(
            {
                "scene_name": scene_name,
                "anchor_a": int(row["anchor_a"]),
                "overlap_level": str(row["overlap_level"]),
                "tuple_name": tuple_name,
                "tuple_root": str(tuple_root),
                "group_a_ids": [int(item) for item in row["group_a_ids"]],
                "group_b_ids": [int(item) for item in row["group_b_ids"]],
                "cross_overlap_mean": float(row["cross_overlap_mean"]),
                "source_image_paths": source_image_paths,
                "group_a_sha256": image_hashes[:5],
                "group_b_sha256": image_hashes[5:],
            }
        )

    records_by_anchor: Dict[tuple[str, int], list[Dict[str, Any]]] = defaultdict(list)
    for record in tuple_records:
        records_by_anchor[(str(record["scene_name"]), int(record["anchor_a"]))].append(record)
    for key, records in records_by_anchor.items():
        if len(records) != len(level_names):
            raise ValueError(f"Anchor {key} materialized {len(records)} levels instead of {len(level_names)}")
        expected_hashes = records[0]["group_a_sha256"]
        if any(record["group_a_sha256"] != expected_hashes for record in records[1:]):
            raise ValueError(f"Group A image bytes differ across overlap levels for anchor {key}")

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "scannetpp_paired_overlap_sweep_v1",
        "manifest_path": str(manifest_path),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "dataset_name": str(dataset_name),
        "num_anchors": int(len(records_by_anchor)),
        "num_tuples": int(len(tuple_records)),
        "tuples": tuple_records,
    }
    protocol_samples: list[Dict[str, Any]] = []
    for (scene_name, anchor_a), records in sorted(records_by_anchor.items()):
        records_by_level = {str(record["overlap_level"]): record for record in records}
        variants: Dict[str, Any] = {}
        for level in level_names:
            record = records_by_level[level]
            tuple_root = Path(record["tuple_root"])
            variants[level] = {
                "variant": level,
                "cross_overlap_mean": float(record["cross_overlap_mean"]),
                "input_dir": str(tuple_root / "color_90"),
                "sequence_dir": str(tuple_root),
                "tuple_meta_path": str(tuple_root / "tuple_meta.json"),
                "frames": [
                    {
                        "index": int(index),
                        "file_name": f"frame_{index:04d}.jpg",
                        "image_path": str(tuple_root / "color_90" / f"frame_{index:04d}.jpg"),
                        "role": "group_a" if index < 5 else "group_b",
                    }
                    for index in range(10)
                ],
            }
        protocol_samples.append(
            {
                "sample_id": f"{scene_name}__a{int(anchor_a):06d}",
                "scene_name": scene_name,
                "anchor_a": int(anchor_a),
                "prefix_size": 5,
                "context_size": 5,
                "variants": variants,
            }
        )
    protocol = {
        "protocol_name": "scannetpp_paired_overlap_sweep_v1",
        "source_root": str(dataset_output_root),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "variants": level_names,
        "prefix_size": 5,
        "context_size": 5,
        "num_samples": int(len(protocol_samples)),
        "samples": protocol_samples,
    }
    shutil.copy2(manifest_path, output_root / "selection_manifest.json")
    write_json(output_root / "protocol.json", protocol)
    write_json(output_root / "materialization_validation.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("preflight", "fixed-preflight", "reselect", "materialize"),
        default="preflight",
    )
    parser.add_argument("--dataset-root", default=DEFAULT_SCANNETPP_ROOT)
    parser.add_argument("--dataset-name", default="scannetpp_overlap_sweep")
    parser.add_argument(
        "--source-layout",
        choices=("scannetpp_official_dslr", "nested_evc"),
        default="scannetpp_official_dslr",
    )
    parser.add_argument("--scene-specs", default="")
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--image-subdir", default="images")
    parser.add_argument("--depth-subdir", default="depths")
    parser.add_argument("--camera-subdir", default="cameras")
    parser.add_argument("--target-num-frames", type=int, default=256)
    parser.add_argument("--source-prestride", type=int, default=3)
    parser.add_argument("--official-render-height", type=int, default=128)
    parser.add_argument("--overlap-sample-stride", type=int, default=4)
    parser.add_argument("--depth-rel-tol", type=float, default=0.08)
    parser.add_argument("--local-overlap-threshold", type=float, default=0.08)
    parser.add_argument("--anchor-stride", type=int, default=1)
    parser.add_argument("--target-anchors", type=int, default=20)
    parser.add_argument("--min-anchors", type=int, default=20)
    parser.add_argument("--max-anchors-per-scene", type=int, default=2)
    parser.add_argument("--allow-insufficient-for-smoke", action="store_true")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--fixed-group-a-manifest", default="")
    parser.add_argument("--fixed-level-mode", choices=("absolute", "quantile"), default="absolute")
    parser.add_argument("--dataset-output-name", default="scannetpp_overlap_sweep")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scene_specs = parse_scene_specs(args.scene_specs)
    if str(args.source_layout) == "scannetpp_official_dslr":
        scene_roots = discover_official_scannetpp_scenes(
            dataset_root=dataset_root,
            scene_specs=scene_specs,
            limit_scenes=int(args.limit_scenes),
        )
    else:
        scene_roots = resolve_scene_roots(
            dataset_root=dataset_root,
            scene_specs=scene_specs,
            limit_scenes=int(args.limit_scenes),
            dataset_name=str(args.dataset_name),
            source_layout=str(args.source_layout),
            image_subdir=str(args.image_subdir),
            depth_subdir=str(args.depth_subdir),
            camera_subdir=str(args.camera_subdir),
        )
    if not scene_roots:
        raise FileNotFoundError(f"No ScanNet++ scenes found under {dataset_root}")

    all_candidates: list[Dict[str, Any]] = []
    all_complete_rows: list[Dict[str, Any]] = []
    scene_summaries: list[Dict[str, Any]] = []
    for scene_index, scene_root in enumerate(scene_roots):
        scene_name = scene_root.name if str(args.source_layout) == "scannetpp_official_dslr" else generic_scene_slug(scene_root, dataset_root)
        started = time.time()
        if str(args.source_layout) == "scannetpp_official_dslr":
            scene_record = load_official_scannetpp_scene_record(
                scene_root=scene_root,
                target_num_frames=int(args.target_num_frames),
                render_height=int(args.official_render_height),
                render_cache_root=output_root / "rendered_depth",
            )
            source_frame_count = int(scene_record["source_frame_count"])
        else:
            scene_record = load_scene_record_for_protocol(
                scene_root=scene_root,
                dataset_root=dataset_root,
                args=args,
                scene_name=scene_name,
            )
            source_frame_count = len(frame_ids(scene_record))
            scene_record = subsample_scene_record(
                scene_record=scene_record,
                target_num_frames=int(args.target_num_frames),
                source_prestride=int(args.source_prestride),
            )
        overlap_table = build_scene_overlap_table(
            scene_record=scene_record,
            sample_stride=int(args.overlap_sample_stride),
            depth_rel_tol=float(args.depth_rel_tol),
        )
        candidate_rows, complete_rows = enumerate_scene_candidates(
            scene_name=scene_name,
            scene_record=scene_record,
            overlap_table=overlap_table,
            local_overlap_threshold=float(args.local_overlap_threshold),
            anchor_stride=int(args.anchor_stride),
        )
        all_candidates.extend(candidate_rows)
        all_complete_rows.extend(complete_rows)
        summary = {
            "scene_index": int(scene_index),
            "scene_name": scene_name,
            "scene_root": str(scene_root),
            "source_frame_count": int(source_frame_count),
            "scanned_frame_count": int(len(frame_ids(scene_record))),
            "candidate_rows": int(len(candidate_rows)),
            "complete_anchors": int(len(complete_rows) // len(BAND_SPECS)),
            "elapsed_seconds": float(time.time() - started),
        }
        scene_summaries.append(summary)
        write_json(output_root / "progress.json", {"scenes": scene_summaries})
        print(
            f"[overlap-preflight] scene={scene_name} frames={summary['scanned_frame_count']} "
            f"candidates={summary['candidate_rows']} complete={summary['complete_anchors']} "
            f"elapsed={summary['elapsed_seconds']:.1f}s",
            flush=True,
        )

    selected_rows = select_balanced_anchor_rows(
        all_complete_rows,
        target_anchors=int(args.target_anchors),
        max_anchors_per_scene=int(args.max_anchors_per_scene),
    )
    band_counts = {
        band: int(sum(str(row["overlap_level"]) == band for row in all_candidates))
        for band in BAND_SPECS
    }
    preflight_summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "scannetpp_paired_overlap_sweep_v1",
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "config": vars(args),
        "num_scenes": int(len(scene_roots)),
        "num_candidate_rows": int(len(all_candidates)),
        "candidate_band_counts": band_counts,
        "num_complete_anchors_before_cap": int(len(_rows_by_anchor(all_complete_rows))),
        "num_selected_anchors": int(len(_rows_by_anchor(selected_rows))),
        "num_selected_rows": int(len(selected_rows)),
        "scenes": scene_summaries,
    }
    manifest = {
        "protocol": "scannetpp_paired_overlap_sweep_v1",
        "dataset_root": str(dataset_root),
        "config": vars(args),
        "rows": selected_rows,
    }
    write_csv(output_root / "candidate_rows.csv", all_candidates)
    write_csv(output_root / "complete_anchor_rows.csv", all_complete_rows)
    write_json(output_root / "selection_manifest.json", manifest)
    write_json(output_root / "preflight_summary.json", preflight_summary)

    required_minimum = 1 if bool(args.allow_insufficient_for_smoke) else int(args.min_anchors)
    if selected_rows:
        validation = validate_selection_rows(
            selected_rows,
            min_anchors=required_minimum,
            local_overlap_threshold=float(args.local_overlap_threshold),
        )
    else:
        validation = {"num_anchors": 0, "num_rows": 0, "num_scenes": 0, "levels": list(BAND_SPECS)}
    preflight_summary["validation"] = validation
    write_json(output_root / "preflight_summary.json", preflight_summary)
    if not bool(args.allow_insufficient_for_smoke) and validation["num_anchors"] < int(args.min_anchors):
        raise RuntimeError(
            f"Formal preflight requires at least {int(args.min_anchors)} balanced anchors; "
            f"found {validation['num_anchors']}. Diagnostics retained at {output_root}"
        )
    return preflight_summary


def main() -> None:
    args = parse_args()
    if str(args.mode) == "fixed-preflight":
        summary = run_fixed_group_a_preflight(args)
        print(
            f"[fixed-overlap-preflight] anchors={summary['num_complete_anchors']} "
            f"rows={summary['num_selected_rows']} output={summary['output_root']}",
            flush=True,
        )
    elif str(args.mode) == "reselect":
        report = reselect_from_preflight(
            preflight_root=Path(args.output_root),
            target_anchors=int(args.target_anchors),
            min_anchors=int(args.min_anchors),
            max_anchors_per_scene=int(args.max_anchors_per_scene),
            local_overlap_threshold=float(args.local_overlap_threshold),
        )
        print(
            f"[overlap-reselect] anchors={report['num_anchors']} rows={report['num_rows']} "
            f"scenes={report['num_scenes']} strategy={report['selection_strategy']}",
            flush=True,
        )
    elif str(args.mode) == "materialize":
        if not args.manifest_path:
            raise ValueError("--manifest-path is required for --mode materialize")
        report = materialize_from_manifest(
            manifest_path=Path(args.manifest_path),
            output_root=Path(args.output_root),
            dataset_name=str(args.dataset_output_name),
            overwrite=bool(args.overwrite),
        )
        print(
            f"[overlap-materialize] anchors={report['num_anchors']} tuples={report['num_tuples']} "
            f"output={report['output_root']}",
            flush=True,
        )
    else:
        summary = run_preflight(args)
        print(
            f"[overlap-preflight] selected_anchors={summary['num_selected_anchors']} "
            f"candidate_rows={summary['num_candidate_rows']} output={summary['output_root']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
