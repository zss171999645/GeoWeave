#!/usr/bin/env python3
"""Derive variable-count distractor tuples from paired clean/noise benchmarks.

Input layout is the PI3 official relpose tuple layout produced by existing
6-clean+4-noise builders:

    <input>/<dataset>/<base>__clean/color_90/frame_0000.jpg ...
    <input>/<dataset>/<base>__noise/color_90/frame_0000.jpg ...

The original noise tuple is assumed to keep the first eval views clean and to
replace the tail with distractor frames.  This script creates fixed-length
variants by replacing only the last k frames with the original distractors:

    noise0 = clean 0..9
    noise1 = clean 0..8 + noise 9
    noise2 = clean 0..7 + noise 8..9
    noise3 = clean 0..6 + noise 7..9
    noise4 = clean 0..5 + noise 6..9

For a more controlled "insert distractors only" protocol, pass
``--clean-fill-mode repeat_last_eval``.  Then the first eval views remain
clean, non-distractor tail slots repeat the last eval clean frame, and only
the last k slots are distractors.

For the cleanest extra-token stress, pass
``--output-total-mode eval_plus_distractors``.  Then noise0 has only the eval
clean frames, and noiseK appends K distractor frames after them.

Evaluation should still use --eval-frame-indices 0,1,2,3,4,5.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


IMAGE_EXTS = (".jpg", ".jpeg", ".png")
PAIR_SUFFIXES = (
    ("__clean", "__noise"),
    ("__clean_tail", "__plausible_noise_tail"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Existing paired clean/noise benchmark root.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--datasets", default="", help="Comma-separated dataset subdirs. Empty auto-discovers.")
    parser.add_argument("--variants", default="0,1,2,3,4", help="Comma-separated distractor counts.")
    parser.add_argument("--total-views", type=int, default=10)
    parser.add_argument("--eval-views", type=int, default=6)
    parser.add_argument(
        "--output-total-mode",
        choices=("fixed", "eval_plus_distractors"),
        default="fixed",
        help="fixed keeps --total-views outputs; eval_plus_distractors writes eval_views + distractor_count outputs.",
    )
    parser.add_argument(
        "--clean-fill-mode",
        choices=("tail_clean", "repeat_last_eval"),
        default="tail_clean",
        help=(
            "tail_clean preserves clean tail views before replacing the last k slots; "
            "repeat_last_eval repeats frame eval_views-1 for non-distractor tail slots."
        ),
    )
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


def parse_csv_ints(value: str) -> List[int]:
    items = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one integer")
    return items


def parse_datasets(input_root: Path, value: str) -> List[str]:
    if value.strip():
        return [item.strip() for item in value.split(",") if item.strip()]
    candidates = [
        path.name
        for path in sorted(input_root.iterdir())
        if path.is_dir() and any(split_clean_name(child.name) is not None for child in path.iterdir() if child.is_dir())
    ]
    if candidates:
        return candidates
    if any(split_clean_name(child.name) is not None for child in input_root.iterdir() if child.is_dir()):
        return ["."]
    raise FileNotFoundError(f"No paired clean/noise tuples found under {input_root}")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_pose_rows(path: Path) -> List[str]:
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No pose rows in {path}")
    return rows


def find_frame_path(tuple_root: Path, index: int) -> Path:
    color_dir = tuple_root / "color_90"
    for ext in IMAGE_EXTS:
        path = color_dir / f"frame_{index:04d}{ext}"
        # The source benchmark often stores bucket images as symlinks.  Use
        # lexists so construction does not block resolving remote symlink
        # targets just to re-symlink the tuple.
        if os.path.lexists(path):
            return path
    candidates = sorted(color_dir.glob(f"frame_{index:04d}.*"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Missing frame_{index:04d} under {color_dir}")


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(str(src), str(dst))
    except OSError:
        shutil.copy2(src, dst)


def materialize_variant(
    output_tuple: Path,
    clean_tuple: Path,
    noise_tuple: Path,
    distractor_count: int,
    source_total_views: int,
    eval_views: int,
    output_total_mode: str,
    clean_fill_mode: str,
    copy_images: bool,
    metadata: Dict[str, Any],
) -> None:
    clean_poses = read_pose_rows(clean_tuple / "pose_90.txt")
    noise_poses = read_pose_rows(noise_tuple / "pose_90.txt")
    if len(clean_poses) < source_total_views or len(noise_poses) < source_total_views:
        raise ValueError(f"Tuple length shorter than total_views={source_total_views}: {clean_tuple}, {noise_tuple}")
    if distractor_count < 0 or distractor_count > source_total_views - eval_views:
        raise ValueError(f"distractor_count={distractor_count} incompatible with total/eval views")
    if output_total_mode == "fixed":
        output_views = int(source_total_views)
    elif output_total_mode == "eval_plus_distractors":
        output_views = int(eval_views) + int(distractor_count)
    else:
        raise ValueError(f"Unsupported output_total_mode={output_total_mode!r}")

    output_tuple.mkdir(parents=True, exist_ok=False)
    color_dir = output_tuple / "color_90"
    color_dir.mkdir()
    pose_rows: List[str] = []
    source_labels: List[str] = []
    switch_index = source_total_views - distractor_count
    for index in range(output_views):
        if output_total_mode == "eval_plus_distractors":
            use_noise = index >= eval_views
        else:
            use_noise = index >= switch_index
        if use_noise:
            source_tuple = noise_tuple
            source_index = index
            source_label = "distractor"
        else:
            source_tuple = clean_tuple
            source_index = index
            source_label = "clean"
            if clean_fill_mode == "tail_clean":
                pass
            elif clean_fill_mode == "repeat_last_eval":
                if index >= eval_views:
                    source_index = eval_views - 1
                    source_label = "clean_repeat_last_eval"
            else:
                raise ValueError(f"Unsupported clean_fill_mode={clean_fill_mode!r}")
        source_path = find_frame_path(source_tuple, source_index)
        suffix = source_path.suffix.lower() or ".jpg"
        link_or_copy(source_path, color_dir / f"frame_{index:04d}{suffix}", copy_images=copy_images)
        pose_rows.append(noise_poses[source_index] if use_noise else clean_poses[source_index])
        source_labels.append(source_label)

    (output_tuple / "pose_90.txt").write_text("\n".join(pose_rows) + "\n", encoding="utf-8")
    payload = dict(metadata)
    payload.update(
        {
            "protocol": "variable_distractor_from_pairs_v1",
            "tuple_kind": f"noise{distractor_count}",
            "distractor_count": int(distractor_count),
            "total_views": int(output_views),
            "source_total_views": int(source_total_views),
            "output_total_mode": output_total_mode,
            "eval_views": int(eval_views),
            "clean_fill_mode": clean_fill_mode,
            "eval_frame_indices": list(range(int(eval_views))),
            "source_labels": source_labels,
            "source_clean_tuple": str(clean_tuple),
            "source_noise_tuple": str(noise_tuple),
        }
    )
    (output_tuple / "tuple_meta.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def split_clean_name(name: str) -> tuple[str, str] | None:
    for clean_suffix, noise_suffix in PAIR_SUFFIXES:
        if name.endswith(clean_suffix):
            return name[: -len(clean_suffix)], noise_suffix
    return None


def paired_clean_noise(dataset_root: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for clean_tuple in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        parsed = split_clean_name(clean_tuple.name)
        if parsed is None:
            continue
        base, noise_suffix = parsed
        noise_tuple = clean_tuple.with_name(base + noise_suffix)
        if noise_tuple.is_dir():
            pairs.append((clean_tuple, noise_tuple))
    return pairs


def build_dataset(
    input_root: Path,
    output_root: Path,
    dataset: str,
    variants: Sequence[int],
    source_total_views: int,
    eval_views: int,
    output_total_mode: str,
    clean_fill_mode: str,
    copy_images: bool,
) -> Dict[str, Any]:
    dataset_input = input_root if dataset == "." else input_root / dataset
    if not dataset_input.is_dir():
        raise FileNotFoundError(f"Missing dataset input root: {dataset_input}")
    pairs = paired_clean_noise(dataset_input)
    if not pairs:
        raise FileNotFoundError(f"No clean/noise pairs found under {dataset_input}")

    dataset_name = dataset_input.name if dataset == "." else dataset
    dataset_output = output_root / dataset_name
    rows: List[Dict[str, Any]] = []
    counts = {f"noise{int(variant)}": 0 for variant in variants}
    for clean_tuple, noise_tuple in pairs:
        clean_meta = read_json(clean_tuple / "tuple_meta.json")
        noise_meta = read_json(noise_tuple / "tuple_meta.json")
        parsed = split_clean_name(clean_tuple.name)
        if parsed is None:
            raise ValueError(f"Unsupported clean tuple suffix: {clean_tuple}")
        base, _noise_suffix = parsed
        for variant in variants:
            variant_root = dataset_output / f"noise{int(variant)}"
            out_name = f"{base}__noise{int(variant)}"
            metadata = dict(clean_meta)
            metadata.update(
                {
                    "dataset": dataset_name,
                    "source_base": base,
                    "source_clean_meta": clean_meta,
                    "source_noise_meta": noise_meta,
                }
            )
            materialize_variant(
                output_tuple=variant_root / out_name,
                clean_tuple=clean_tuple,
                noise_tuple=noise_tuple,
                distractor_count=int(variant),
                source_total_views=int(source_total_views),
                eval_views=int(eval_views),
                output_total_mode=output_total_mode,
                clean_fill_mode=clean_fill_mode,
                copy_images=copy_images,
                metadata=metadata,
            )
            counts[f"noise{int(variant)}"] += 1
        rows.append({"base": base, "clean_tuple": str(clean_tuple), "noise_tuple": str(noise_tuple)})

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "variable_distractor_from_pairs_v1",
        "dataset": dataset_name,
        "input_root": str(dataset_input),
        "output_root": str(dataset_output),
        "total_views": int(source_total_views),
        "source_total_views": int(source_total_views),
        "output_total_mode": output_total_mode,
        "eval_views": int(eval_views),
        "clean_fill_mode": clean_fill_mode,
        "variants": [int(item) for item in variants],
        "counts": counts,
        "num_pairs": len(pairs),
        "pairs": rows,
    }
    dataset_output.mkdir(parents=True, exist_ok=True)
    (dataset_output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    variants = parse_csv_ints(args.variants)
    if int(args.eval_views) >= int(args.total_views):
        raise ValueError("--eval-views must be smaller than --total-views")
    max_distractors = int(args.total_views) - int(args.eval_views)
    if any(variant < 0 or variant > max_distractors for variant in variants):
        raise ValueError(f"variants must be in [0, {max_distractors}], got {variants}")

    output_root.mkdir(parents=True, exist_ok=True)
    datasets = parse_datasets(input_root, args.datasets)
    summaries = [
        build_dataset(
            input_root=input_root,
            output_root=output_root,
            dataset=dataset,
            variants=variants,
            source_total_views=int(args.total_views),
            eval_views=int(args.eval_views),
            output_total_mode=str(args.output_total_mode),
            clean_fill_mode=str(args.clean_fill_mode),
            copy_images=bool(args.copy_images),
        )
        for dataset in datasets
    ]
    payload = {
        "output_root": str(output_root),
        "datasets": [summary["dataset"] for summary in summaries],
        "summaries": summaries,
    }
    (output_root / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
