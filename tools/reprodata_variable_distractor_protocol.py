#!/usr/bin/env python3
"""Build and validate a five-level fixed-prefix distractor protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


VARIANTS = ("noise0", "noise1", "noise2", "noise3", "noise4")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PREFIX_SIZE = 6
TOTAL_VIEWS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-limit", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_images(sequence_dir: Path) -> list[Path]:
    color_dir = sequence_dir / "color_90"
    images = sorted(path for path in color_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file())
    if len(images) != TOTAL_VIEWS:
        raise RuntimeError(f"Expected {TOTAL_VIEWS} images under {color_dir}, got {len(images)}")
    return images


def parse_pose_rows(sequence_dir: Path) -> list[str]:
    path = sequence_dir / "pose_90.txt"
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != TOTAL_VIEWS:
        raise RuntimeError(f"Expected {TOTAL_VIEWS} pose rows at {path}, got {len(rows)}")
    return rows


def split_variant_name(name: str, variant: str) -> str:
    suffix = f"__{variant}"
    if not name.endswith(suffix):
        raise ValueError(f"Directory {name!r} does not end with {suffix!r}")
    return name[: -len(suffix)]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_variant_payload(sequence_dir: Path, variant: str) -> dict[str, Any]:
    images = collect_images(sequence_dir)
    parse_pose_rows(sequence_dir)
    meta_path = sequence_dir / "tuple_meta.json"
    meta = load_json(meta_path)
    distractor_count = int(variant.removeprefix("noise"))
    expected_labels = ["clean"] * (TOTAL_VIEWS - distractor_count) + ["distractor"] * distractor_count
    if meta.get("eval_frame_indices") != list(range(PREFIX_SIZE)):
        raise RuntimeError(f"Invalid eval_frame_indices for {sequence_dir}")
    if meta.get("source_labels") != expected_labels:
        raise RuntimeError(
            f"Invalid source_labels for {sequence_dir}: expected {expected_labels}, got {meta.get('source_labels')}"
        )
    frames = []
    for index, image_path in enumerate(images):
        source_label = expected_labels[index]
        if index < PREFIX_SIZE:
            role = "prefix"
        elif source_label == "distractor":
            role = "distractor_context"
        else:
            role = "clean_context"
        frames.append(
            {
                "index": index,
                "file_name": image_path.name,
                "image_path": str(image_path),
                "role": role,
                "source_label": source_label,
            }
        )
    return {
        "variant": variant,
        "distractor_count": distractor_count,
        "context_distractor_ratio": float(distractor_count / (TOTAL_VIEWS - PREFIX_SIZE)),
        "input_dir": str(sequence_dir / "color_90"),
        "sequence_dir": str(sequence_dir),
        "tuple_meta_path": str(meta_path),
        "frames": frames,
        "prefix_hashes": [file_sha256(path) for path in images[:PREFIX_SIZE]],
    }


def discover_samples(root: Path) -> dict[str, dict[str, Path]]:
    grouped: dict[str, dict[str, Path]] = {}
    for variant in VARIANTS:
        variant_root = root / variant
        if not variant_root.is_dir():
            raise FileNotFoundError(f"Missing variant root: {variant_root}")
        for sequence_dir in sorted(path for path in variant_root.iterdir() if path.is_dir()):
            sample_id = split_variant_name(sequence_dir.name, variant)
            grouped.setdefault(sample_id, {})[variant] = sequence_dir
    return grouped


def validate_fixed_prefix(sample_id: str, variants: dict[str, dict[str, Any]]) -> None:
    reference = variants[VARIANTS[0]]["prefix_hashes"]
    for variant in VARIANTS[1:]:
        if variants[variant]["prefix_hashes"] != reference:
            raise RuntimeError(f"fixed prefix mismatch for {sample_id} at {variant}")


def build_protocol(root: str | Path, *, sample_limit: int = 0) -> dict[str, Any]:
    source_root = Path(root).expanduser().resolve()
    grouped = discover_samples(source_root)
    samples: list[dict[str, Any]] = []
    for sample_id, sequence_dirs in sorted(grouped.items()):
        missing = [variant for variant in VARIANTS if variant not in sequence_dirs]
        if missing:
            raise RuntimeError(f"Sample {sample_id} missing variants: {missing}")
        variant_payloads = {
            variant: build_variant_payload(sequence_dirs[variant], variant)
            for variant in VARIANTS
        }
        validate_fixed_prefix(sample_id, variant_payloads)
        samples.append(
            {
                "sample_id": sample_id,
                "prefix_size": PREFIX_SIZE,
                "context_size": TOTAL_VIEWS - PREFIX_SIZE,
                "variants": variant_payloads,
            }
        )
        if sample_limit > 0 and len(samples) >= sample_limit:
            break
    return {
        "protocol_name": "waymo_fixed10_distractor_ratio_v1",
        "source_root": str(source_root),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "variants": list(VARIANTS),
        "prefix_size": PREFIX_SIZE,
        "context_size": TOTAL_VIEWS - PREFIX_SIZE,
        "num_samples": len(samples),
        "samples": samples,
        "validation": {
            "prefix_hashes_match": True,
            "num_variant_inputs": len(samples) * len(VARIANTS),
        },
    }


def main() -> None:
    args = parse_args()
    protocol = build_protocol(args.root, sample_limit=int(args.sample_limit))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[variable-distractor-protocol] wrote {output}")
    print(
        f"[variable-distractor-protocol] samples={protocol['num_samples']} "
        f"variants={protocol['validation']['num_variant_inputs']}"
    )


if __name__ == "__main__":
    main()
