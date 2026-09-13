#!/usr/bin/env python3
"""Build a clean/noise Waymo protocol from reproduced rebuttal data.

The produced JSON matches the small protocol shape used by the rebuttal helper
scripts: each sample owns two variants with an `input_dir` and per-frame roles.
It references the existing image folders in place and does not copy or modify
the reproduced data.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


VARIANTS = ("clean_tail", "plausible_noise_tail")
PREFIX_ROLE = "prefix"
CONTEXT_ROLES = {
    "clean_tail": "clean_context",
    "plausible_noise_tail": "distractor_context",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Root containing paired clean/noise sequence directories.")
    parser.add_argument("--output", required=True, help="Destination protocol JSON.")
    parser.add_argument("--sample-limit", type=int, default=0)
    return parser.parse_args()


def split_variant_suffix(name: str) -> tuple[str, str] | None:
    for variant in VARIANTS:
        suffix = f"__{variant}"
        if name.endswith(suffix):
            return name[: -len(suffix)], variant
    return None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_image_files(input_dir: Path) -> list[Path]:
    files = [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files)


def frame_name_from_meta(meta: dict[str, Any], index: int, prefix_size: int) -> str | None:
    if index < prefix_size:
        names = meta.get("prefix_frame_names")
        if isinstance(names, list) and index < len(names):
            return str(names[index])
    names = meta.get("tail_frame_names")
    tail_index = index - prefix_size
    if isinstance(names, list) and 0 <= tail_index < len(names):
        return str(names[tail_index])
    return None


def infer_prefix_size(meta: dict[str, Any], image_count: int) -> int:
    eval_indices = meta.get("eval_frame_indices")
    if isinstance(eval_indices, list) and eval_indices:
        return len(eval_indices)
    prefix_names = meta.get("prefix_frame_names")
    if isinstance(prefix_names, list) and prefix_names:
        return len(prefix_names)
    return min(6, image_count)


def build_variant_payload(seq_dir: Path, variant: str) -> tuple[dict[str, Any], int, int]:
    input_dir = seq_dir / "color_90"
    meta_path = seq_dir / "tuple_meta.json"
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing color_90 directory: {input_dir}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing tuple_meta.json: {meta_path}")

    meta = load_json(meta_path)
    image_files = collect_image_files(input_dir)
    if not image_files:
        raise FileNotFoundError(f"No images found in {input_dir}")
    prefix_size = infer_prefix_size(meta, len(image_files))
    frames: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_files):
        role = PREFIX_ROLE if index < prefix_size else CONTEXT_ROLES[variant]
        frame: dict[str, Any] = {
            "index": index,
            "file_name": image_path.name,
            "image_path": str(image_path),
            "role": role,
        }
        frame_name = frame_name_from_meta(meta, index, prefix_size)
        if frame_name is not None:
            frame["source_frame_name"] = frame_name
        frames.append(frame)

    payload = {
        "variant": variant,
        "input_dir": str(input_dir),
        "sequence_dir": str(seq_dir),
        "tuple_meta_path": str(meta_path),
        "tuple_sample_id": str(meta.get("sample_id", seq_dir.name)),
        "frames": frames,
    }
    return payload, prefix_size, len(image_files) - prefix_size


def discover_pairs(root: Path) -> tuple[dict[str, dict[str, Path]], list[dict[str, Any]]]:
    grouped: dict[str, dict[str, Path]] = {}
    ignored: list[dict[str, Any]] = []
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        parsed = split_variant_suffix(child.name)
        if parsed is None:
            ignored.append({"directory": str(child), "reason": "name_without_known_variant_suffix"})
            continue
        sample_id, variant = parsed
        grouped.setdefault(sample_id, {})[variant] = child
    return grouped, ignored


def build_protocol(root: str | Path, *, sample_limit: int = 0) -> dict[str, Any]:
    source_root = Path(root).expanduser().resolve()
    grouped, ignored_dirs = discover_pairs(source_root)
    samples: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for sample_id, variants in sorted(grouped.items()):
        missing = [variant for variant in VARIANTS if variant not in variants]
        if missing:
            skipped.append({"sample_id": sample_id, "missing_variants": missing})
            continue

        variant_payloads: dict[str, Any] = {}
        prefix_sizes: list[int] = []
        context_sizes: list[int] = []
        errors: list[str] = []
        for variant in VARIANTS:
            try:
                payload, prefix_size, context_size = build_variant_payload(variants[variant], variant)
                variant_payloads[variant] = payload
                prefix_sizes.append(prefix_size)
                context_sizes.append(context_size)
            except Exception as exc:
                errors.append(f"{variant}:{type(exc).__name__}:{exc}")
        if errors:
            skipped.append({"sample_id": sample_id, "errors": errors})
            continue
        if len(set(prefix_sizes)) != 1 or len(set(context_sizes)) != 1:
            skipped.append(
                {
                    "sample_id": sample_id,
                    "errors": [
                        f"inconsistent_prefix_or_context_sizes:prefix={prefix_sizes}:context={context_sizes}"
                    ],
                }
            )
            continue

        samples.append(
            {
                "sample_id": sample_id,
                "prefix_size": int(prefix_sizes[0]),
                "context_size": int(context_sizes[0]),
                "variants": variant_payloads,
            }
        )
        if sample_limit > 0 and len(samples) >= sample_limit:
            break

    prefix_size = samples[0]["prefix_size"] if samples else 6
    context_size = samples[0]["context_size"] if samples else 4
    return {
        "protocol_name": "waymo_simple_plausible_wrong_context_reprodata",
        "source_root": str(source_root),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "variants": list(VARIANTS),
        "prefix_size": int(prefix_size),
        "context_size": int(context_size),
        "num_samples": len(samples),
        "samples": samples,
        "skipped_samples": skipped,
        "ignored_directories": ignored_dirs,
    }


def main() -> None:
    args = parse_args()
    protocol = build_protocol(args.root, sample_limit=int(args.sample_limit))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[waymo-protocol] wrote {output}")
    print(f"[waymo-protocol] samples={protocol['num_samples']} skipped={len(protocol['skipped_samples'])}")


if __name__ == "__main__":
    main()
