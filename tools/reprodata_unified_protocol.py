#!/usr/bin/env python3
"""Build unified protocol JSONs for reproduced rebuttal settings."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    single = sub.add_parser("single-variant", help="Build a protocol from sequence dirs with one variant.")
    single.add_argument("--root", required=True)
    single.add_argument("--output", required=True)
    single.add_argument("--protocol-name", required=True)
    single.add_argument("--variant-name", default="weak_overlap")
    single.add_argument("--default-prefix-size", type=int, default=5)
    single.add_argument("--sample-limit", type=int, default=0)

    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_image_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing color_90 directory: {input_dir}")
    files = [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No images found in {input_dir}")
    return files


def infer_prefix_size(meta: dict[str, Any], image_count: int, default_prefix_size: int) -> int:
    for key in ("views_per_cluster", "views_per_camera", "context_start"):
        value = meta.get(key)
        if isinstance(value, int) and 0 < value < image_count:
            return int(value)
    eval_indices = meta.get("eval_frame_indices")
    if isinstance(eval_indices, list) and 0 < len(eval_indices) < image_count:
        return len(eval_indices)
    return min(max(int(default_prefix_size), 1), image_count)


def frame_payload_from_meta(meta: dict[str, Any], index: int, image_path: Path, prefix_size: int) -> dict[str, Any]:
    frames_meta = meta.get("frames")
    frame_meta = frames_meta[index] if isinstance(frames_meta, list) and index < len(frames_meta) else {}
    role = frame_meta.get("role") if isinstance(frame_meta, dict) else None
    if not role:
        role = "group_a" if index < int(prefix_size) else "group_b"

    frame: dict[str, Any] = {
        "index": int(index),
        "file_name": image_path.name,
        "image_path": str(image_path),
        "role": str(role),
    }
    if isinstance(frame_meta, dict):
        frame_id = frame_meta.get("frame_id")
        if frame_id is not None:
            frame["source_frame_name"] = str(frame_id)
        group = frame_meta.get("group")
        if group is not None:
            frame["group"] = str(group)
    return frame


def build_single_variant_payload(
    seq_dir: Path,
    *,
    variant_name: str,
    default_prefix_size: int,
) -> tuple[dict[str, Any], int, int]:
    input_dir = seq_dir / "color_90"
    meta_path = seq_dir / "tuple_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing tuple_meta.json: {meta_path}")
    meta = load_json(meta_path)
    image_files = collect_image_files(input_dir)
    prefix_size = infer_prefix_size(meta, len(image_files), default_prefix_size)
    frames = [
        frame_payload_from_meta(meta, index, image_path, prefix_size)
        for index, image_path in enumerate(image_files)
    ]
    payload = {
        "variant": str(variant_name),
        "input_dir": str(input_dir),
        "sequence_dir": str(seq_dir),
        "tuple_meta_path": str(meta_path),
        "tuple_sample_id": str(meta.get("sample_id") or meta.get("tuple_name") or seq_dir.name),
        "frames": frames,
    }
    return payload, int(prefix_size), int(len(image_files) - prefix_size)


def build_single_variant_protocol(
    root: str | Path,
    *,
    protocol_name: str,
    variant_name: str = "weak_overlap",
    default_prefix_size: int = 5,
    sample_limit: int = 0,
) -> dict[str, Any]:
    source_root = Path(root).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Root not found: {source_root}")

    samples: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for seq_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        if not (seq_dir / "color_90").is_dir():
            continue
        try:
            variant_payload, prefix_size, context_size = build_single_variant_payload(
                seq_dir,
                variant_name=variant_name,
                default_prefix_size=default_prefix_size,
            )
        except Exception as exc:
            skipped.append({"sequence_dir": str(seq_dir), "error": f"{type(exc).__name__}:{exc}"})
            continue
        samples.append(
            {
                "sample_id": seq_dir.name,
                "prefix_size": int(prefix_size),
                "context_size": int(context_size),
                "variants": {str(variant_name): variant_payload},
            }
        )
        if sample_limit > 0 and len(samples) >= sample_limit:
            break

    prefix_size = int(samples[0]["prefix_size"]) if samples else int(default_prefix_size)
    context_size = int(samples[0]["context_size"]) if samples else 0
    return {
        "protocol_name": str(protocol_name),
        "source_root": str(source_root),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "variants": [str(variant_name)],
        "prefix_size": prefix_size,
        "context_size": context_size,
        "num_samples": len(samples),
        "samples": samples,
        "skipped_samples": skipped,
    }


def main() -> None:
    args = parse_args()
    if args.cmd == "single-variant":
        protocol = build_single_variant_protocol(
            args.root,
            protocol_name=str(args.protocol_name),
            variant_name=str(args.variant_name),
            default_prefix_size=int(args.default_prefix_size),
            sample_limit=int(args.sample_limit),
        )
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[unified-protocol] wrote {output}")
        print(f"[unified-protocol] samples={protocol['num_samples']} skipped={len(protocol['skipped_samples'])}")
        return
    raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
