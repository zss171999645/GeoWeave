"""Auto-generate Pi3-style seq-id-map JSON from an EVC-format dataset directory.

For each sequence directory under ``dataset_root``, the script discovers available
frame indices from camera files (``intri.yml``) or image directories, then sub-samples
frames at a regular keyframe interval (``--kf-step``).
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


def discover_frame_indices_from_intri(seq_dir: Path) -> Optional[List[int]]:
    intri = seq_dir / "intri.yml"
    if not intri.is_file():
        return None
    indices: List[int] = []
    with open(intri, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = re.match(r"^(\d+):$", line.strip())
            if match:
                indices.append(int(match.group(1)))
    return sorted(indices) if indices else None


def discover_frame_indices_from_images(seq_dir: Path) -> Optional[List[int]]:
    img_dir = seq_dir / "images"
    if not img_dir.is_dir():
        indices: List[int] = []
        for path in seq_dir.iterdir():
            if not path.is_file():
                continue
            match = re.match(r"^frame-(\d+)\.color\.(png|jpg|jpeg)$", path.name, flags=re.IGNORECASE)
            if match:
                indices.append(int(match.group(1)))
        return sorted(indices) if indices else None
    indices: List[int] = []
    for path in img_dir.iterdir():
        if not path.is_file():
            continue
        match = re.match(r"^img(\d+)\.(png|jpg|jpeg)$", path.name, flags=re.IGNORECASE)
        if match:
            indices.append(int(match.group(1)))
    if indices:
        return sorted(indices)
    fallback: List[int] = []
    for path in img_dir.iterdir():
        stem = path.stem
        if stem.isdigit():
            fallback.append(int(stem))
    return sorted(fallback) if fallback else None


def discover_frames(seq_dir: Path) -> List[int]:
    frames = discover_frame_indices_from_intri(seq_dir)
    if frames:
        return frames
    frames = discover_frame_indices_from_images(seq_dir)
    if frames:
        return frames
    return []


def subsample_frames(frames: List[int], kf_step: int, max_views: int = 0) -> List[int]:
    if kf_step <= 0:
        selected = frames
    else:
        selected = frames[::kf_step]
    if not selected:
        selected = frames[:1]
    if max_views > 0 and len(selected) > max_views:
        selected = selected[:max_views]
    return selected


def discover_sequences_7scenes(dataset_root: Path) -> Dict[str, Path]:
    seqs: Dict[str, Path] = {}
    for scene_dir in sorted(dataset_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        has_subseq = False
        for sub in sorted(scene_dir.iterdir()):
            if not sub.is_dir():
                continue
            if sub.name.startswith("seq-"):
                seqs[f"{scene_dir.name}_{sub.name}"] = sub
                has_subseq = True
        if not has_subseq and ((scene_dir / "intri.yml").is_file() or (scene_dir / "images").is_dir()):
            seqs[scene_dir.name] = scene_dir
    return seqs


def discover_sequences_flat(dataset_root: Path) -> Dict[str, Path]:
    return {path.name: path for path in sorted(dataset_root.iterdir()) if path.is_dir()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=str, required=True, help="Root dir of the dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="auto",
        choices=["auto", "7scenes", "nrgbd", "flat"],
        help="Dataset type. 'auto' tries 7scenes (scene/seq-NN) then flat.",
    )
    parser.add_argument("--kf-step", type=int, default=5, help="Keyframe step for sub-sampling frames.")
    parser.add_argument("--max-views", type=int, default=0, help="Max views per sequence (0=no limit).")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path.")
    parser.add_argument("--min-frames", type=int, default=2, help="Skip sequences with fewer frames than this.")
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    if not root.is_dir():
        print(f"ERROR: dataset root not found: {root}", file=sys.stderr)
        raise SystemExit(1)

    if args.dataset == "7scenes":
        seq_map = discover_sequences_7scenes(root)
    elif args.dataset in ("nrgbd", "flat"):
        seq_map = discover_sequences_flat(root)
    else:
        seq_map = discover_sequences_7scenes(root)
        if not seq_map:
            seq_map = discover_sequences_flat(root)

    if not seq_map:
        print(f"ERROR: no sequences found under {root}", file=sys.stderr)
        raise SystemExit(1)

    result = {}
    for key, seq_dir in sorted(seq_map.items()):
        frames = discover_frames(seq_dir)
        if len(frames) < args.min_frames:
            print(f"  SKIP {key}: only {len(frames)} frames (< {args.min_frames})")
            continue
        selected = subsample_frames(frames, args.kf_step, args.max_views)
        result[key] = selected
        print(f"  {key}: {len(frames)} frames -> {len(selected)} selected (kf_step={args.kf_step})")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated seq-id-map with {len(result)} sequences -> {output_path}")


if __name__ == "__main__":
    main()
