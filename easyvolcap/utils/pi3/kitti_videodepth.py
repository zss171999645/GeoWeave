from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def list_regular_png_files(dir_path: Path) -> List[Path]:
    return sorted([path for path in dir_path.iterdir() if path.is_file() and path.suffix.lower() == ".png" and not path.name.startswith(".")])


def has_kitti_monst3r_gathered_layout(data_root: Path) -> bool:
    return (data_root / "image_gathered").is_dir() and (data_root / "groundtruth_depth_gathered").is_dir()


def has_kitti_official_flat_layout(data_root: Path) -> bool:
    return (data_root / "image").is_dir() and (data_root / "groundtruth_depth").is_dir()


def explain_kitti_videodepth_layout_error(data_root: Path) -> str:
    return (
        f"KITTI PI3 videodepth expects the MonST3R/PI3 gathered layout under {data_root}: "
        "image_gathered/<drive>_02/*.png and groundtruth_depth_gathered/<drive>_02/*.png. "
        "The flat image/groundtruth_depth validation root is not the PI3 videodepth protocol. "
        "Prepare the gathered dataset from KITTI raw + data_depth_annotated with "
        "aidi/scripts/baselines/prepare_pi3_kitti_videodepth_dataset.py."
    )


def _resolve_annotated_sequence_roots(annotated_root: Path, camera: str) -> List[Tuple[str, Path]]:
    candidate_roots = [annotated_root / "val", annotated_root]
    sequence_roots: List[Tuple[str, Path]] = []
    seen = set()
    for root in candidate_roots:
        if not root.is_dir():
            continue
        for seq_dir in sorted([path for path in root.iterdir() if path.is_dir()]):
            gt_dir = seq_dir / "proj_depth" / "groundtruth" / f"image_{camera}"
            if gt_dir.is_dir() and seq_dir.name not in seen:
                sequence_roots.append((seq_dir.name, gt_dir))
                seen.add(seq_dir.name)
    if not sequence_roots:
        raise FileNotFoundError(
            f"Cannot find KITTI annotated sequences under {annotated_root}: expected val/<drive>/proj_depth/groundtruth/image_{camera}"
        )
    return sequence_roots


def _resolve_raw_image_dir(raw_root: Path, drive: str, camera: str) -> Path:
    drive_date = drive.split("_drive_", 1)[0]
    candidates = [
        raw_root / drive_date / drive / f"image_{camera}" / "data",
        raw_root / drive / f"image_{camera}" / "data",
        raw_root / drive / f"image_{camera}",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Cannot find KITTI raw RGB frames for {drive} under {raw_root}: expected {drive_date}/{drive}/image_{camera}/data"
    )


def prepare_kitti_monst3r_gathered_layout(
    raw_root: Path,
    annotated_root: Path,
    output_root: Path,
    camera: str = "02",
    max_frames_per_seq: int = 110,
) -> Dict[str, object]:
    raw_root = Path(raw_root).expanduser().resolve()
    annotated_root = Path(annotated_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()

    image_gathered_root = output_root / "image_gathered"
    gt_gathered_root = output_root / "groundtruth_depth_gathered"
    image_gathered_root.mkdir(parents=True, exist_ok=True)
    gt_gathered_root.mkdir(parents=True, exist_ok=True)

    per_sequence_frames: Dict[str, int] = {}
    for drive, gt_dir in _resolve_annotated_sequence_roots(annotated_root, camera):
        raw_image_dir = _resolve_raw_image_dir(raw_root, drive, camera)
        seq_name = f"{drive}_{camera}"
        seq_image_dir = image_gathered_root / seq_name
        seq_gt_dir = gt_gathered_root / seq_name
        seq_image_dir.mkdir(parents=True, exist_ok=True)
        seq_gt_dir.mkdir(parents=True, exist_ok=True)

        gt_files = list_regular_png_files(gt_dir)
        if max_frames_per_seq > 0:
            gt_files = gt_files[:max_frames_per_seq]
        if not gt_files:
            raise FileNotFoundError(f"No KITTI annotated depth PNGs found under {gt_dir}")

        copied = 0
        for gt_path in gt_files:
            image_path = raw_image_dir / gt_path.name
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing KITTI raw RGB frame {image_path} for annotated depth {gt_path}")
            shutil.copy2(image_path, seq_image_dir / image_path.name)
            shutil.copy2(gt_path, seq_gt_dir / gt_path.name)
            copied += 1
        per_sequence_frames[seq_name] = copied

    return {
        "raw_root": str(raw_root),
        "annotated_root": str(annotated_root),
        "output_root": str(output_root),
        "camera": str(camera),
        "max_frames_per_seq": int(max_frames_per_seq),
        "num_sequences": len(per_sequence_frames),
        "num_frames": int(sum(per_sequence_frames.values())),
        "sequence_frames": per_sequence_frames,
    }
