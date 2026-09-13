from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple


TUM_MAX_ASSOCIATION_DELTA = 0.02
POSE_TARGET_NUM_FRAMES = 90
POSE_SAMPLE_STRIDE = 3


def read_file_list(filename: Path) -> Dict[float, List[str]]:
    with open(filename, "r", encoding="utf-8") as file:
        data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    parsed = [[value.strip() for value in line.split(" ") if value.strip() != ""] for line in lines if line and line[0] != "#"]
    return dict((float(parts[0]), parts[1:]) for parts in parsed if len(parts) > 1)


def associate(
    first_list: Dict[float, List[str]],
    second_list: Dict[float, List[str]],
    offset: float = 0.0,
    max_difference: float = TUM_MAX_ASSOCIATION_DELTA,
) -> List[Tuple[float, float]]:
    first_keys = set(first_list.keys())
    second_keys = set(second_list.keys())
    potential_matches = [
        (abs(a - (b + offset)), a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference
    ]
    potential_matches.sort()
    matches = []
    for _, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    matches.sort()
    return matches


def _ensure_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")


def _ensure_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing required directory: {path}")


def _write_pose_lines(output_path: Path, rows: Sequence[Sequence[object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{' '.join(map(str, row))}\n")


def read_pose_txt_flat(pose_path: Path) -> List[float]:
    rows: List[float] = []
    with open(pose_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [item for item in line.strip().split() if item]
            if not parts:
                continue
            rows.extend(float(item) for item in parts)
    if len(rows) != 16:
        raise ValueError(f"Expected 16 values in ScanNet pose file, got {len(rows)} from {pose_path}")
    return rows


def prepare_tum_sequence(
    source_seq: Path,
    output_seq: Path,
    target_num_frames: int = POSE_TARGET_NUM_FRAMES,
    stride: int = POSE_SAMPLE_STRIDE,
    max_difference: float = TUM_MAX_ASSOCIATION_DELTA,
) -> Dict[str, object]:
    source_seq = Path(source_seq).expanduser().resolve()
    output_seq = Path(output_seq).expanduser().resolve()
    rgb_txt = source_seq / "rgb.txt"
    groundtruth_txt = source_seq / "groundtruth.txt"
    _ensure_file(rgb_txt)
    _ensure_file(groundtruth_txt)

    first_list = read_file_list(rgb_txt)
    second_list = read_file_list(groundtruth_txt)
    matches = associate(first_list, second_list, 0.0, max_difference)

    frames: List[Path] = []
    gt_rows: List[List[object]] = []
    for rgb_stamp, gt_stamp in matches:
        frame = source_seq / first_list[rgb_stamp][0]
        _ensure_file(frame)
        frames.append(frame)
        gt_rows.append([gt_stamp] + second_list[gt_stamp])

    selected_frames = frames[::stride][:target_num_frames]
    selected_gt_rows = gt_rows[::stride][:target_num_frames]

    rgb_90_dir = output_seq / "rgb_90"
    rgb_90_dir.mkdir(parents=True, exist_ok=True)
    for frame in selected_frames:
        shutil.copy(frame, rgb_90_dir / frame.name)
    _write_pose_lines(output_seq / "groundtruth_90.txt", selected_gt_rows)

    return {
        "sequence": source_seq.name,
        "source_seq": str(source_seq),
        "output_seq": str(output_seq),
        "num_source_frames": len(frames),
        "num_selected_frames": len(selected_frames),
    }


def _sorted_numeric_paths(directory: Path, pattern: str) -> List[Path]:
    return sorted(directory.glob(pattern), key=lambda path: int(path.stem))


def prepare_scannet_sequence(
    source_seq: Path,
    output_seq: Path,
    target_num_frames: int = POSE_TARGET_NUM_FRAMES,
    stride: int = POSE_SAMPLE_STRIDE,
) -> Dict[str, object]:
    source_seq = Path(source_seq).expanduser().resolve()
    output_seq = Path(output_seq).expanduser().resolve()
    color_dir = source_seq / "color"
    depth_dir = source_seq / "depth"
    pose_dir = source_seq / "pose"
    _ensure_dir(color_dir)
    _ensure_dir(depth_dir)
    _ensure_dir(pose_dir)

    img_paths = _sorted_numeric_paths(color_dir, "*.jpg")
    depth_paths = _sorted_numeric_paths(depth_dir, "*.png")
    pose_paths = _sorted_numeric_paths(pose_dir, "*.txt")

    selected_img_paths = img_paths[: target_num_frames * stride : stride]
    selected_depth_paths = depth_paths[: target_num_frames * stride : stride]
    selected_pose_paths = pose_paths[: target_num_frames * stride : stride]

    color_90_dir = output_seq / "color_90"
    depth_90_dir = output_seq / "depth_90"
    color_90_dir.mkdir(parents=True, exist_ok=True)
    depth_90_dir.mkdir(parents=True, exist_ok=True)

    for index, (img_path, depth_path) in enumerate(zip(selected_img_paths, selected_depth_paths)):
        shutil.copy(img_path, color_90_dir / f"frame_{index:04d}.jpg")
        shutil.copy(depth_path, depth_90_dir / f"frame_{index:04d}.png")

    pose_rows: List[List[object]] = []
    for pose_path in selected_pose_paths:
        pose_rows.append(read_pose_txt_flat(pose_path))
    _write_pose_lines(output_seq / "pose_90.txt", pose_rows)

    return {
        "sequence": source_seq.name,
        "source_seq": str(source_seq),
        "output_seq": str(output_seq),
        "num_source_frames": len(img_paths),
        "num_selected_frames": len(selected_img_paths),
    }


def _list_sequence_dirs(source_root: Path) -> List[Path]:
    return sorted(path for path in source_root.iterdir() if path.is_dir())


def prepare_dataset(
    dataset_name: str,
    source_root: Path,
    output_root: Path,
    limit_seqs: int = 0,
) -> Dict[str, object]:
    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Dataset source root not found: {source_root}")

    handlers: Dict[str, Callable[[Path, Path], Dict[str, object]]] = {
        "tum": prepare_tum_sequence,
        "scannetv2": prepare_scannet_sequence,
    }
    if dataset_name not in handlers:
        raise ValueError(f"Unsupported exact preparation dataset: {dataset_name}")

    seq_dirs = _list_sequence_dirs(source_root)
    if limit_seqs > 0:
        seq_dirs = seq_dirs[:limit_seqs]

    output_root.mkdir(parents=True, exist_ok=True)
    per_sequence = [handlers[dataset_name](seq_dir, output_root / seq_dir.name) for seq_dir in seq_dirs]
    summary = {
        "dataset": dataset_name,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "num_sequences": len(per_sequence),
        "sequences": per_sequence,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
