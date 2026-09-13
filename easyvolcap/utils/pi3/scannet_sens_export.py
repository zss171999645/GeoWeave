from __future__ import annotations

import json
import os
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Sequence, Type

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


class RGBDFrame:
    def load(self, file_handle) -> None:
        import numpy as np

        self.camera_to_world = np.asarray(struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = file_handle.read(self.color_size_bytes)
        self.depth_data = file_handle.read(self.depth_size_bytes)

    def decompress_depth(self, compression_type: str) -> bytes:
        if compression_type == "zlib_ushort":
            return zlib.decompress(self.depth_data)
        raise ValueError(f"Unsupported ScanNet depth compression type: {compression_type}")

    def decompress_color(self, compression_type: str):
        if compression_type != "jpeg":
            raise ValueError(f"Unsupported ScanNet color compression type: {compression_type}")
        import cv2
        import numpy as np

        encoded = np.frombuffer(self.color_data, dtype=np.uint8)
        color = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if color is None:
            raise ValueError("Failed decoding ScanNet JPEG color frame")
        return color


class SensorData:
    version = 4

    def __init__(self, filename: str | Path):
        self.load(filename)

    def load(self, filename: str | Path) -> None:
        import numpy as np

        with open(filename, "rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            if self.version != version:
                raise ValueError(f"Unsupported ScanNet sens version: {version}, expected {self.version}")
            strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = f.read(strlen).decode("utf-8", errors="replace")
            self.intrinsic_color = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack("i", f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack("i", f.read(4))[0]]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]
            self.frames: List[RGBDFrame] = []
            for _ in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    @staticmethod
    def _ensure_dir(output_path: str | Path) -> Path:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def _save_mat_to_file(matrix, filename: str | Path) -> None:
        import numpy as np

        with open(filename, "w", encoding="utf-8") as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt="%f")

    def export_depth_images(
        self,
        output_path: str | Path,
        image_size=None,
        frame_skip: int = 1,
        max_frames: int | None = None,
    ) -> None:
        import cv2
        import numpy as np

        output_dir = self._ensure_dir(output_path)
        num_frames = len(self.frames) if max_frames is None else min(len(self.frames), max_frames)
        for frame_idx in range(0, num_frames, frame_skip):
            depth_data = self.frames[frame_idx].decompress_depth(self.depth_compression_type)
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
            if image_size is not None:
                depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(output_dir / f"{frame_idx}.png"), depth)

    def export_color_images(
        self,
        output_path: str | Path,
        image_size=None,
        frame_skip: int = 1,
        max_frames: int | None = None,
    ) -> None:
        import cv2

        output_dir = self._ensure_dir(output_path)
        num_frames = len(self.frames) if max_frames is None else min(len(self.frames), max_frames)
        for frame_idx in range(0, num_frames, frame_skip):
            color = self.frames[frame_idx].decompress_color(self.color_compression_type)
            if image_size is not None:
                color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(output_dir / f"{frame_idx}.jpg"), color)

    def export_poses(self, output_path: str | Path, frame_skip: int = 1, max_frames: int | None = None) -> None:
        output_dir = self._ensure_dir(output_path)
        num_frames = len(self.frames) if max_frames is None else min(len(self.frames), max_frames)
        for frame_idx in range(0, num_frames, frame_skip):
            self._save_mat_to_file(self.frames[frame_idx].camera_to_world, output_dir / f"{frame_idx}.txt")

    def export_intrinsics(self, output_path: str | Path) -> None:
        output_dir = self._ensure_dir(output_path)
        self._save_mat_to_file(self.intrinsic_color, output_dir / "intrinsic_color.txt")
        self._save_mat_to_file(self.extrinsic_color, output_dir / "extrinsic_color.txt")
        self._save_mat_to_file(self.intrinsic_depth, output_dir / "intrinsic_depth.txt")
        self._save_mat_to_file(self.extrinsic_depth, output_dir / "extrinsic_depth.txt")


def _count_files(directory: Path, pattern: str) -> int:
    return len(list(directory.glob(pattern))) if directory.is_dir() else 0


def list_scannet_sens_paths(source_root: str | Path, limit_seqs: int = 0) -> List[Path]:
    source_root = Path(source_root).expanduser().resolve()
    if source_root.is_file():
        if source_root.suffix.lower() != ".sens":
            raise FileNotFoundError(f"Expected a .sens file, got: {source_root}")
        paths = [source_root]
    elif source_root.is_dir():
        paths = sorted(path for path in source_root.iterdir() if path.is_file() and path.suffix.lower() == ".sens")
        if not paths:
            raise FileNotFoundError(f"No ScanNet .sens files found under: {source_root}")
    else:
        raise FileNotFoundError(f"ScanNet sens source root not found: {source_root}")
    if limit_seqs > 0:
        paths = paths[:limit_seqs]
    return paths


def export_scannet_sens_sequence(
    sens_path: str | Path,
    output_seq: str | Path,
    sensor_data_cls: Type[SensorData] | None = None,
    max_frames: int | None = None,
) -> Dict[str, object]:
    sens_path = Path(sens_path).expanduser().resolve()
    output_seq = Path(output_seq).expanduser().resolve()
    if not sens_path.is_file():
        raise FileNotFoundError(f"Missing ScanNet sens file: {sens_path}")

    sensor_data_cls = SensorData if sensor_data_cls is None else sensor_data_cls
    sensor_data = sensor_data_cls(str(sens_path))

    color_dir = output_seq / "color"
    depth_dir = output_seq / "depth"
    pose_dir = output_seq / "pose"
    intrinsic_dir = output_seq / "intrinsic"
    sensor_data.export_color_images(str(color_dir), max_frames=max_frames)
    sensor_data.export_depth_images(str(depth_dir), max_frames=max_frames)
    sensor_data.export_poses(str(pose_dir), max_frames=max_frames)
    sensor_data.export_intrinsics(str(intrinsic_dir))

    return {
        "sequence": sens_path.stem,
        "sens_path": str(sens_path),
        "output_seq": str(output_seq),
        "num_source_frames": len(sensor_data.frames),
        "num_requested_raw_frames": max_frames,
        "num_exported_color_frames": _count_files(color_dir, "*.jpg"),
        "num_exported_depth_frames": _count_files(depth_dir, "*.png"),
        "num_exported_pose_frames": _count_files(pose_dir, "*.txt"),
    }


def export_scannet_sens_dataset(
    source_root: str | Path,
    raw_output_root: str | Path,
    limit_seqs: int = 0,
    sensor_data_cls: Type[SensorData] | None = None,
    max_frames: int | None = None,
) -> Dict[str, object]:
    source_root = Path(source_root).expanduser().resolve()
    raw_output_root = Path(raw_output_root).expanduser().resolve()
    sens_paths = list_scannet_sens_paths(source_root, limit_seqs=limit_seqs)
    raw_output_root.mkdir(parents=True, exist_ok=True)
    summaries = [
        export_scannet_sens_sequence(
            sens_path=sens_path,
            output_seq=raw_output_root / sens_path.stem,
            sensor_data_cls=sensor_data_cls,
            max_frames=max_frames,
        )
        for sens_path in sens_paths
    ]
    summary = {
        "source_root": str(source_root),
        "raw_output_root": str(raw_output_root),
        "num_sens_files": len(sens_paths),
        "max_raw_frames": max_frames,
        "sequences": summaries,
    }
    return summary


def write_scannet_sens_summary(summary_path: str | Path, summary: Dict[str, object]) -> None:
    summary_path = Path(summary_path).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
