#!/usr/bin/env python3
"""Run a small GeoWeave/Pi3 inference smoke test on a folder of images."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DEFAULT_WEIGHTS_ROOT = Path(
    "/mnt/cfs/zhoufeng/geoweave_rebuttal_code_weights_20260604/"
    "weights/geoweave_paper_handoff_20260602_final"
)
DEFAULT_PI3_ROOT = DEFAULT_WEIGHTS_ROOT / "pi3_geoweave_native_sparse_20260505_checkpoint_79"
DEFAULT_CKPT = DEFAULT_PI3_ROOT / "checkpoint_79" / "pytorch_model.bin"
DEFAULT_CONFIG = DEFAULT_PI3_ROOT / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing input_v*.png/jpg images.")
    parser.add_argument("--output-dir", required=True, help="Directory for smoke-test outputs.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT), help="GeoWeave/Pi3 checkpoint path.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="GeoWeave/Pi3 Hydra config path.")
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--load-img-size", type=int, default=224)
    parser.add_argument("--device", default="cuda", help="Use cuda with CUDA_VISIBLE_DEVICES for a specific GPU.")
    parser.add_argument("--point-source", choices=["native", "depth_pose"], default="native")
    parser.add_argument("--point-stride", type=int, default=8, help="Spatial stride for the preview PLY.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def collect_images(input_dir: Path, max_images: int) -> list[Path]:
    image_paths = []
    for pattern in ("input_v*.png", "input_v*.jpg", "input_v*.jpeg", "*.png", "*.jpg", "*.jpeg"):
        for path in sorted(input_dir.glob(pattern)):
            if path.is_file() and path not in image_paths:
                image_paths.append(path)
    image_paths = [path for path in image_paths if "column" not in path.stem.lower()]
    if max_images > 0:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise FileNotFoundError(f"No input images found under {input_dir}")
    return image_paths


def read_rgb_arrays(image_paths: list[Path], target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    arrays = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        arrays.append(np.asarray(img, dtype=np.uint8))
    return np.stack(arrays, axis=0)


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray, stride: int) -> int:
    stride = max(int(stride), 1)
    sampled_points = points[:, ::stride, ::stride, :].reshape(-1, 3)
    sampled_colors = colors[:, ::stride, ::stride, :].reshape(-1, 3)
    finite = np.isfinite(sampled_points).all(axis=1)
    bounded = np.abs(sampled_points).max(axis=1) < 1.0e6
    valid = finite & bounded
    sampled_points = sampled_points[valid].astype(np.float32, copy=False)
    sampled_colors = sampled_colors[valid].astype(np.uint8, copy=False)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(sampled_points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for xyz, rgb in zip(sampled_points, sampled_colors):
            handle.write(
                f"{xyz[0]:.7g} {xyz[1]:.7g} {xyz[2]:.7g} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )
    return int(len(sampled_points))


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()

    image_paths = collect_images(input_dir, args.max_images)
    first_image = Image.open(image_paths[0]).convert("RGB")
    data_size = (first_image.height, first_image.width)
    metadata = {
        "repo_root": str(repo_root),
        "input_dir": str(input_dir),
        "image_paths": [str(path) for path in image_paths],
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint),
        "config": str(config),
        "max_images": int(args.max_images),
        "load_img_size": int(args.load_img_size),
        "device": str(args.device),
        "point_source": str(args.point_source),
        "data_size_hw": list(data_size),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    if args.dry_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
        return

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not config.is_file():
        raise FileNotFoundError(f"Config not found: {config}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    pi3_native_root = repo_root / "aidi" / "third_party" / "pi3_training"
    if str(pi3_native_root) not in sys.path:
        sys.path.insert(0, str(pi3_native_root))

    os.environ["PI3_MODEL_IMPL"] = "native_sparse"
    os.environ["PI3_CONFIG"] = str(config)
    os.environ["PI3_NATIVE_ROOT"] = str(pi3_native_root)
    os.environ.setdefault("PI3_INDEXER_EVAL_MODE", "auto")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from aidi.scripts.baselines.eval_pi3_mv_recon_core import infer_pi3_mv_pointclouds, load_pi3_model

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    print(f"[smoke] loading model checkpoint={checkpoint}", flush=True)
    model, loaded_checkpoint = load_pi3_model(None, str(checkpoint), device=device)
    metadata["loaded_checkpoint"] = loaded_checkpoint
    metadata["torch_version"] = torch.__version__
    metadata["cuda_available"] = bool(torch.cuda.is_available())
    metadata["cuda_device_count"] = int(torch.cuda.device_count())
    if device.type == "cuda":
        metadata["cuda_device_name"] = torch.cuda.get_device_name(device)

    print(f"[smoke] running inference views={len(image_paths)} data_size={data_size}", flush=True)
    points = infer_pi3_mv_pointclouds(
        filelist=[str(path) for path in image_paths],
        model=model,
        load_img_size=int(args.load_img_size),
        device=str(args.device),
        verbose=bool(args.verbose),
        data_size=data_size,
        point_source=str(args.point_source),
    )
    points = points.astype(np.float32, copy=False)
    colors = read_rgb_arrays(image_paths, data_size)

    npz_path = output_dir / "points.npz"
    ply_path = output_dir / f"points_stride{max(int(args.point_stride), 1)}.ply"
    metadata_path = output_dir / "metadata.json"
    np.savez_compressed(npz_path, points=points, colors=colors, image_paths=np.array(metadata["image_paths"]))
    ply_vertices = write_ascii_ply(ply_path, points=points, colors=colors, stride=args.point_stride)

    metadata["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    metadata["points_shape"] = list(points.shape)
    metadata["points_dtype"] = str(points.dtype)
    metadata["points_finite_ratio"] = float(np.isfinite(points).all(axis=-1).mean())
    metadata["points_min"] = [float(x) for x in np.nanmin(points.reshape(-1, 3), axis=0)]
    metadata["points_max"] = [float(x) for x in np.nanmax(points.reshape(-1, 3), axis=0)]
    metadata["npz_path"] = str(npz_path)
    metadata["ply_path"] = str(ply_path)
    metadata["ply_vertices"] = int(ply_vertices)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[smoke] wrote {npz_path}", flush=True)
    print(f"[smoke] wrote {ply_path} vertices={ply_vertices}", flush=True)
    print(f"[smoke] wrote {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
