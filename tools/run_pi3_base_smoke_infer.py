#!/usr/bin/env python3
"""Run a small official Pi3 inference smoke test on a folder of images."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


DEFAULT_WEIGHTS_ROOT = Path(
    "/mnt/cfs/zhoufeng/geoweave_rebuttal_code_weights_20260604/"
    "weights/geoweave_paper_handoff_20260602_final"
)
DEFAULT_CKPT = DEFAULT_WEIGHTS_ROOT / "pi3_base_yyfz233" / "Pi3_model.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--load-img-size", type=int, default=518)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--point-source", choices=["native", "depth_pose"], default="native")
    parser.add_argument("--point-stride", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Ensure the shared loader selects official Pi3 rather than native-sparse GeoWeave.
    for key in ("PI3_MODEL_IMPL", "PI3_CONFIG", "PI3_NATIVE_ROOT", "PI3_INDEXER_EVAL_MODE"):
        os.environ.pop(key, None)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from tools.run_pi3_geoweave_smoke_infer import collect_images, read_rgb_arrays, write_ascii_ply
    from aidi.scripts.baselines.eval_pi3_mv_recon_core import infer_pi3_mv_pointclouds, load_pi3_model

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    image_paths = collect_images(input_dir, args.max_images)
    from PIL import Image

    first_image = Image.open(image_paths[0]).convert("RGB")
    data_size = (first_image.height, first_image.width)
    metadata = {
        "repo_root": str(repo_root),
        "input_dir": str(input_dir),
        "image_paths": [str(path) for path in image_paths],
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint),
        "max_images": int(args.max_images),
        "load_img_size": int(args.load_img_size),
        "device": str(args.device),
        "point_source": str(args.point_source),
        "data_size_hw": list(data_size),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[base-smoke] loading official Pi3 checkpoint={checkpoint}", flush=True)
    model, loaded_checkpoint = load_pi3_model(None, str(checkpoint), device=device)
    metadata["loaded_checkpoint"] = loaded_checkpoint
    metadata["torch_version"] = torch.__version__
    metadata["cuda_available"] = bool(torch.cuda.is_available())
    metadata["cuda_device_count"] = int(torch.cuda.device_count())
    if device.type == "cuda":
        metadata["cuda_device_name"] = torch.cuda.get_device_name(device)

    print(f"[base-smoke] running inference views={len(image_paths)} data_size={data_size}", flush=True)
    points = infer_pi3_mv_pointclouds(
        filelist=[str(path) for path in image_paths],
        model=model,
        load_img_size=int(args.load_img_size),
        device=str(args.device),
        verbose=bool(args.verbose),
        data_size=data_size,
        point_source=str(args.point_source),
    ).astype(np.float32, copy=False)
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
    print(f"[base-smoke] wrote {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
