#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare ETH3D in the same directory layout used by Pi3/CUT3R mv_recon official code.

Input raw root layout:
  <raw_root>/rgb/<scene>/dslr_calibration_jpg/{cameras.txt,images.txt}
  <raw_root>/rgb/<scene>/images/dslr_images/*.JPG
  <raw_root>/depth/<scene>_dslr_depth/<scene>/ground_truth_depth/dslr_images/*.JPG

Output layout:
  <output_root>/<scene>/images/custom_undistorted/*.JPG
  <output_root>/<scene>/ground_truth_depth/custom_undistorted/*.JPG  # raw float32 binary
  <output_root>/<scene>/custom_undistorted_cam/*.npz
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from easyvolcap.utils.colmap_utils import qvec2rotmat


TRAIN_SCENES = [
    "courtyard",
    "delivery_area",
    "electro",
    "facade",
    "kicker",
    "meadow",
    "office",
    "pipes",
    "playground",
    "relief",
    "relief_2",
    "terrace",
    "terrains",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ETH3D in Pi3 official mv_recon format.")
    parser.add_argument(
        "--raw-root",
        type=str,
        default="/horizon-bucket/saturn_v_4dlabel/008_Simulation/001_users/junyuan.deng/evaluation_datasets/metricdepth/eth3d_full",
    )
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--scenes", type=str, default="")
    parser.add_argument(
        "--seq-map",
        type=str,
        default="",
        help="Optional official ETH3D seq-id-map JSON. When set, only sampled frames are prepared.",
    )
    parser.add_argument(
        "--write-remapped-seq-map",
        type=str,
        default="",
        help="Optional output JSON path for remapped seq-id-map when --seq-map is used.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def read_cameras_txt(path: Path) -> Dict[int, dict]:
    cameras = {}
    with path.open("r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array(list(map(float, parts[4:])), dtype=np.float32)

            dist_params = {}
            if model == "SIMPLE_PINHOLE":
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model == "PINHOLE":
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif model == "THIN_PRISM_FISHEYE":
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                dist_params = {
                    "k1": params[4],
                    "k2": params[5],
                    "p1": params[6],
                    "p2": params[7],
                    "k3": params[8],
                    "k4": params[9],
                    "sx1": params[10],
                    "sy1": params[11],
                }
            else:
                raise NotImplementedError(f"Unsupported ETH3D camera model: {model}")

            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
            cameras[camera_id] = {
                "K": K,
                "dist_params": dist_params,
                "model": model,
                "width": width,
                "height": height,
            }
    return cameras


def read_images_txt(path: Path) -> Dict[int, dict]:
    images = {}
    with path.open("r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])), dtype=np.float32)
        tvec = np.array(list(map(float, parts[5:8])), dtype=np.float32)
        camera_id = int(parts[8])
        image_name = parts[9]
        images[image_id] = {
            "R": qvec2rotmat(qvec).astype(np.float32),
            "T": tvec.astype(np.float32),
            "camera_id": camera_id,
            "name": image_name,
        }
        i += 2
    return images


def load_depth(path: Path, height: int, width: int) -> np.ndarray:
    depth = np.fromfile(str(path), dtype=np.float32).reshape(height, width)
    return np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def undistort_rgb_and_depth(
    rgb_image: np.ndarray,
    depthmap: np.ndarray,
    intrinsic: np.ndarray,
    dist_params_dict: dict,
):
    height, width = rgb_image.shape[:2]
    K = intrinsic.astype(np.float32)
    D = np.array(
        [
            dist_params_dict.get("k1", 0.0),
            dist_params_dict.get("k2", 0.0),
            dist_params_dict.get("k3", 0.0),
            dist_params_dict.get("k4", 0.0),
        ],
        dtype=np.float32,
    )

    K_new = K.copy()
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, (width, height), cv2.CV_16SC2)
    rgb_image_undistorted = cv2.remap(
        rgb_image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )

    v_dist, u_dist = np.indices((height, width))
    pixels_dist = np.stack([u_dist.ravel(), v_dist.ravel()], axis=-1).astype(np.float32).reshape(-1, 1, 2)
    normalized_coords = cv2.fisheye.undistortPoints(pixels_dist, K, D)

    depth_values = depthmap.ravel()
    valid_mask = np.logical_and(depth_values > 0, np.isfinite(depth_values))
    points_x = normalized_coords.ravel()[0::2][valid_mask] * depth_values[valid_mask]
    points_y = normalized_coords.ravel()[1::2][valid_mask] * depth_values[valid_mask]
    points_z = depth_values[valid_mask]

    fx_new, fy_new = K_new[0, 0], K_new[1, 1]
    cx_new, cy_new = K_new[0, 2], K_new[1, 2]
    u_new = (points_x * fx_new / points_z) + cx_new
    v_new = (points_y * fy_new / points_z) + cy_new

    depth_out = np.zeros((height, width), dtype=np.float32)
    u_new_int = np.round(u_new).astype(int)
    v_new_int = np.round(v_new).astype(int)
    valid_mask = (
        (u_new_int >= 0)
        & (u_new_int < width)
        & (v_new_int >= 0)
        & (v_new_int < height)
    )
    depth_out[v_new_int[valid_mask], u_new_int[valid_mask]] = points_z[valid_mask]
    return rgb_image_undistorted, depth_out, K_new.astype(np.float32)


def process_scene(
    raw_root: Path,
    output_root: Path,
    scene: str,
    skip_existing: bool,
    selected_ids: Optional[List[int]] = None,
) -> None:
    cameras = read_cameras_txt(raw_root / "rgb" / scene / "dslr_calibration_jpg" / "cameras.txt")
    images = read_images_txt(raw_root / "rgb" / scene / "dslr_calibration_jpg" / "images.txt")
    sorted_items = sorted(images.items(), key=lambda kv: kv[1]["name"])
    if selected_ids is not None:
        sorted_items = [sorted_items[idx] for idx in selected_ids]

    image_dir = output_root / scene / "images" / "custom_undistorted"
    depth_dir = output_root / scene / "ground_truth_depth" / "custom_undistorted"
    cam_dir = output_root / scene / "custom_undistorted_cam"
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)

    for _, meta in sorted_items:
        out_img = image_dir / Path(meta["name"]).name
        out_depth = depth_dir / Path(meta["name"]).name
        out_cam = cam_dir / (Path(meta["name"]).stem + ".npz")
        if skip_existing and out_img.exists() and out_depth.exists() and out_cam.exists():
            continue

        img_path = raw_root / "rgb" / scene / "images" / meta["name"]
        depth_path = raw_root / "depth" / f"{scene}_dslr_depth" / scene / "ground_truth_depth" / meta["name"]
        rgb_image = np.array(Image.open(img_path))
        height, width = rgb_image.shape[:2]
        depthmap = load_depth(depth_path, height=height, width=width)

        cam_meta = cameras[meta["camera_id"]]
        if cam_meta["model"] == "THIN_PRISM_FISHEYE":
            rgb_undist, depth_undist, intrinsic = undistort_rgb_and_depth(
                rgb_image=rgb_image,
                depthmap=depthmap,
                intrinsic=cam_meta["K"],
                dist_params_dict=cam_meta["dist_params"],
            )
        else:
            rgb_undist = rgb_image
            depth_undist = depthmap
            intrinsic = cam_meta["K"].astype(np.float32)

        Image.fromarray(rgb_undist).save(out_img)
        depth_undist.astype(np.float32).tofile(out_depth)

        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = meta["R"]
        extrinsic[:3, 3] = meta["T"]
        np.savez(out_cam, intrinsics=intrinsic, extrinsics=extrinsic)


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()] if args.scenes else TRAIN_SCENES
    seq_map = None
    remapped_seq_map = {}
    if args.seq_map:
        with Path(args.seq_map).open("r") as f:
            seq_map = json.load(f)
    for scene in scenes:
        scene_ids = None if seq_map is None else [int(x) for x in seq_map[scene]]
        process_scene(
            raw_root=raw_root,
            output_root=output_root,
            scene=scene,
            skip_existing=args.skip_existing,
            selected_ids=scene_ids,
        )
        if scene_ids is not None:
            remapped_seq_map[scene] = list(range(len(scene_ids)))
        print(f"[prepare-eth3d-pi3-style] done scene={scene}", flush=True)
    if args.write_remapped_seq_map:
        seq_map_path = Path(args.write_remapped_seq_map).resolve()
        seq_map_path.parent.mkdir(parents=True, exist_ok=True)
        with seq_map_path.open("w") as f:
            json.dump(remapped_seq_map, f, indent=2)
        print(f"[prepare-eth3d-pi3-style] wrote remapped seq-map: {seq_map_path}", flush=True)


if __name__ == "__main__":
    main()
