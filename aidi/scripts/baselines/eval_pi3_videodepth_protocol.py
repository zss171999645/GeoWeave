#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvf
from PIL import Image
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2

_ORIGINAL_ARGV = sys.argv[:]
try:
    # easyvolcap.engine parses sys.argv at import time; hide this script's argparse
    # flags so standalone baseline evals do not get interpreted as EVC overrides.
    sys.argv = [sys.argv[0]]
    from easyvolcap.engine import Config, MODELS
finally:
    sys.argv = _ORIGINAL_ARGV
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.pi3.kitti_videodepth import (
    explain_kitti_videodepth_layout_error,
    has_kitti_monst3r_gathered_layout,
    has_kitti_official_flat_layout,
)
from aidi.scripts.baselines.eval_config_utils import load_resolved_config
from aidi.scripts.baselines.exr_read_utils import read_exr_depth


TAG_FLOAT = 202021.25
DEFAULT_HF_CKPT = "yyfz233/Pi3"
DEFAULT_LOAD_IMG_SIZE = 512
DEFAULT_MAX_SIZE = 1036
DEFAULT_ALIGN_SIZE = 14
DEFAULT_SAFE_BOUND = 4
DEFAULT_VGGT_OFFICIAL_ROOT = (
    "/horizon-bucket/saturn_v_dev/01_users/tao02.xie/projects/GFM/gfm/data/trained_model/vggt/official/VGGT-1B"
)
DEFAULT_VGGT_PT34_CKPT = (
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/trained_model/vggt/official/"
    "finetune_5090_sparse_20260226_0139_resume_finetune_5090_sparse_topk2048-layers9_19-resumept30-fp16-pointhead/34.pt"
)
DEFAULT_DA3_REPO_CANDIDATES = (
    "third_party/Depth-Anything-3",
    "tmp/depthanything3_fetch/Depth-Anything-3",
    "/home/feng01.zhou/workspace/meshx_pi3_depthgt79_eval_2cec9d45_20260512_2223/third_party/Depth-Anything-3",
)
DEFAULT_DA3_MODEL_CANDIDATES = (
    "tmp/pretrained_official/depth-anything__DA3-SMALL",
    "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/pretrained/depthanything3/depth-anything__DA3-SMALL",
    "/home/feng01.zhou/workspace/meshx_pi3_depthgt79_eval_2cec9d45_20260512_2223/tmp/pretrained_official/depth-anything__DA3-SMALL",
)
KITTI_OFFICIAL_FLAT_IMAGE_RE = re.compile(
    r"(?P<drive>.+)_image_(?P<frame>\d+)_image_(?P<camera>\d{2})\.png$"
)
KITTI_OFFICIAL_FLAT_GT_RE = re.compile(
    r"(?P<drive>.+)_groundtruth_depth_(?P<frame>\d+)_image_(?P<camera>\d{2})\.png$"
)
SINTEL_SEQUENCES = [
    "alley_2",
    "ambush_4",
    "ambush_5",
    "ambush_6",
    "cave_2",
    "cave_4",
    "market_2",
    "market_5",
    "market_6",
    "shaman_3",
    "sleeping_1",
    "sleeping_2",
    "temple_2",
    "temple_3",
]


def first_existing_path(candidates: Sequence[str]) -> str:
    for item in candidates:
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = _REPO_ROOT / path
        if path.is_dir():
            return str(path)
    first = Path(candidates[0]).expanduser()
    return str(first if first.is_absolute() else _REPO_ROOT / first)
BONN_SEQUENCES = [
    "rgbd_bonn_balloon2",
    "rgbd_bonn_crowd2",
    "rgbd_bonn_crowd3",
    "rgbd_bonn_person_tracking2",
    "rgbd_bonn_synchronous",
]
# Same scene layout as easyvolcap GeneralizableDataset: <meta_root>/<scene>/images/<cam>/...
MULTIVIEW_EVC_DATASETS = frozenset({"kitti", "eth3d", "vkitti2", "co3dv2", "generic_npy", "generic_png_mm"})
VIDEO_PAPER_TARGETS = {
    "pi3": {
        "sintel": {"Abs Rel": 0.233},
        "bonn": {"Abs Rel": 0.049, "δ < 1.25": 0.975},
        "kitti": {"Abs Rel": 0.038, "δ < 1.25": 0.986, "fps": 57.4},
    },
    "vggt": {
        "sintel": {"Abs Rel": 0.299},
        "bonn": {"Abs Rel": 0.057, "δ < 1.25": 0.966},
        "kitti": {"Abs Rel": 0.062, "δ < 1.25": 0.969, "fps": 43.2},
    },
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    img_subdir: str
    gt_subdir: str
    img_ext: str
    gt_ext: str
    depth_reader: Callable[[str], np.ndarray]
    max_depth: Optional[float]
    post_clip_max: Optional[float]
    sequence_names: Optional[Sequence[str]] = None
    # If True, apply foreground masks from <seq>/masks/<camera>/<image_stem>.png when present (co3dv2 / configs).
    use_fg_masks: bool = False
    # Exclude GT below this (meters) when computing metrics; avoids Abs Rel / Sq Rel blow-up from near-zero denominators.
    min_depth: Optional[float] = None


def depth_read_sintel(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        if check != TAG_FLOAT:
            raise ValueError(f"Wrong Sintel depth tag: expected {TAG_FLOAT}, got {check}")
        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        height = np.fromfile(f, dtype=np.int32, count=1)[0]
        size = width * height
        if width <= 0 or height <= 0 or size <= 1 or size >= 100000000:
            raise ValueError(f"Invalid Sintel depth size: width={width}, height={height}")
        depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


def depth_read_kitti(filename: str) -> np.ndarray:
    depth_png = np.array(Image.open(filename), dtype=np.int32)
    if np.max(depth_png) <= 255:
        raise ValueError(f"KITTI depth is not 16-bit: {filename}")
    depth = depth_png.astype(np.float32) / 256.0
    depth[depth_png == 0] = -1.0
    return depth


def depth_read_bonn(filename: str) -> np.ndarray:
    depth_png = np.array(Image.open(filename), dtype=np.int32)
    if np.max(depth_png) <= 255:
        raise ValueError(f"Bonn depth is not 16-bit: {filename}")
    depth = depth_png.astype(np.float32) / 5000.0
    depth[depth_png == 0] = -1.0
    return depth


def depth_read_png_mm(filename: str) -> np.ndarray:
    depth_png = np.array(Image.open(filename), dtype=np.int32)
    if np.max(depth_png) <= 255:
        raise ValueError(f"Millimeter depth PNG is not 16-bit: {filename}")
    depth = depth_png.astype(np.float32) / 1000.0
    depth[depth_png == 0] = -1.0
    return depth


def depth_read_exr(filename: str) -> np.ndarray:
    return read_exr_depth(filename)


# EVC view-major layouts:
# - eth3d / 7scenes / blendedmvs / scannetpp (scannetpp_to_easyvolcap, etc.): images/<view06d>/000000.(jpg|png) + depths/<view06d>/000000.exr
# - mvs_synth (mvs_synth_to_easyvolcap): images/<view04d>/000000.png + depths/<view04d>/000000.exr


def _view_major_dir_pattern(digit_width: int) -> re.Pattern:
    return re.compile(rf"^\d{{{digit_width}}}$")


def is_view_major_evc_layout(seq_path: Path, digit_width: int) -> bool:
    pat = _view_major_dir_pattern(digit_width)
    img_root = seq_path / "images"
    if not img_root.is_dir():
        return False
    view_dirs = [p for p in img_root.iterdir() if p.is_dir() and pat.fullmatch(p.name)]
    if not view_dirs:
        return False
    vd = sorted(view_dirs, key=lambda p: p.name)[0]
    if not any(vd.glob("*.jpg")) and not any(vd.glob("*.png")) and not any(vd.glob("*.jpeg")):
        return False
    ddir = seq_path / "depths" / vd.name
    if not ddir.is_dir():
        return False
    return bool(any(ddir.glob("*.exr")) or any(ddir.glob("*.npy")))


def is_eth3d_evc_train_layout(seq_path: Path) -> bool:
    return is_view_major_evc_layout(seq_path, 6)


def is_eth3d_pi3_style_layout(seq_path: Path) -> bool:
    image_root = seq_path / "images" / "custom_undistorted"
    depth_root = seq_path / "ground_truth_depth" / "custom_undistorted"
    return image_root.is_dir() and depth_root.is_dir()


def is_mvs_synth_evc_layout(seq_path: Path) -> bool:
    return is_view_major_evc_layout(seq_path, 4)


def list_view_major_evc_frame_pairs(seq_path: Path, digit_width: int) -> Tuple[List[Path], List[Path]]:
    """Ordered (rgb, depth) for each view subdir under images/ (videodepth stack)."""
    pat = _view_major_dir_pattern(digit_width)
    img_root = seq_path / "images"
    view_dirs = sorted([p for p in img_root.iterdir() if p.is_dir() and pat.fullmatch(p.name)])
    image_files: List[Path] = []
    gt_files: List[Path] = []
    for vd in view_dirs:
        inner = vd / "000000.jpg"
        if not inner.is_file():
            inner = vd / f"{vd.name}.jpg"
        if not inner.is_file():
            inner = vd / "000000.png"
        if not inner.is_file():
            inner = vd / f"{vd.name}.png"
        if not inner.is_file():
            cands = sorted(
                [p for p in vd.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".JPG"}]
            )
            if not cands:
                continue
            inner = cands[0]
        dp_dir = seq_path / "depths" / vd.name
        if not dp_dir.is_dir():
            continue
        depth = dp_dir / "000000.exr"
        if not depth.is_file():
            depth = dp_dir / f"{vd.name}.exr"
        if not depth.is_file():
            depth = dp_dir / "000000.npy"
        if not depth.is_file():
            exrs = sorted(dp_dir.glob("*.exr"))
            npys = sorted(dp_dir.glob("*.npy"))
            depth = exrs[0] if exrs else (npys[0] if npys else None)
        if depth is None or not depth.is_file():
            continue
        image_files.append(inner)
        gt_files.append(depth)
    return image_files, gt_files


def list_eth3d_evc_train_frame_pairs(seq_path: Path) -> Tuple[List[Path], List[Path]]:
    return list_view_major_evc_frame_pairs(seq_path, 6)


def list_eth3d_pi3_style_frame_pairs(seq_path: Path) -> Tuple[List[Path], List[Path]]:
    image_root = seq_path / "images" / "custom_undistorted"
    depth_root = seq_path / "ground_truth_depth" / "custom_undistorted"
    image_files: List[Path] = []
    gt_files: List[Path] = []
    for image_path in sorted(image_root.iterdir(), key=lambda p: p.name):
        if not image_path.is_file() or image_path.name.startswith("."):
            continue
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        depth_path = depth_root / image_path.name
        if not depth_path.is_file():
            continue
        image_files.append(image_path)
        gt_files.append(depth_path)
    return image_files, gt_files


def list_mvs_synth_evc_frame_pairs(seq_path: Path) -> Tuple[List[Path], List[Path]]:
    return list_view_major_evc_frame_pairs(seq_path, 4)


# DIODE val: <root>/{indoors,outdoor}/scene_*/scan_*/*.png + <stem>_depth.npy (+ optional <stem>_depth_mask.npy)
DIODE_VAL_SPLITS = ("indoors", "outdoor")


def diode_scan_has_pairs(scan_dir: Path) -> bool:
    if not scan_dir.is_dir():
        return False
    for im in scan_dir.iterdir():
        if im.is_file() and im.suffix.lower() == ".png" and "_depth" not in im.name.lower():
            if (scan_dir / f"{im.stem}_depth.npy").is_file():
                return True
    return False


def diode_val_scene_relpaths(data_root: Path) -> List[str]:
    root = data_root.resolve()
    if not root.is_dir():
        return []
    out: List[str] = []
    for split in DIODE_VAL_SPLITS:
        sp = root / split
        if not sp.is_dir():
            continue
        for scene_dir in sorted(p for p in sp.iterdir() if p.is_dir() and p.name.startswith("scene_")):
            for scan_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir() and p.name.startswith("scan_")):
                if diode_scan_has_pairs(scan_dir):
                    out.append(scan_dir.relative_to(root).as_posix())
    return sorted(set(out))


def list_diode_scan_frame_pairs(scan_dir: Path) -> Tuple[List[Path], List[Path]]:
    imgs = sorted(
        p
        for p in scan_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".png" and "_depth" not in p.name.lower()
    )
    paired_img: List[Path] = []
    paired_gt: List[Path] = []
    for im in imgs:
        d = scan_dir / f"{im.stem}_depth.npy"
        if d.is_file():
            paired_img.append(im)
            paired_gt.append(d)
    return paired_img, paired_gt


def depth_read_diode(filename: str) -> np.ndarray:
    p = Path(filename)
    depth = np.load(p).astype(np.float32)
    mask_p = p.parent / f"{p.stem}_mask.npy"
    if mask_p.is_file():
        m = np.load(mask_p).astype(np.float32)
        if m.shape == depth.shape:
            depth = np.where(m > 0.5, depth, 0.0)
    return depth


def depth_read_npy(filename: str) -> np.ndarray:
    depth = np.load(filename).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth <= 1e-4] = 0.0
    return depth


def blendedmvs_scene_relpaths(data_root: Path) -> List[str]:
    """
    BlendedMVS EVC export: scenes may be <root>/<scene> or <root>/<subset>/<scene>
    (e.g. test/BlendedMVS++/588084032366dd5d06e59e82), each with view-major images/<view06d>/.
    """
    root = data_root.resolve()
    if not root.is_dir():
        return []
    out: List[str] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        if is_eth3d_evc_train_layout(d):
            out.append(d.relative_to(root).as_posix())
        for b in sorted([p for p in d.iterdir() if p.is_dir()]):
            if is_eth3d_evc_train_layout(b):
                out.append(b.relative_to(root).as_posix())
    return sorted(set(out))


def is_multiview_evc_dataset(name: str) -> bool:
    return name in MULTIVIEW_EVC_DATASETS


def _evc_scene_roots_from_glob_slow(data_root: Path) -> List[Path]:
    """Full recursive glob — can take minutes on huge trees (e.g. CO3Dv2 on bucket); use as last resort."""
    roots: List[Path] = []
    root = data_root.resolve()
    if not root.is_dir():
        return roots
    for images_dir in sorted(root.glob("**/images")):
        if not images_dir.is_dir():
            continue
        scene_root = images_dir.parent
        if (scene_root / "depths").is_dir():
            roots.append(scene_root.resolve())
    return sorted(set(roots), key=lambda p: str(p))


def _load_scene_roots_from_data_roots_txt(data_root: Path, *, verify_scene_dirs: bool = True) -> List[Path]:
    """GeneralizableDataset may write data_roots.txt with one relative scene path per line."""
    txt = data_root / "data_roots.txt"
    if not txt.is_file():
        return []
    root = data_root.resolve()
    out: List[Path] = []
    for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Avoid per-entry resolve() on network storage; it is much slower than simple path joins.
        scene = root / line
        if not verify_scene_dirs or ((scene / "images").is_dir() and (scene / "depths").is_dir()):
            out.append(scene)
    return sorted(set(out), key=lambda p: str(p))


def _scene_roots_to_rel_names(data_root: Path, scene_roots: Sequence[Path]) -> List[str]:
    root = data_root.resolve()
    names: List[str] = []
    for scene_root in scene_roots:
        try:
            names.append(scene_root.relative_to(root).as_posix())
        except ValueError:
            names.append(scene_root.name)
    return sorted(set(names))


def _evc_two_level_scene_roots(data_root: Path) -> List[Path]:
    """VKITTI2 / CO3Dv2 EVC: <root>/<a>/<b>/images + depths (no recursive glob)."""
    root = data_root.resolve()
    if not root.is_dir():
        return []
    out: List[Path] = []
    for a in sorted([p for p in root.iterdir() if p.is_dir()]):
        for b in sorted([p for p in a.iterdir() if p.is_dir()]):
            if (b / "images").is_dir() and (b / "depths").is_dir():
                out.append(b.resolve())
    return sorted(set(out), key=lambda p: str(p))


def _evc_one_level_scene_roots(data_root: Path) -> List[Path]:
    """KITTI / ETH3D-style: <root>/<scene>/images + depths."""
    root = data_root.resolve()
    if not root.is_dir():
        return []
    out: List[Path] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        if (d / "images").is_dir() and (d / "depths").is_dir():
            out.append(d.resolve())
    return sorted(set(out), key=lambda p: str(p))


def evc_scene_roots_discover(data_root: Path, dataset_name: str) -> List[Path]:
    """
    Fast scene listing for EVC layouts. Avoids `**/images` on CO3Dv2-scale buckets (would hang on network FS).
    """
    root = data_root.resolve()
    if not root.is_dir():
        return []
    # CO3Dv2 / VKITTI2 can ship large valid scene lists on network storage.
    # Trust the curated list there and avoid thousands of remote stat() calls.
    verify_scene_dirs = dataset_name not in ("co3dv2", "vkitti2")
    from_txt = _load_scene_roots_from_data_roots_txt(root, verify_scene_dirs=verify_scene_dirs)
    if from_txt:
        return from_txt
    if dataset_name in ("co3dv2", "vkitti2"):
        found = _evc_two_level_scene_roots(root)
        if found:
            return found
    found_one = _evc_one_level_scene_roots(root)
    if found_one:
        return found_one
    print(
        "WARNING: No scenes via data_roots.txt / one- or two-level layout; falling back to recursive **/images glob "
        "(can be very slow on large datasets).",
        flush=True,
    )
    return _evc_scene_roots_from_glob_slow(root)


def resolve_camera_for_multiview_evc(data_root: Path, camera: str, dataset_name: str = "") -> str:
    """Pick an existing camera id under <scene>/images/<cam> (falls back to first sorted cam)."""
    scene_roots = evc_scene_roots_discover(data_root, dataset_name)
    if not scene_roots:
        seq_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
        if not seq_dirs:
            return camera
        probe = seq_dirs[0]
    else:
        probe = scene_roots[0]
    images_root = probe / "images"
    if not images_root.is_dir():
        return camera
    if (images_root / camera).is_dir() and (probe / "depths" / camera).is_dir():
        return camera
    cams = sorted([p.name for p in images_root.iterdir() if p.is_dir()])
    for cam in cams:
        if (probe / "depths" / cam).is_dir():
            return cam
    return camera


def nested_multiview_evc_scene_relpaths(data_root: Path, camera: str, dataset_name: str) -> List[str]:
    """VKITTI2 / CO3Dv2: scene roots can be nested (e.g. Scene01/15-deg-left), not only data_root/<scene>."""
    root = data_root.resolve()
    rels: List[str] = []
    for scene_root in evc_scene_roots_discover(data_root, dataset_name):
        if dataset_name in ("vkitti2", "co3dv2"):
            rels.append(scene_root.relative_to(root).as_posix())
            continue
        # co3d_to_easyvolcap: images/<view06d>/000000.jpg — do not require `--camera` to match for listing.
        if dataset_name == "co3dv2" and is_eth3d_evc_train_layout(scene_root):
            rels.append(scene_root.relative_to(root).as_posix())
            continue
        if (scene_root / "images" / camera).is_dir() and (scene_root / "depths" / camera).is_dir():
            rels.append(scene_root.relative_to(root).as_posix())
    return sorted(set(rels))


def _img_ext_candidates(primary: str) -> List[str]:
    primary = primary.lower().lstrip(".")
    out: List[str] = []
    for ext in (primary, "jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
        e = ext.lower().lstrip(".")
        if e and e not in out:
            out.append(e)
    return out


def list_paired_multiview_evc_frames(image_dir: Path, gt_dir: Path, img_ext: str, gt_ext: str) -> Tuple[List[Path], List[Path]]:
    """Pair RGB and depth by matching stem (EVC: .jpg + .exr, or .png + .png for KITTI)."""
    gt_ext_l = gt_ext.lower().lstrip(".")
    gts = sorted(p for p in gt_dir.iterdir() if p.is_file() and p.suffix.lower() == f".{gt_ext_l}")
    gt_by_stem = {p.stem: p for p in gts}
    images: List[Path] = []
    for ext_try in _img_ext_candidates(img_ext):
        images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() == f".{ext_try.lower()}")
        if images:
            break
    paired_img: List[Path] = []
    paired_gt: List[Path] = []
    for im in images:
        gt = gt_by_stem.get(im.stem)
        if gt is not None:
            paired_img.append(im)
            paired_gt.append(gt)
    return paired_img, paired_gt


def apply_fg_mask_to_depth(depth: np.ndarray, mask_path: Path) -> np.ndarray:
    if not mask_path.is_file():
        return depth
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return depth
    if m.shape[:2] != depth.shape[:2]:
        m = cv2.resize(m, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = depth.astype(np.float32, copy=True)
    out[m < 128] = 0.0
    return out


def evc_fg_mask_path(data_root: Path, seq: str, image_path: Path, camera: str) -> Path:
    """masks/<subdir>/<stem>.png mirroring images/<subdir>/file when subdir exists (view-major / per-cam)."""
    images_root = (data_root / seq / "images").resolve()
    try:
        rel = image_path.resolve().relative_to(images_root)
    except ValueError:
        return data_root / seq / "masks" / camera / f"{image_path.stem}.png"
    if len(rel.parts) >= 2:
        return data_root / seq / "masks" / rel.parts[0] / f"{image_path.stem}.png"
    return data_root / seq / "masks" / camera / f"{image_path.stem}.png"


def load_gt_depth_path(gt_path: Path, spec: DatasetSpec) -> np.ndarray:
    suf = gt_path.suffix.lower()
    if suf in {".png", ".dpt"}:
        return spec.depth_reader(str(gt_path))
    if spec.name == "diode" and suf == ".npy":
        return spec.depth_reader(str(gt_path))
    return read_depth_file(gt_path)


def read_eth3d_pi3_style_depth(depth_path: Path, image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        width, height = image.size
    depth = np.fromfile(str(depth_path), dtype=np.float32)
    expected = int(height * width)
    if depth.size != expected:
        raise RuntimeError(
            f"ETH3D prepared depth size mismatch for {depth_path}: got {depth.size}, expected {expected}"
        )
    depth = depth.reshape(height, width)
    return np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def stack_gt_depth_sequence(
    spec: DatasetSpec,
    image_files: Sequence[Path],
    gt_files: Sequence[Path],
    data_root: Path,
    seq: str,
    camera: str,
) -> np.ndarray:
    layers: List[np.ndarray] = []
    for im_p, gt_p in zip(image_files, gt_files):
        if spec.name == "eth3d" and "ground_truth_depth" in gt_p.parts:
            d = read_eth3d_pi3_style_depth(gt_p, im_p)
        else:
            d = load_gt_depth_path(gt_p, spec)
        if spec.use_fg_masks:
            d = apply_fg_mask_to_depth(d, evc_fg_mask_path(data_root, seq, im_p, camera))
        layers.append(d)
    return np.stack(layers, axis=0)


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "sintel": DatasetSpec(
        name="sintel",
        img_subdir="final/{seq}",
        gt_subdir="depth/{seq}",
        img_ext="png",
        gt_ext="dpt",
        depth_reader=depth_read_sintel,
        max_depth=70.0,
        post_clip_max=70.0,
        sequence_names=SINTEL_SEQUENCES,
    ),
    "bonn": DatasetSpec(
        name="bonn",
        img_subdir="{seq}/rgb_110",
        gt_subdir="{seq}/depth_110",
        img_ext="png",
        gt_ext="png",
        depth_reader=depth_read_bonn,
        max_depth=70.0,
        post_clip_max=None,
        sequence_names=BONN_SEQUENCES,
    ),
    "kitti": DatasetSpec(
        name="kitti",
        img_subdir="image_gathered/{seq}",
        gt_subdir="groundtruth_depth_gathered/{seq}",
        img_ext="png",
        gt_ext="png",
        depth_reader=depth_read_kitti,
        max_depth=None,
        post_clip_max=None,
        sequence_names=None,
    ),
    # EVC layout: <data_root>/<scene>/images/<camera>/*.<img_ext> + depths/<camera>/*.<gt_ext>
    # Matches configs/datasets/academic.yaml & VGGT exps (eth3d,7scenes,blendedmvs,mvs_synth,scannetpp,vkitti2,co3dv2).
    "eth3d": DatasetSpec(
        name="eth3d",
        img_subdir="",
        gt_subdir="",
        img_ext="jpg",
        gt_ext="exr",
        depth_reader=depth_read_exr,
        max_depth=None,
        post_clip_max=None,
        sequence_names=None,
    ),
    "7scenes": DatasetSpec(
        name="7scenes",
        img_subdir="",
        gt_subdir="",
        img_ext="png",
        gt_ext="exr",
        depth_reader=depth_read_exr,
        max_depth=None,
        post_clip_max=None,
        sequence_names=None,
    ),
    "blendedmvs": DatasetSpec(
        name="blendedmvs",
        img_subdir="",
        gt_subdir="",
        img_ext="jpg",
        gt_ext="exr",
        depth_reader=depth_read_exr,
        max_depth=None,
        post_clip_max=None,
        sequence_names=None,
    ),
    "mvs_synth": DatasetSpec(
        name="mvs_synth",
        img_subdir="",
        gt_subdir="",
        img_ext="png",
        gt_ext="exr",
        depth_reader=depth_read_exr,
        max_depth=None,
        post_clip_max=None,
        sequence_names=None,
    ),
    "scannetpp": DatasetSpec(
        name="scannetpp",
        img_subdir="",
        gt_subdir="",
        img_ext="jpg",
        gt_ext="exr",
        depth_reader=depth_read_exr,
        max_depth=None,
        post_clip_max=None,
        sequence_names=None,
    ),
    "diode": DatasetSpec(
        name="diode",
        img_subdir="",
        gt_subdir="",
        img_ext="png",
        gt_ext="npy",
        depth_reader=depth_read_diode,
        max_depth=None,
        post_clip_max=None,
        sequence_names=None,
        min_depth=1e-2,
    ),
    "vkitti2": DatasetSpec(
        name="vkitti2",
        img_subdir="",
        gt_subdir="",
        img_ext="jpg",
        gt_ext="exr",
        depth_reader=depth_read_exr,
        max_depth=200.0,
        post_clip_max=None,
        sequence_names=None,
    ),
    "co3dv2": DatasetSpec(
        name="co3dv2",
        img_subdir="",
        gt_subdir="",
        img_ext="jpg",
        gt_ext="exr",
        depth_reader=depth_read_exr,
        max_depth=None,
        post_clip_max=None,
        sequence_names=None,
        use_fg_masks=True,
    ),
    "generic_npy": DatasetSpec(
        name="generic_npy",
        img_subdir="",
        gt_subdir="",
        img_ext="jpg",
        gt_ext="npy",
        depth_reader=depth_read_npy,
        max_depth=None,
        post_clip_max=None,
        sequence_names=None,
    ),
    "generic_png_mm": DatasetSpec(
        name="generic_png_mm",
        img_subdir="",
        gt_subdir="",
        img_ext="jpg",
        gt_ext="png",
        depth_reader=depth_read_png_mm,
        max_depth=None,
        post_clip_max=None,
        sequence_names=None,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PI3-style videodepth with the official PI3 Table 4 protocol.",
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS.keys()), required=True)
    parser.add_argument(
        "--model-family",
        choices=["pi3", "vggt", "vggt_omega", "depthanything3"],
        default="pi3",
        help="Model family used to generate sequence depth predictions.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Dataset root matching PI3 official gathered layout or EVC seq/images+depths layout.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for predictions and metrics. Defaults to tmp/pi3_videodepth_eval_<dataset>_<timestamp>.",
    )
    parser.add_argument(
        "--pred-root",
        type=str,
        default="",
        help="Prediction root. Defaults to <output-dir>/predictions/<dataset>.",
    )
    parser.add_argument(
        "--alignment",
        choices=["scale&shift", "scale", "metric"],
        default="scale&shift",
        help="Sequence depth alignment mode. PI3 videodepth defaults to scale&shift.",
    )
    parser.add_argument(
        "--layout",
        choices=["auto", "official", "evc"],
        default="auto",
        help="Dataset layout. `official` accepts PI3 gathered/raw layout or KITTI official flat layout; `evc` expects seq/images+depths.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="02",
        help="Camera id used for KITTI EVC layout. Ignored by official PI3 layout.",
    )
    parser.add_argument("--load-img-size", type=int, default=DEFAULT_LOAD_IMG_SIZE)
    parser.add_argument(
        "--max-size",
        type=int,
        default=DEFAULT_MAX_SIZE,
        help="Resize both input images and GT depth to max(H,W)=this before inference/eval (1036p style).",
    )
    parser.add_argument(
        "--align-size",
        type=int,
        default=DEFAULT_ALIGN_SIZE,
        help="Align resized H/W to this multiple (e.g. 14).",
    )
    parser.add_argument(
        "--safe-bound",
        type=int,
        default=DEFAULT_SAFE_BOUND,
        help="MVPD-style preprocessing safe bound (same meaning as MultiviewPointDataset.safe_bound).",
    )
    parser.add_argument(
        "--preprocess-style",
        choices=["simple", "mvpd"],
        default="mvpd",
        help=(
            "Input preprocessing style. "
            "`simple`: scale by max(H,W)=max_size then align to align_size. "
            "`mvpd`: mimic MultiviewPointDataset.get_sources() resize-to-safe-bound then center-crop to aligned target."
        ),
    )
    parser.add_argument(
        "--eval-gt-shape",
        choices=["preprocessed", "official"],
        default="preprocessed",
        help=(
            "Which GT shape to use for metric evaluation. "
            "`preprocessed`: resize GT to match 1036p preprocessing (default, image+gt both 1036p). "
            "`official`: keep GT at original dataset resolution and only resize predictions back for evaluation. "
            "Note: `--dataset kitti` always uses native GT resolution for metrics (same as `official`); "
            "`--eval-resize-to-1036p` is disabled for KITTI."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--eval-device",
        type=str,
        default="",
        help="Optional device for metric evaluation. Defaults to --device when empty.",
    )
    parser.add_argument("--ckpt", type=str, default="", help="Local Pi3 checkpoint path or HF model id.")
    parser.add_argument(
        "--vggt-model-tag",
        choices=["official", "pt34"],
        default="official",
        help="VGGT checkpoint mode when --model-family=vggt.",
    )
    parser.add_argument(
        "--vggt-input-preprocess",
        choices=["official", "custom"],
        default="custom",
        help=(
            "VGGT inference-time image preprocessing. "
            "`official` uses easyvolcap/official_vggt/utils/load_fn.py (518/crop|pad). "
            "`custom` uses this script's preprocessing (simple/mvpd)."
        ),
    )
    parser.add_argument(
        "--vggt-config",
        type=str,
        default="",
        help="Optional VGGT config path. Defaults follow the selected VGGT model tag.",
    )
    parser.add_argument(
        "--vggt-official-ckpt-root",
        type=str,
        default=DEFAULT_VGGT_OFFICIAL_ROOT,
        help="Root directory with official VGGT component checkpoints.",
    )
    parser.add_argument(
        "--vggt-pt34-ckpt",
        type=str,
        default=DEFAULT_VGGT_PT34_CKPT,
        help="Full pt34 OfficialVGGTModel checkpoint path.",
    )
    parser.add_argument(
        "--vggt-topk-override",
        type=int,
        default=0,
        help="Optional sparse inference top-k override for VGGT configs. Non-positive keeps the config default.",
    )
    parser.add_argument(
        "--vggt-short-topk-override",
        type=int,
        default=0,
        help="Optional VGGT sparse inference top-k override for shorter sequences. Requires --vggt-long-topk-min-frames.",
    )
    parser.add_argument(
        "--vggt-long-topk-override",
        type=int,
        default=0,
        help="Optional VGGT sparse inference top-k override for longer sequences. Requires --vggt-long-topk-min-frames.",
    )
    parser.add_argument(
        "--vggt-long-topk-min-frames",
        type=int,
        default=0,
        help="Use --vggt-long-topk-override when sequence frames >= this value, otherwise use --vggt-short-topk-override.",
    )
    parser.add_argument(
        "--vggt-dtype-override",
        type=str,
        default="",
        help="Optional VGGT model dtype override, for example `float16` or `bfloat16`.",
    )
    parser.add_argument("--vggt-omega-repo", type=str, default="", help="Local VGGT-Omega repository checkout.")
    parser.add_argument("--vggt-omega-checkpoint", type=str, default="", help="VGGT-Omega checkpoint path.")
    parser.add_argument("--vggt-omega-resolution", type=int, default=512)
    parser.add_argument("--vggt-omega-mode", choices=("balanced", "max_size"), default="balanced")
    parser.add_argument(
        "--eval-fov-policy",
        choices=("auto", "full", "vggt_omega_center_crop"),
        default="auto",
        help=(
            "Field-of-view used for depth metrics. `auto` preserves legacy behavior: full FOV for most models, "
            "VGGT-Omega center crop when its crop flag is enabled. `vggt_omega_center_crop` evaluates every "
            "model on the same center crop used by VGGT-Omega preprocessing for extreme aspect ratios."
        ),
    )
    parser.add_argument(
        "--scale-shift-fit-max-pixels",
        type=int,
        default=0,
        help=(
            "Optional deterministic pixel cap for fitting scale&shift. Metrics are still computed on all valid pixels. "
            "0 keeps the exact legacy all-pixel fit."
        ),
    )
    parser.add_argument(
        "--no-vggt-omega-crop-gt-to-input-fov",
        dest="vggt_omega_crop_gt_to_input_fov",
        action="store_false",
        help=(
            "Disable the default VGGT-Omega fair-depth correction that crops GT depth to the "
            "same center FOV used by VGGT-Omega preprocessing for extreme aspect ratios."
        ),
    )
    parser.set_defaults(vggt_omega_crop_gt_to_input_fov=True)
    parser.add_argument(
        "--vggt-omega-global-attention-mode",
        choices=("original", "camera_register_query", "query_view_image_global"),
        default="original",
        help="Experimental VGGT-Omega global attention mode. Use original for released-checkpoint baselines.",
    )
    parser.add_argument(
        "--vggt-omega-query-view-index",
        type=int,
        default=0,
        help="Query view index used when --vggt-omega-global-attention-mode=query_view_image_global.",
    )
    parser.add_argument(
        "--vggt-omega-query-view-sweep",
        action="store_true",
        help=(
            "For VGGT-Omega query_view_image_global, run each sequence once per view and evaluate only "
            "that query view's depth."
        ),
    )
    parser.add_argument("--da3-repo", default=first_existing_path(DEFAULT_DA3_REPO_CANDIDATES))
    parser.add_argument("--da3-model", default=first_existing_path(DEFAULT_DA3_MODEL_CANDIDATES))
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument(
        "--process-res-method",
        choices=("upper_bound_resize", "upper_bound_crop", "lower_bound_resize", "lower_bound_crop"),
        default="upper_bound_resize",
    )
    parser.add_argument("--ref-view-strategy", default="first")
    parser.add_argument("--use-ray-pose", action="store_true")
    parser.add_argument("--align-to-input-ext-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-only", action="store_true", help="Skip inference and only evaluate existing .npy predictions.")
    parser.add_argument(
        "--eval-resize-to-1036p",
        action="store_true",
        help=(
            "Evaluate in 1036p space: resize BOTH pred+GT to (max(H,W)=max_size, align=align_size) "
            "before computing metrics. Inference input is unchanged."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions.")
    parser.add_argument("--save-png", action="store_true", help="Also save visualization PNGs per frame.")
    parser.add_argument(
        "--save-gt-png",
        action="store_true",
        help="Also save GT visualization PNGs in the evaluation space (e.g. 1036p when preprocessed).",
    )
    parser.add_argument(
        "--save-aligned-pred-png",
        action="store_true",
        help="Also save aligned prediction PNGs (pred resized to GT eval shape).",
    )
    parser.add_argument(
        "--save-image-png",
        action="store_true",
        help="Also save input RGB image PNGs for visualization (in the evaluation space when applicable).",
    )
    parser.add_argument(
        "--vis-max-frames",
        type=int,
        default=0,
        help="Limit number of frames saved for visualization per sequence (0 means all).",
    )
    parser.add_argument("--max-seqs", type=int, default=0, help="Optional sequence limit for smoke testing.")
    parser.add_argument("--max-frames-per-seq", type=int, default=0, help="Optional frame limit per sequence for smoke testing.")
    parser.add_argument(
        "--eval-frame-indices",
        type=str,
        default="",
        help=(
            "Optional comma-separated frame indices, or `tuple_meta`, used only for depth metrics. "
            "Inference still consumes the full sequence; useful for clean-core target-only diagnostics."
        ),
    )
    return parser.parse_args()


def resolve_default_output_dir(dataset: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("tmp") / f"pi3_videodepth_eval_{dataset}_{timestamp}"


def log_progress(message: str) -> None:
    print(message, flush=True)


def append_progress_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()


def resolve_default_ckpt() -> str:
    env_ckpt = os.environ.get("PI3_PRETRAIN_CKPT", "").strip()
    if env_ckpt and Path(env_ckpt).is_file():
        return env_ckpt

    user_name = os.environ.get("USER", "feng01.zhou")
    if not user_name or user_name.isdigit():
        user_name = "feng01.zhou"
    candidates = [
        Path(f"/horizon-bucket/saturn_v_dev/01_users/{user_name}/projects/meshx/baseline/pretrained/pi3/yyfz233_Pi3_model.safetensors"),
        Path("weights/pi3/model.safetensors"),
    ]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError:
            continue
    return DEFAULT_HF_CKPT


def parse_eval_frame_indices(value: str) -> List[int]:
    raw = (value or "").strip()
    if not raw:
        return []
    if raw == "tuple_meta":
        raise ValueError("Use resolve_eval_frame_indices_for_sequence() for tuple_meta eval-frame-indices")
    indices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if any(index < 0 for index in indices):
        raise ValueError(f"eval-frame-indices must be non-negative: {indices}")
    return indices


def resolve_eval_frame_indices_for_sequence(
    eval_frame_indices_spec: Union[Sequence[int], str],
    data_root: Path,
    seq: str,
) -> List[int]:
    if isinstance(eval_frame_indices_spec, str):
        raw = eval_frame_indices_spec.strip()
        if not raw:
            return []
        if raw != "tuple_meta":
            return parse_eval_frame_indices(raw)
        meta_path = Path(data_root) / seq / "tuple_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"[{seq}] tuple_meta eval-frame-indices requested but missing: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        indices = meta.get("eval_frame_indices", [])
    else:
        indices = list(eval_frame_indices_spec)
    parsed = [int(index) for index in indices]
    if any(index < 0 for index in parsed):
        raise ValueError(f"[{seq}] eval-frame-indices must be non-negative: {parsed}")
    return parsed


def subset_depth_sequence_for_eval(
    depth_sequence: np.ndarray,
    eval_indices: Sequence[int],
    sequence_name: str,
    value_name: str,
) -> np.ndarray:
    sequence = np.asarray(depth_sequence)
    if not eval_indices:
        return sequence
    max_index = int(sequence.shape[0]) - 1
    missing = [int(index) for index in eval_indices if int(index) > max_index]
    if missing:
        raise IndexError(
            f"[{sequence_name}] eval-frame-indices out of range for {value_name} length={sequence.shape[0]}: {missing}"
        )
    return sequence[np.asarray([int(index) for index in eval_indices], dtype=np.int64)]


def load_pi3_model(ckpt: str, device: torch.device) -> Pi3:
    from easyvolcap.utils.pi3.models.pi3 import Pi3
    from aidi.scripts.baselines.pi3_checkpoint_loader import load_pi3_model_for_eval

    model, _ = load_pi3_model_for_eval(
        ckpt or DEFAULT_HF_CKPT,
        device,
        official_pi3_cls=Pi3,
        native_root=os.environ.get("PI3_NATIVE_ROOT", ""),
        model_impl=os.environ.get("PI3_MODEL_IMPL", ""),
        config_path=os.environ.get("PI3_CONFIG", ""),
    )
    return model


def load_vggt_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    from easyvolcap.models.official_vggt_model import OfficialVGGTModel

    if args.vggt_model_tag == "official":
        cfg_path = args.vggt_config or "configs/exps/vggt/vggt_official_eval_paper.yaml"
    else:
        cfg_path = args.vggt_config or "aidi/configs/vggt/finetune_5090_sparse_topk2048_l9_19_p34_record.yaml"

    cfg = load_resolved_config(cfg_path)
    raw_model_cfg = dotdict(cfg.model_cfg)
    if int(args.vggt_topk_override) > 0:
        vggt_cfg = dotdict(raw_model_cfg.get("vggt_cfg", {}))
        indexer_cfg = dotdict(vggt_cfg.get("indexer_cfg", {}))
        indexer_cfg.topk = int(args.vggt_topk_override)
        vggt_cfg.indexer_cfg = indexer_cfg
        raw_model_cfg.vggt_cfg = vggt_cfg
    if args.vggt_dtype_override:
        raw_model_cfg.dtype = str(args.vggt_dtype_override)

    model_type = raw_model_cfg.get("type", "")
    uses_registry_builder = "sampler_cfg" in raw_model_cfg or (isinstance(model_type, str) and model_type and model_type != "OfficialVGGTModel")

    if not uses_registry_builder:
        model_cfg = dotdict(raw_model_cfg)
        model_cfg.pop("type", None)
        model_cfg.pretrained_path = ""

        if args.vggt_model_tag == "official":
            ckpt_root = Path(args.vggt_official_ckpt_root)
            model_cfg.agg_ckpt = str(ckpt_root / "aggregator.pt")
            model_cfg.cam_ckpt = str(ckpt_root / "camera.pt")
            model_cfg.xyz_ckpt = str(ckpt_root / "point.pt")
            model_cfg.dpt_ckpt = str(ckpt_root / "depth.pt")
            model_cfg.tra_ckpt = str(ckpt_root / "track.pt")
            model = OfficialVGGTModel(**model_cfg).to(device=device).eval()
            return model

        model_cfg.agg_ckpt = ""
        model_cfg.cam_ckpt = ""
        model_cfg.xyz_ckpt = ""
        model_cfg.dpt_ckpt = ""
        model_cfg.tra_ckpt = ""
        model = OfficialVGGTModel(**model_cfg).to(device=device).eval()
        checkpoint = torch.load(str(Path(args.vggt_pt34_ckpt)), map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        state_dict, _ = OfficialVGGTModel._remap_special_token_keys(state_dict)
        model.load_state_dict(state_dict, strict=False)
        return model

    # Registry build path: supports DepthModel and other easyvolcap models.
    model = MODELS.build(raw_model_cfg).to(device=device).eval()
    ckpt_path = Path(args.vggt_pt34_ckpt) if args.vggt_model_tag != "official" else Path("")
    if ckpt_path and str(ckpt_path) and ckpt_path.is_file():
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    return model


def load_vggt_omega_videodepth_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if not args.vggt_omega_repo:
        raise ValueError("--vggt-omega-repo is required when --model-family=vggt_omega")
    if not args.vggt_omega_checkpoint:
        raise ValueError("--vggt-omega-checkpoint is required when --model-family=vggt_omega")
    from aidi.scripts.vggt.vggt_omega_eval_utils import load_vggt_omega_model

    model, _ = load_vggt_omega_model(
        repo_path=args.vggt_omega_repo,
        checkpoint_path=args.vggt_omega_checkpoint,
        device=device,
        global_attention_mode=args.vggt_omega_global_attention_mode,
    )
    return model


def load_depthanything3_videodepth_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    from aidi.scripts.vggt.eval_depthanything3_business_10view_badcase import load_depthanything3_model

    model, _ = load_depthanything3_model(args, device)
    return model


def set_vggt_runtime_topk(model: OfficialVGGTModel, topk: int) -> None:
    topk = int(topk)
    if topk <= 0:
        return
    model.indexer_cfg.topk = topk
    aggregator = getattr(model.vggt, "aggregator", None)
    if aggregator is not None and hasattr(aggregator, "indexer_cfg"):
        aggregator.indexer_cfg.topk = topk


def resolve_vggt_runtime_topk(args: argparse.Namespace, num_frames: int) -> int:
    if args.model_family != "vggt":
        return 0

    long_min_frames = int(args.vggt_long_topk_min_frames)
    short_topk = int(args.vggt_short_topk_override)
    long_topk = int(args.vggt_long_topk_override)
    if long_min_frames > 0 and short_topk > 0 and long_topk > 0:
        return long_topk if int(num_frames) >= long_min_frames else short_topk

    static_topk = int(args.vggt_topk_override)
    return static_topk if static_topk > 0 else 0


def list_regular_files(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.iterdir() if p.is_file() and not p.name.startswith(".")])


def parse_kitti_official_flat_image_name(file_name: str) -> Tuple[str, str, str]:
    match = KITTI_OFFICIAL_FLAT_IMAGE_RE.fullmatch(file_name)
    if match is None:
        raise ValueError(f"Unsupported KITTI official flat image name: {file_name}")
    return match.group("drive"), match.group("frame"), match.group("camera")


def build_kitti_official_flat_gt_name(drive: str, frame: str, camera: str) -> str:
    return f"{drive}_groundtruth_depth_{frame}_image_{camera}.png"


def has_kitti_official_gathered_layout(data_root: Path) -> bool:
    return has_kitti_monst3r_gathered_layout(data_root)


def group_kitti_official_flat_files(data_root: Path) -> Dict[str, List[Tuple[Path, Path]]]:
    if not has_kitti_official_flat_layout(data_root):
        raise FileNotFoundError(
            f"KITTI official flat layout requires image/ and groundtruth_depth/ under {data_root}"
        )

    image_root = data_root / "image"
    gt_root = data_root / "groundtruth_depth"
    image_files = sorted(image_root.glob("*.png"))
    gt_by_name = {path.name: path for path in sorted(gt_root.glob("*.png"))}
    grouped: Dict[str, List[Tuple[Path, Path]]] = {}
    expected_gt_names = set()

    for image_path in image_files:
        drive, frame, camera = parse_kitti_official_flat_image_name(image_path.name)
        gt_name = build_kitti_official_flat_gt_name(drive, frame, camera)
        expected_gt_names.add(gt_name)
        gt_path = gt_by_name.get(gt_name)
        if gt_path is None:
            raise FileNotFoundError(f"Missing KITTI GT for image {image_path.name} under {gt_root}")
        seq_name = f"{drive}_{camera}"
        grouped.setdefault(seq_name, []).append((image_path, gt_path))

    if not grouped:
        raise FileNotFoundError(f"No KITTI official flat PNG pairs found under {data_root}")

    for seq_name, pairs in grouped.items():
        grouped[seq_name] = sorted(pairs, key=lambda pair: parse_kitti_official_flat_image_name(pair[0].name)[1])
    return grouped


def detect_layout(spec: DatasetSpec, data_root: Path, camera: str) -> str:
    if spec.name == "sintel":
        if (data_root / "final").is_dir() and (data_root / "depth").is_dir():
            return "official"
        sample_seq = next((seq for seq in SINTEL_SEQUENCES if (data_root / seq).is_dir()), None)
        if sample_seq is not None and (data_root / sample_seq / "images").is_dir() and (data_root / sample_seq / "depths").is_dir():
            return "evc"
    elif spec.name == "bonn":
        sample_seq = next((seq for seq in BONN_SEQUENCES if (data_root / seq / "rgb_110").is_dir()), None)
        if sample_seq is not None and (data_root / sample_seq / "depth_110").is_dir():
            return "official"
    elif spec.name == "kitti":
        if has_kitti_official_gathered_layout(data_root):
            return "official"
        if has_kitti_official_flat_layout(data_root):
            return "official"
        seq_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()]) if data_root.is_dir() else []
        for seq_dir in seq_dirs:
            if (seq_dir / "images" / camera).is_dir() and (seq_dir / "depths" / camera).is_dir():
                return "evc"
    elif spec.name in ("eth3d", "7scenes", "blendedmvs", "scannetpp"):
        seq_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()]) if data_root.is_dir() else []
        for seq_dir in seq_dirs:
            if is_eth3d_evc_train_layout(seq_dir) or (spec.name == "eth3d" and is_eth3d_pi3_style_layout(seq_dir)):
                return "evc"
        if spec.name == "blendedmvs" and blendedmvs_scene_relpaths(data_root):
            return "evc"
        cam_try = resolve_camera_for_multiview_evc(data_root, camera, "eth3d")
        for seq_dir in seq_dirs:
            if (seq_dir / "images" / cam_try).is_dir() and (seq_dir / "depths" / cam_try).is_dir():
                return "evc"
    elif spec.name == "mvs_synth":
        seq_dirs = _load_scene_roots_from_data_roots_txt(data_root, verify_scene_dirs=True)
        if not seq_dirs:
            seq_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()]) if data_root.is_dir() else []
        for seq_dir in seq_dirs:
            if is_mvs_synth_evc_layout(seq_dir):
                return "evc"
        cam_try = resolve_camera_for_multiview_evc(data_root, camera, "eth3d")
        for seq_dir in seq_dirs:
            if (seq_dir / "images" / cam_try).is_dir() and (seq_dir / "depths" / cam_try).is_dir():
                return "evc"
    elif spec.name == "diode":
        if diode_val_scene_relpaths(data_root):
            return "evc"
    elif spec.name in ("vkitti2", "co3dv2"):
        cam_try = resolve_camera_for_multiview_evc(data_root, camera, spec.name)
        if nested_multiview_evc_scene_relpaths(data_root, cam_try, spec.name):
            return "evc"
    elif is_multiview_evc_dataset(spec.name) and spec.name not in (
        "kitti",
        "eth3d",
        "7scenes",
        "blendedmvs",
        "scannetpp",
        "mvs_synth",
        "vkitti2",
        "co3dv2",
        "diode",
    ):
        seq_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()]) if data_root.is_dir() else []
        cam_try = resolve_camera_for_multiview_evc(data_root, camera, spec.name)
        for seq_dir in seq_dirs:
            if (seq_dir / "images" / cam_try).is_dir() and (seq_dir / "depths" / cam_try).is_dir():
                return "evc"
    raise FileNotFoundError(
        f"Cannot detect supported layout for dataset={spec.name} under {data_root}. "
        "Expected PI3 official raw layout or EVC seq/images+depths layout."
    )


def get_sequence_names_for_layout(spec: DatasetSpec, data_root: Path, layout: str, camera: str) -> List[str]:
    if spec.sequence_names is not None:
        names = [seq for seq in spec.sequence_names if (data_root / seq).is_dir() or layout == "official"]
        if layout == "official" and spec.name == "kitti":
            names = []
    if layout == "official":
        if spec.name == "kitti":
            if has_kitti_official_gathered_layout(data_root):
                img_root = data_root / "image_gathered"
                names = sorted([p.name for p in img_root.iterdir() if p.is_dir()])
            elif has_kitti_official_flat_layout(data_root):
                names = sorted(group_kitti_official_flat_files(data_root).keys())
            else:
                raise FileNotFoundError(
                    f"Cannot find KITTI official videodepth layout under {data_root}: "
                    "expected image_gathered/groundtruth_depth_gathered or image/groundtruth_depth."
                )
        elif not spec.sequence_names:
            img_root = Path(spec.img_subdir.split("/{seq}", 1)[0])
            seq_root = data_root / img_root
            if not seq_root.is_dir():
                raise FileNotFoundError(f"Cannot list sequences under {seq_root}")
            names = sorted([p.name for p in seq_root.iterdir() if p.is_dir()])
    else:
        seq_root = data_root
        if not seq_root.is_dir():
            raise FileNotFoundError(f"Cannot list sequences under {seq_root}")
        if spec.sequence_names is None:
            names = []
        if spec.name == "diode":
            names = diode_val_scene_relpaths(data_root)
        else:
            if not names:
                names = sorted([p.name for p in seq_root.iterdir() if p.is_dir()])
            elif spec.name == "kitti":
                names = sorted([p.name for p in seq_root.iterdir() if p.is_dir()])
            else:
                names = sorted(names)
            if spec.name == "blendedmvs":
                names = blendedmvs_scene_relpaths(data_root)
            elif spec.name in ("eth3d", "7scenes", "scannetpp"):
                eth3d_vm = bool(names) and is_eth3d_evc_train_layout(data_root / names[0])
                eth3d_pi3 = spec.name == "eth3d" and bool(names) and is_eth3d_pi3_style_layout(data_root / names[0])
                if eth3d_vm or eth3d_pi3:
                    names = sorted(names)
                else:
                    names = [
                        seq for seq in names
                        if (data_root / seq / "images" / camera).is_dir() and (data_root / seq / "depths" / camera).is_dir()
                    ]
            elif spec.name == "mvs_synth":
                mvs_txt_roots = _load_scene_roots_from_data_roots_txt(data_root, verify_scene_dirs=True)
                if mvs_txt_roots:
                    names = _scene_roots_to_rel_names(data_root, mvs_txt_roots)
                else:
                    mvs_vm = [seq for seq in names if is_mvs_synth_evc_layout(data_root / seq)]
                    if mvs_vm:
                        names = mvs_vm
                    else:
                        names = [
                            seq for seq in names
                            if (data_root / seq / "images" / camera).is_dir() and (data_root / seq / "depths" / camera).is_dir()
                        ]
            elif spec.name in ("vkitti2", "co3dv2"):
                names = nested_multiview_evc_scene_relpaths(data_root, camera, spec.name)
            elif is_multiview_evc_dataset(spec.name):
                names = [
                    seq for seq in names
                    if (data_root / seq / "images" / camera).is_dir() and (data_root / seq / "depths" / camera).is_dir()
                ]
            else:
                names = [
                    seq for seq in names
                    if (data_root / seq / "images").is_dir() and (data_root / seq / "depths").is_dir()
                ]
    if not names:
        raise FileNotFoundError(f"No sequences found for dataset={spec.name}, layout={layout}, root={data_root}")
    return names


def resolve_evc_dirs(spec: DatasetSpec, data_root: Path, seq: str, camera: str) -> Tuple[Path, Path]:
    if is_multiview_evc_dataset(spec.name):
        image_dir = data_root / seq / "images" / camera
        gt_dir = data_root / seq / "depths" / camera
    else:
        sintel_candidates = [
            (data_root / seq / "images" / "final", data_root / seq / "depths" / "final"),
            (data_root / seq / "images", data_root / seq / "depths"),
        ]
        image_dir, gt_dir = next(
            ((img_dir, depth_dir) for img_dir, depth_dir in sintel_candidates if img_dir.is_dir() and depth_dir.is_dir()),
            (data_root / seq / "images", data_root / seq / "depths"),
        )
    return image_dir, gt_dir


def list_sequence_files(data_root: Path, subdir_pattern: str, seq: str, ext: str) -> List[Path]:
    subdir = data_root / subdir_pattern.format(seq=seq)
    return sorted(subdir.glob(f"*.{ext}"))


def list_sequence_files_for_layout(
    spec: DatasetSpec,
    data_root: Path,
    layout: str,
    seq: str,
    camera: str,
) -> Tuple[List[Path], List[Path]]:
    if layout == "official":
        if spec.name == "kitti" and has_kitti_official_flat_layout(data_root):
            pairs = group_kitti_official_flat_files(data_root).get(seq, [])
            image_files = [image_path for image_path, _ in pairs]
            gt_files = [gt_path for _, gt_path in pairs]
        else:
            image_files = list_sequence_files(data_root, spec.img_subdir, seq, spec.img_ext)
            gt_files = list_sequence_files(data_root, spec.gt_subdir, seq, spec.gt_ext)
    else:
        if spec.name == "diode":
            return list_diode_scan_frame_pairs(data_root / seq)
        if spec.name == "eth3d" and is_eth3d_pi3_style_layout(data_root / seq):
            return list_eth3d_pi3_style_frame_pairs(data_root / seq)
        if spec.name in ("eth3d", "co3dv2", "7scenes", "blendedmvs", "scannetpp") and is_eth3d_evc_train_layout(
            data_root / seq
        ):
            return list_eth3d_evc_train_frame_pairs(data_root / seq)
        if spec.name == "mvs_synth" and is_mvs_synth_evc_layout(data_root / seq):
            return list_mvs_synth_evc_frame_pairs(data_root / seq)
        image_dir, gt_dir = resolve_evc_dirs(spec, data_root, seq, camera)
        if is_multiview_evc_dataset(spec.name):
            image_files, gt_files = list_paired_multiview_evc_frames(image_dir, gt_dir, spec.img_ext, spec.gt_ext)
        else:
            image_files = list_regular_files(image_dir)
            gt_files = list_regular_files(gt_dir)
    return image_files, gt_files


def read_depth_file(depth_path: Path) -> np.ndarray:
    suffix = depth_path.suffix.lower()
    if suffix == ".dpt":
        return depth_read_sintel(str(depth_path))
    if suffix == ".png":
        raise ValueError("PNG depth should be decoded via dataset-specific spec.depth_reader")
    if suffix == ".exr":
        return depth_read_exr(str(depth_path))
    if suffix == ".npy":
        return np.load(depth_path).astype(np.float32)
    raise ValueError(f"Unsupported videodepth ground-truth suffix: {depth_path}")


def load_and_resize14_sequence(file_paths: Sequence[Path], new_width: int, device: torch.device) -> torch.Tensor:
    images: List[torch.Tensor] = []
    first = Image.open(file_paths[0]).convert("RGB")
    w_orig, h_orig = first.size
    target_w = new_width
    target_h = max(14, round(h_orig * (new_width / max(w_orig, 1)) / 14) * 14)
    to_tensor = tvf.ToTensor()
    for file_path in file_paths:
        img = Image.open(file_path).convert("RGB")
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        images.append(to_tensor(img))
    batch = torch.stack(images, dim=0).to(device)
    patch_h, patch_w = batch.shape[-2] // 14, batch.shape[-1] // 14
    batch = F.interpolate(
        batch,
        (patch_h * 14, patch_w * 14),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return batch.unsqueeze(0)


def _compute_aligned_hw(h: int, w: int, max_size: int, align_size: int) -> Tuple[int, int]:
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid input size for resize: H={h}, W={w}")
    if max_size <= 0:
        return int(h), int(w)
    ratio = float(max_size) / float(max(h, w))
    new_h = max(1, int(round(h * ratio)))
    new_w = max(1, int(round(w * ratio)))
    if align_size and align_size > 1:
        new_h = max(int(align_size), int(round(new_h / align_size) * align_size))
        new_w = max(int(align_size), int(round(new_w / align_size) * align_size))
    return int(new_h), int(new_w)


def preprocess_sequence_and_gt(
    image_paths: Sequence[Path],
    gt_depth: np.ndarray,
    max_size: int,
    align_size: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Resize an image sequence and corresponding GT depth sequence to the same easyvolcap-style 1036p shape.
    - image_paths: length T
    - gt_depth: (T, H, W)
    Returns:
    - images: (T, 3, H', W') float tensor in [0,1]
    - gt_depth_resized: (T, H', W') float32
    """
    first = Image.open(image_paths[0]).convert("RGB")
    w0, h0 = first.size
    h1, w1 = _compute_aligned_hw(h0, w0, max_size=max_size, align_size=align_size)
    to_tensor = tvf.ToTensor()
    frames: List[torch.Tensor] = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        if img.size != (w0, h0):
            # If shapes differ, still resize each frame to the shared target size.
            pass
        if (h1, w1) != (h0, w0):
            img = img.resize((w1, h1), Image.Resampling.LANCZOS)
        frames.append(to_tensor(img))
    images = torch.stack(frames, dim=0)  # (T,3,H,W)

    gt = gt_depth.astype(np.float32, copy=False)
    if gt.shape[1] != h1 or gt.shape[2] != w1:
        gt_resized = []
        for t in range(gt.shape[0]):
            gt_resized.append(cv2.resize(gt[t], (w1, h1), interpolation=cv2.INTER_NEAREST).astype(np.float32))
        gt = np.stack(gt_resized, axis=0)
    return images, gt


def preprocess_sequence_only(
    image_paths: Sequence[Path],
    max_size: int,
    align_size: int,
) -> torch.Tensor:
    """Same resize as preprocess_sequence_and_gt but without touching GT."""
    first = Image.open(image_paths[0]).convert("RGB")
    w0, h0 = first.size
    h1, w1 = _compute_aligned_hw(h0, w0, max_size=max_size, align_size=align_size)
    to_tensor = tvf.ToTensor()
    frames: List[torch.Tensor] = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        if (h1, w1) != (h0, w0):
            img = img.resize((w1, h1), Image.Resampling.LANCZOS)
        frames.append(to_tensor(img))
    return torch.stack(frames, dim=0)


def _mvpd_target_hw(
    h0: int,
    w0: int,
    proc_max_size: int,
    proc_align_size: int,
    *,
    co3dv2_portrait_square: bool = False,
) -> Tuple[int, int]:
    """
    MVPd-style target (H, W). Default fixes W to proc_max_size (MultiviewPointDataset-like).
    CO3Dv2 portrait: if h0>w0, use a square target so max side stays proc_max_size (e.g. 1036×1036).
    """
    if co3dv2_portrait_square and h0 > w0:
        if proc_max_size > 0:
            s = int(proc_max_size)
        else:
            s = int(max(h0, w0))
        if proc_align_size > 1:
            s = max(int(proc_align_size), int(s // proc_align_size * proc_align_size))
        return int(s), int(s)
    aspect_ratio = float(h0) / float(w0) if w0 > 0 else 1.0
    if proc_max_size > 0:
        h = int(aspect_ratio * proc_max_size)
        w = int(proc_max_size)
    else:
        h = int(aspect_ratio * max(h0, w0))
        w = int(max(h0, w0))
    ht = max(int(proc_align_size), int(h // proc_align_size * proc_align_size)) if proc_align_size > 1 else int(h)
    wt = max(int(proc_align_size), int(w // proc_align_size * proc_align_size)) if proc_align_size > 1 else int(w)
    return int(ht), int(wt)


def preprocess_sequence_and_gt_mvpd(
    image_paths: Sequence[Path],
    gt_depth: np.ndarray,
    proc_max_size: int,
    proc_align_size: int,
    safe_bound: int,
    *,
    co3dv2_portrait_square: bool = False,
) -> Tuple[torch.Tensor, np.ndarray]:
    first = Image.open(image_paths[0]).convert("RGB")
    w0, h0 = first.size
    ht, wt = _mvpd_target_hw(
        h0,
        w0,
        proc_max_size=proc_max_size,
        proc_align_size=proc_align_size,
        co3dv2_portrait_square=co3dv2_portrait_square,
    )
    safe = int(max(0, safe_bound))
    ratio = max((ht + safe) / float(h0), (wt + safe) / float(w0))
    h1 = max(1, int(h0 * ratio))
    w1 = max(1, int(w0 * ratio))
    start_h = max(0, (h1 - ht) // 2)
    start_w = max(0, (w1 - wt) // 2)

    to_tensor = tvf.ToTensor()
    frames: List[torch.Tensor] = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        img_np = np.asarray(img, dtype=np.uint8)
        if (h1, w1) != (h0, w0):
            img_np = cv2.resize(img_np, (w1, h1), interpolation=cv2.INTER_AREA)
        img_np = img_np[start_h:start_h + ht, start_w:start_w + wt, :]
        frames.append(torch.from_numpy(img_np).to(dtype=torch.float32).permute(2, 0, 1) / 255.0)
    images = torch.stack(frames, dim=0)  # (T,3,ht,wt)

    gt = gt_depth.astype(np.float32, copy=False)
    if gt.shape[1] != h1 or gt.shape[2] != w1:
        resized = []
        for t in range(gt.shape[0]):
            resized.append(cv2.resize(gt[t], (w1, h1), interpolation=cv2.INTER_NEAREST).astype(np.float32))
        gt = np.stack(resized, axis=0)
    gt = gt[:, start_h:start_h + ht, start_w:start_w + wt]
    return images, gt


def preprocess_sequence_only_mvpd(
    image_paths: Sequence[Path],
    proc_max_size: int,
    proc_align_size: int,
    safe_bound: int,
    *,
    co3dv2_portrait_square: bool = False,
) -> torch.Tensor:
    first = Image.open(image_paths[0]).convert("RGB")
    w0, h0 = first.size
    ht, wt = _mvpd_target_hw(
        h0,
        w0,
        proc_max_size=proc_max_size,
        proc_align_size=proc_align_size,
        co3dv2_portrait_square=co3dv2_portrait_square,
    )
    safe = int(max(0, safe_bound))
    ratio = max((ht + safe) / float(h0), (wt + safe) / float(w0))
    h1 = max(1, int(h0 * ratio))
    w1 = max(1, int(w0 * ratio))
    start_h = max(0, (h1 - ht) // 2)
    start_w = max(0, (w1 - wt) // 2)
    frames: List[torch.Tensor] = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        img_np = np.asarray(img, dtype=np.uint8)
        if (h1, w1) != (h0, w0):
            img_np = cv2.resize(img_np, (w1, h1), interpolation=cv2.INTER_AREA)
        img_np = img_np[start_h:start_h + ht, start_w:start_w + wt, :]
        frames.append(torch.from_numpy(img_np).to(dtype=torch.float32).permute(2, 0, 1) / 255.0)
    return torch.stack(frames, dim=0)


def resize_depth_to_1036p(depth: np.ndarray, max_size: int, align_size: int, is_gt: bool) -> np.ndarray:
    h0, w0 = int(depth.shape[-2]), int(depth.shape[-1])
    h1, w1 = _compute_aligned_hw(h0, w0, max_size=int(max_size), align_size=int(align_size))
    if (h1, w1) == (h0, w0):
        return depth.astype(np.float32, copy=False)
    interp = cv2.INTER_NEAREST if is_gt else cv2.INTER_CUBIC
    if depth.ndim == 2:
        return cv2.resize(depth.astype(np.float32), (w1, h1), interpolation=interp).astype(np.float32)
    if depth.ndim == 3:
        return np.stack(
            [cv2.resize(depth[t].astype(np.float32), (w1, h1), interpolation=interp).astype(np.float32) for t in range(depth.shape[0])],
            axis=0,
        )
    raise ValueError(f"Unexpected depth ndim: {depth.ndim}")


def vggt_omega_supported_fov_crop_box(
    *,
    image_width: int,
    image_height: int,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
) -> Tuple[int, int, int, int]:
    """Return VGGT-Omega's center-crop box as (left, top, right, bottom)."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size for VGGT-Omega crop: {image_width}x{image_height}")
    aspect_ratio = float(image_height) / float(max(image_width, 1))
    if aspect_ratio < min_aspect_ratio:
        crop_width = min(image_width, max(1, int(round(image_height / min_aspect_ratio))))
        left = max((image_width - crop_width) // 2, 0)
        return left, 0, left + crop_width, image_height
    if aspect_ratio > max_aspect_ratio:
        crop_height = min(image_height, max(1, int(round(image_width * max_aspect_ratio))))
        top = max((image_height - crop_height) // 2, 0)
        return 0, top, image_width, top + crop_height
    return 0, 0, image_width, image_height


def crop_vggt_omega_gt_to_input_fov(
    *,
    gt_depth: np.ndarray,
    image_paths: Sequence[Path],
) -> Tuple[np.ndarray, Dict[str, int]]:
    if not image_paths:
        raise ValueError("image_paths must be non-empty when cropping VGGT-Omega GT FOV")
    with Image.open(image_paths[0]) as image:
        image_width, image_height = image.size
    left, top, right, bottom = vggt_omega_supported_fov_crop_box(
        image_width=int(image_width),
        image_height=int(image_height),
    )
    gt = gt_depth.astype(np.float32, copy=False)
    if gt.ndim != 3:
        raise ValueError(f"Expected GT depth sequence shape (T,H,W), got {gt.shape}")
    gt_height, gt_width = int(gt.shape[1]), int(gt.shape[2])
    left_d = int(round(left * gt_width / max(image_width, 1)))
    right_d = int(round(right * gt_width / max(image_width, 1)))
    top_d = int(round(top * gt_height / max(image_height, 1)))
    bottom_d = int(round(bottom * gt_height / max(image_height, 1)))
    left_d = min(max(left_d, 0), gt_width - 1)
    right_d = min(max(right_d, left_d + 1), gt_width)
    top_d = min(max(top_d, 0), gt_height - 1)
    bottom_d = min(max(bottom_d, top_d + 1), gt_height)
    cropped = gt[:, top_d:bottom_d, left_d:right_d]
    crop_info = {
        "image_width": int(image_width),
        "image_height": int(image_height),
        "image_left": int(left),
        "image_top": int(top),
        "image_right": int(right),
        "image_bottom": int(bottom),
        "gt_width": int(gt_width),
        "gt_height": int(gt_height),
        "gt_left": int(left_d),
        "gt_top": int(top_d),
        "gt_right": int(right_d),
        "gt_bottom": int(bottom_d),
    }
    return cropped, crop_info


def crop_depth_sequence_with_crop_info(depth: np.ndarray, crop_info: Dict[str, int]) -> np.ndarray:
    arr = depth.astype(np.float32, copy=False)
    if arr.ndim != 3:
        raise ValueError(f"Expected depth sequence shape (T,H,W), got {arr.shape}")
    height, width = int(arr.shape[1]), int(arr.shape[2])
    source_height = int(crop_info.get("gt_height", height))
    source_width = int(crop_info.get("gt_width", width))
    left = int(round(int(crop_info["gt_left"]) * width / max(source_width, 1)))
    right = int(round(int(crop_info["gt_right"]) * width / max(source_width, 1)))
    top = int(round(int(crop_info["gt_top"]) * height / max(source_height, 1)))
    bottom = int(round(int(crop_info["gt_bottom"]) * height / max(source_height, 1)))
    left = min(max(left, 0), width - 1)
    right = min(max(right, left + 1), width)
    top = min(max(top, 0), height - 1)
    bottom = min(max(bottom, top + 1), height)
    return arr[:, top:bottom, left:right]


def resolve_eval_fov_policy(args: argparse.Namespace) -> str:
    policy = str(getattr(args, "eval_fov_policy", "auto"))
    if policy != "auto":
        return policy
    if args.model_family == "vggt_omega" and bool(args.vggt_omega_crop_gt_to_input_fov):
        return "vggt_omega_center_crop"
    return "full"


def resize_predictions_for_eval_fov(
    *,
    pred_raw: Sequence[np.ndarray],
    gt_depth: np.ndarray,
    gt_depth_full: np.ndarray,
    crop_info: Optional[Dict[str, int]],
    model_family: str,
) -> np.ndarray:
    if crop_info is None or model_family == "vggt_omega":
        return np.stack([resize_prediction(p, gt_depth.shape[1:]) for p in pred_raw], axis=0)
    pred_full = np.stack([resize_prediction(p, gt_depth_full.shape[1:]) for p in pred_raw], axis=0)
    return crop_depth_sequence_with_crop_info(pred_full, crop_info)


def infer_pi3_videodepth_from_tensor(
    model,
    images_t3hw: torch.Tensor,
    device: torch.device,
) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
    # images_t3hw: (T,3,H,W) -> (1,T,3,H,W)
    imgs = images_t3hw.to(device=device).unsqueeze(0)
    dtype = get_amp_dtype(device)
    start = time.perf_counter()
    with torch.no_grad():
        if dtype is not None:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                pred = model(imgs)
        else:
            pred = model(imgs)
    elapsed = time.perf_counter() - start
    depth_maps = pred["local_points"][0, ..., -1].detach().float().cpu().numpy()
    conf_self = pred.get("conf", None)
    if conf_self is not None:
        conf_self = conf_self[0, ..., 0].detach().float().cpu().numpy()
    return elapsed, depth_maps, conf_self


def infer_vggt_videodepth_from_tensor(
    model: torch.nn.Module,
    images_t3hw: torch.Tensor,
    device: torch.device,
) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
    # images_t3hw: (T,3,H,W)
    images = images_t3hw.to(device=device)
    height, width = int(images.shape[-2]), int(images.shape[-1])

    # Provide both representations to maximize compatibility across model implementations.
    rgb_flat = images.permute(0, 2, 3, 1).reshape(1, int(images.shape[0]), height * width, 3)  # (B=1,S=T,P,3)
    batch = dotdict(
        images=images.unsqueeze(0),  # (1,T,3,H,W)
        rgb=rgb_flat,  # (1,T,P,3)
        scale=torch.tensor([1.0], device=images.device),
        loss_scaler=torch.tensor(1.0, device=images.device),
        meta=dotdict(
            iter=torch.tensor(0, device=images.device),
            H=torch.tensor([height], device=images.device),
            W=torch.tensor([width], device=images.device),
        ),
    )

    def _is_depth_model(m: torch.nn.Module) -> bool:
        return m.__class__.__name__ == "DepthModel" and hasattr(m, "sampler")

    dtype = get_amp_dtype(device)
    start = time.perf_counter()
    with torch.no_grad():
        if _is_depth_model(model):
            sampler = model.sampler
            old_use_cam_emb = getattr(sampler, "use_cam_emb", None)
            patched_cam_emb = False
            if bool(old_use_cam_emb):
                setattr(sampler, "use_cam_emb", False)
                patched_cam_emb = True
            if not hasattr(batch, "cam"):
                batch.cam = torch.zeros((1, int(images.shape[0]), 9), device=images.device, dtype=images.dtype)
            out_batch = sampler(batch=batch)
            if patched_cam_emb:
                setattr(sampler, "use_cam_emb", old_use_cam_emb)
            output = out_batch.output
        else:
            if dtype is not None:
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    output = model(batch)
            else:
                output = model(batch)
    elapsed = time.perf_counter() - start

    if hasattr(output, "dpt_map"):
        dpt_map = output.dpt_map
    elif hasattr(output, "output") and hasattr(output.output, "dpt_map"):
        dpt_map = output.output.dpt_map
    else:
        raise ValueError("Model output does not contain dpt_map")

    depth_maps = dpt_map[0, :, :, 0].detach().float().reshape(int(images.shape[0]), height, width).cpu().numpy()
    conf_self = None
    dpt_cnf = None
    if hasattr(output, "dpt_cnf"):
        dpt_cnf = output.dpt_cnf
    elif hasattr(output, "output") and hasattr(output.output, "dpt_cnf"):
        dpt_cnf = output.output.dpt_cnf
    if dpt_cnf is not None:
        conf_self = dpt_cnf[0, :, :, 0].detach().float().reshape(int(images.shape[0]), height, width).cpu().numpy()
    return elapsed, depth_maps, conf_self


def get_amp_dtype(device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device=device)
    return torch.bfloat16 if major >= 8 else torch.float16


def infer_pi3_videodepth(model: Pi3, file_paths: Sequence[Path], load_img_size: int, device: torch.device) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
    imgs = load_and_resize14_sequence(file_paths, load_img_size, device)
    dtype = get_amp_dtype(device)
    start = time.perf_counter()
    with torch.no_grad():
        if dtype is not None:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                pred = model(imgs)
        else:
            pred = model(imgs)
    elapsed = time.perf_counter() - start
    depth_maps = pred["local_points"][0, ..., -1].detach().float().cpu().numpy()
    conf_self = pred.get("conf", None)
    if conf_self is not None:
        conf_self = conf_self[0, ..., 0].detach().float().cpu().numpy()
    return elapsed, depth_maps, conf_self


def infer_vggt_videodepth(
    model: OfficialVGGTModel,
    file_paths: Sequence[Path],
    device: torch.device,
) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
    # Official VGGT inference preprocessing (518 + divisibility by 14).
    # This matches easyvolcap/official_vggt/utils/load_fn.py behavior and avoids
    # patch_embed assertion errors (H/W must be multiples of 14).
    from easyvolcap.official_vggt.utils.load_fn import load_and_preprocess_images as load_and_preprocess_vggt_images

    images = load_and_preprocess_vggt_images([str(p) for p in file_paths])  # (T, 3, H, W), H/W divisible by 14
    return infer_vggt_videodepth_from_tensor(model, images, device)


def infer_vggt_omega_videodepth(
    model: torch.nn.Module,
    file_paths: Sequence[Path],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
    from aidi.scripts.vggt.vggt_omega_eval_utils import predict_vggt_omega, set_vggt_omega_query_view_index

    if args.vggt_omega_global_attention_mode == "query_view_image_global":
        set_vggt_omega_query_view_index(model, int(args.vggt_omega_query_view_index))

    start = time.perf_counter()
    predictions, _, _ = predict_vggt_omega(
        image_files=[str(path) for path in file_paths],
        model=model,
        repo_path=args.vggt_omega_repo,
        image_resolution=int(args.vggt_omega_resolution),
        mode=args.vggt_omega_mode,
        device=device,
    )
    elapsed = time.perf_counter() - start

    depth = predictions["depth"]
    if depth.ndim == 5:
        depth_maps = depth[0, ..., 0].detach().float().cpu().numpy()
    elif depth.ndim == 4:
        depth_maps = depth[..., 0].detach().float().cpu().numpy()
    else:
        raise ValueError(f"Unexpected VGGT-Omega depth shape: {tuple(depth.shape)}")

    conf_self = None
    depth_conf = predictions.get("depth_conf")
    if depth_conf is not None:
        if depth_conf.ndim == 4:
            conf_self = depth_conf[0].detach().float().cpu().numpy()
        elif depth_conf.ndim == 3:
            conf_self = depth_conf.detach().float().cpu().numpy()
    return elapsed, depth_maps, conf_self


def _normalize_depthanything3_depth(depth: Any) -> np.ndarray:
    if torch.is_tensor(depth):
        depth_arr = depth.detach().float().cpu().numpy()
    else:
        depth_arr = np.asarray(depth, dtype=np.float32)
    depth_arr = np.asarray(depth_arr, dtype=np.float32)

    if depth_arr.ndim == 5 and depth_arr.shape[0] == 1 and depth_arr.shape[-1] == 1:
        depth_arr = depth_arr[0, ..., 0]
    elif depth_arr.ndim == 4 and depth_arr.shape[0] == 1:
        depth_arr = depth_arr[0]
    elif depth_arr.ndim == 4 and depth_arr.shape[-1] == 1:
        depth_arr = depth_arr[..., 0]
    elif depth_arr.ndim == 4 and depth_arr.shape[1] == 1:
        depth_arr = depth_arr[:, 0]

    if depth_arr.ndim != 3:
        raise ValueError(f"Unexpected Depth Anything 3 depth shape: {tuple(depth_arr.shape)}")
    return depth_arr.astype(np.float32, copy=False)


def infer_depthanything3_videodepth(
    model: torch.nn.Module,
    file_paths: Sequence[Path],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
    start = time.perf_counter()
    with torch.no_grad():
        prediction = model.inference(
            [str(path) for path in file_paths],
            process_res=int(args.process_res),
            process_res_method=str(args.process_res_method),
            ref_view_strategy=str(args.ref_view_strategy),
            use_ray_pose=bool(args.use_ray_pose),
            align_to_input_ext_scale=bool(args.align_to_input_ext_scale),
        )
    elapsed = time.perf_counter() - start
    return elapsed, _normalize_depthanything3_depth(prediction.depth), None


def write_prediction_png(depth: np.ndarray, png_path: Path) -> None:
    normalized = depth - depth.min()
    denom = float(normalized.max())
    if denom > 0:
        normalized = normalized / denom
    png = (normalized * 255.0).astype(np.uint8)
    Image.fromarray(png).save(png_path)


def write_depth_vis_png(depth: np.ndarray, png_path: Path) -> None:
    """Robust depth visualization for qualitative checks (handles invalid <=0)."""
    depth = depth.astype(np.float32, copy=False)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        Image.fromarray(np.zeros(depth.shape[:2], dtype=np.uint8)).save(png_path)
        return
    d = depth[valid]
    lo = float(np.percentile(d, 1.0))
    hi = float(np.percentile(d, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(d))
        hi = float(np.max(d))
        if hi <= lo:
            Image.fromarray(np.zeros(depth.shape[:2], dtype=np.uint8)).save(png_path)
            return
    scaled = (np.clip(depth, lo, hi) - lo) / (hi - lo)
    scaled[~valid] = 0.0
    Image.fromarray((scaled * 255.0).astype(np.uint8)).save(png_path)


def write_rgb_png(rgb: np.ndarray, png_path: Path) -> None:
    """Save an RGB image. Accepts HWC uint8 or float in [0,1]."""
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB HWC, got shape={arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(png_path)


def save_sequence_predictions(
    depth_maps: np.ndarray,
    seq_pred_dir: Path,
    conf_self: Optional[np.ndarray],
    save_png: bool,
) -> None:
    seq_pred_dir.mkdir(parents=True, exist_ok=True)
    for idx, depth in enumerate(depth_maps):
        stem = f"frame_{idx:04d}"
        np.save(seq_pred_dir / f"{stem}.npy", depth)
        if save_png:
            write_prediction_png(depth, seq_pred_dir / f"{stem}.png")
            if conf_self is not None:
                write_prediction_png(np.log(np.clip(conf_self[idx], a_min=1e-8, a_max=None)), seq_pred_dir / f"{stem}_conf.png")


def clear_sequence_predictions(seq_pred_dir: Path) -> None:
    if not seq_pred_dir.is_dir():
        return
    for path in seq_pred_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".npy", ".png", ".json"}:
            path.unlink()


def resize_prediction(prediction: np.ndarray, shape_hw: Sequence[int]) -> np.ndarray:
    height, width = int(shape_hw[0]), int(shape_hw[1])
    return cv2.resize(prediction.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)


def absolute_value_scaling2(
    predicted_depth: torch.Tensor,
    ground_truth_depth: torch.Tensor,
    s_init: float,
    t_init: float = 0.0,
    lr: float = 1e-4,
    max_iters: int = 1000,
    tol: float = 1e-6,
) -> Tuple[float, float]:
    s = torch.tensor([s_init], requires_grad=True, device=predicted_depth.device, dtype=predicted_depth.dtype)
    t = torch.tensor([t_init], requires_grad=True, device=predicted_depth.device, dtype=predicted_depth.dtype)
    optimizer = torch.optim.Adam([s, t], lr=lr)
    prev_loss = None

    for _ in range(max_iters):
        optimizer.zero_grad()
        predicted_aligned = s * predicted_depth + t
        loss = torch.sum(torch.abs(predicted_aligned - ground_truth_depth))
        loss.backward()
        optimizer.step()
        if prev_loss is not None and abs(prev_loss - loss.item()) < tol:
            break
        prev_loss = loss.item()

    return float(s.detach().item()), float(t.detach().item())


def align_with_scale(predicted_depth: torch.Tensor, ground_truth_depth: torch.Tensor) -> torch.Tensor:
    scale = torch.nanmean(ground_truth_depth) / torch.nanmean(predicted_depth)
    for _ in range(10):
        residuals = scale * predicted_depth - ground_truth_depth
        weights = 1.0 / (residuals.abs() + 1e-8)
        weighted_dot_pred_gt = torch.sum(weights * predicted_depth * ground_truth_depth)
        weighted_dot_pred_pred = torch.sum(weights * predicted_depth**2)
        scale = weighted_dot_pred_gt / weighted_dot_pred_pred
    scale = scale.clamp(min=1e-3).detach()
    return predicted_depth * scale


def subsample_depth_fit_pixels(
    predicted_depth: torch.Tensor,
    ground_truth_depth: torch.Tensor,
    max_pixels: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_pixels = int(max_pixels)
    if max_pixels <= 0 or predicted_depth.numel() <= max_pixels:
        return predicted_depth, ground_truth_depth
    step = int(math.ceil(predicted_depth.numel() / max_pixels))
    indices = torch.arange(0, predicted_depth.numel(), step, device=predicted_depth.device)[:max_pixels]
    return predicted_depth.index_select(0, indices), ground_truth_depth.index_select(0, indices)


def evaluate_depth_sequence(
    predicted_depth_original: np.ndarray,
    ground_truth_depth_original: np.ndarray,
    alignment: str,
    max_depth: Optional[float],
    post_clip_max: Optional[float],
    device: torch.device,
    min_depth: Optional[float] = None,
    scale_shift_fit_max_pixels: int = 0,
) -> Dict[str, float]:
    predicted_depth_original = torch.from_numpy(predicted_depth_original).to(device=device, dtype=torch.float32)
    ground_truth_depth_original = torch.from_numpy(ground_truth_depth_original).to(device=device, dtype=torch.float32)

    if predicted_depth_original.dim() == 3:
        _, _, width = predicted_depth_original.shape
        # Use reshape instead of view because inputs may be non-contiguous
        # (e.g. produced by slicing/cropping before stacking).
        predicted_depth_original = predicted_depth_original.reshape(-1, width)
        ground_truth_depth_original = ground_truth_depth_original.reshape(-1, width)

    mask = ground_truth_depth_original > 0
    if min_depth is not None:
        mask = mask & (ground_truth_depth_original >= min_depth)
    if max_depth is not None:
        mask = mask & (ground_truth_depth_original < max_depth)

    predicted_depth = predicted_depth_original[mask]
    ground_truth_depth = ground_truth_depth_original[mask]

    if predicted_depth.numel() == 0:
        return {
            "Abs Rel": 0.0,
            "Sq Rel": 0.0,
            "RMSE": 0.0,
            "Log RMSE": 0.0,
            "δ < 1.": 0.0,
            "δ < 1.25": 0.0,
            "δ < 1.25^2": 0.0,
            "δ < 1.25^3": 0.0,
            "valid_pixels": 0.0,
        }

    if alignment == "scale&shift":
        fit_predicted_depth, fit_ground_truth_depth = subsample_depth_fit_pixels(
            predicted_depth=predicted_depth,
            ground_truth_depth=ground_truth_depth,
            max_pixels=scale_shift_fit_max_pixels,
        )
        s_init = (torch.median(fit_ground_truth_depth) / torch.median(fit_predicted_depth)).item()
        s, t = absolute_value_scaling2(fit_predicted_depth, fit_ground_truth_depth, s_init=s_init)
        predicted_depth = s * predicted_depth + t
    elif alignment == "scale":
        predicted_depth = align_with_scale(predicted_depth, ground_truth_depth)
    elif alignment == "metric":
        predicted_depth = predicted_depth
    else:
        raise ValueError(f"Unknown alignment method: {alignment}")

    if post_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=post_clip_max)

    abs_rel = torch.mean(torch.abs(predicted_depth - ground_truth_depth) / ground_truth_depth).item()
    sq_rel = torch.mean(((predicted_depth - ground_truth_depth) ** 2) / ground_truth_depth).item()
    rmse = torch.sqrt(torch.mean((predicted_depth - ground_truth_depth) ** 2)).item()

    predicted_depth = torch.clamp(predicted_depth, min=1e-5)
    log_rmse = torch.sqrt(torch.mean((torch.log(predicted_depth) - torch.log(ground_truth_depth)) ** 2)).item()

    max_ratio = torch.maximum(predicted_depth / ground_truth_depth, ground_truth_depth / predicted_depth)
    threshold_0 = torch.mean((max_ratio < 1.0).float()).item()
    threshold_1 = torch.mean((max_ratio < 1.25).float()).item()
    threshold_2 = torch.mean((max_ratio < 1.25**2).float()).item()
    threshold_3 = torch.mean((max_ratio < 1.25**3).float()).item()

    return {
        "Abs Rel": abs_rel,
        "Sq Rel": sq_rel,
        "RMSE": rmse,
        "Log RMSE": log_rmse,
        "δ < 1.": threshold_0,
        "δ < 1.25": threshold_1,
        "δ < 1.25^2": threshold_2,
        "δ < 1.25^3": threshold_3,
        "valid_pixels": float(mask.sum().item()),
    }


def weighted_average(metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        raise ValueError("Cannot average empty metrics list.")
    weights = np.array([m["valid_pixels"] for m in metrics], dtype=np.float64)
    result: Dict[str, float] = {}
    numeric_keys = [
        key
        for key, value in metrics[0].items()
        if isinstance(value, (int, float, np.floating))
    ]
    for key in numeric_keys:
        if key == "valid_pixels":
            result[key] = float(weights.sum())
            continue
        values = np.array([m[key] for m in metrics], dtype=np.float64)
        if weights.sum() > 0:
            result[key] = float(np.average(values, weights=weights))
        else:
            result[key] = 0.0
    return result


def save_csv(csv_path: Path, metrics: Dict[str, float]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(metrics.keys()))
        writer.writerow([metrics[k] for k in metrics.keys()])


def resolve_paper_target(model_family: str, vggt_model_tag: str, dataset: str) -> Dict[str, float]:
    if model_family == "pi3":
        return dict(VIDEO_PAPER_TARGETS.get("pi3", {}).get(dataset, {}))
    if model_family == "vggt" and vggt_model_tag == "official":
        return dict(VIDEO_PAPER_TARGETS.get("vggt", {}).get(dataset, {}))
    return {}


def main() -> None:
    args = parse_args()
    spec = DATASET_SPECS[args.dataset]
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir) if args.output_dir else resolve_default_output_dir(args.dataset)
    pred_root = Path(args.pred_root) if args.pred_root else output_dir / "predictions" / args.dataset
    progress_path = output_dir / f"{args.dataset}_videodepth_protocol_progress.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)

    if args.layout == "official" and spec.name in {
        "eth3d",
        "7scenes",
        "blendedmvs",
        "scannetpp",
        "mvs_synth",
        "vkitti2",
        "co3dv2",
        "diode",
    }:
        raise ValueError(
            f"Dataset {spec.name} is only supported in EVC layout under --data-root "
            "(view-major: <scene>/images/<view##>/000000.(jpg|png) + depths/<view##>/000000.exr; "
            "mvs_synth uses 4-digit view dirs; eth3d/7scenes/blendedmvs/scannetpp use 6-digit; "
            "flat multiview: <scene>/images/<camera>/; "
            "DIODE val: indoors|outdoor/scene_*/scan_* with png + *_depth.npy). Use --layout evc or --layout auto."
        )

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    eval_device = torch.device(args.eval_device) if args.eval_device else device
    model: Optional[object] = None
    if args.model_family == "pi3":
        ckpt = args.ckpt or resolve_default_ckpt()
    elif args.model_family == "vggt_omega":
        ckpt = str(Path(args.vggt_omega_checkpoint).expanduser().resolve()) if args.vggt_omega_checkpoint else ""
    elif args.model_family == "depthanything3":
        ckpt = str(Path(args.da3_model).expanduser().resolve()) if args.da3_model else ""
    else:
        ckpt = ""
    if not args.eval_only:
        if args.model_family == "pi3":
            model = load_pi3_model(ckpt, device)
        elif args.model_family == "vggt_omega":
            model = load_vggt_omega_videodepth_model(args, device)
        elif args.model_family == "depthanything3":
            model = load_depthanything3_videodepth_model(args, device)
        else:
            model = load_vggt_model(args, device)

    layout = args.layout if args.layout != "auto" else detect_layout(spec, data_root, args.camera)
    log_progress(f"Using dataset layout: {layout}")
    if layout == "evc" and (
        is_multiview_evc_dataset(spec.name) or spec.name in ("7scenes", "blendedmvs", "mvs_synth", "scannetpp")
    ):
        probe_seqs = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
        eth3d_view_major = (
            spec.name in ("eth3d", "7scenes", "blendedmvs", "scannetpp")
            and bool(probe_seqs)
            and is_eth3d_evc_train_layout(data_root / probe_seqs[0])
        )
        mvs_synth_view_major = (
            spec.name == "mvs_synth"
            and bool(probe_seqs)
            and is_mvs_synth_evc_layout(data_root / probe_seqs[0])
        )
        blendedmvs_nested_vm = spec.name == "blendedmvs" and bool(blendedmvs_scene_relpaths(data_root))
        co3d_view_major = False
        if spec.name == "co3dv2":
            roots_vm = evc_scene_roots_discover(data_root, "co3dv2")
            co3d_view_major = bool(roots_vm) and is_eth3d_evc_train_layout(roots_vm[0])
        if eth3d_view_major or mvs_synth_view_major or blendedmvs_nested_vm:
            log_progress(
                "EVC view-major: each images/<view##>/ is one temporal frame (`--camera` ignored); "
                "6-digit views (eth3d/7scenes/blendedmvs/scannetpp) or 4-digit (mvs_synth). "
                "BlendedMVS may use <root>/<subset>/<scene> paths."
            )
        elif co3d_view_major:
            log_progress(
                "CO3Dv2 EVC (co3d_to_easyvolcap layout): each images/<view06d>/ is one frame in the videodepth stack; "
                "`--camera` does not subset views."
            )
            resolved_cam = resolve_camera_for_multiview_evc(data_root, args.camera, args.dataset)
            if resolved_cam != args.camera:
                log_progress(f"Multiview EVC (probe): using camera directory '{resolved_cam}' (--camera was '{args.camera}')")
            args.camera = resolved_cam
        else:
            resolved_cam = resolve_camera_for_multiview_evc(data_root, args.camera, args.dataset)
            if resolved_cam != args.camera:
                log_progress(f"Multiview EVC: using camera directory '{resolved_cam}' (--camera was '{args.camera}')")
            args.camera = resolved_cam

    log_progress(f"[{args.dataset}] discovering sequences under {data_root}")
    sequence_names = get_sequence_names_for_layout(spec, data_root, layout, args.camera)
    if args.max_seqs > 0:
        sequence_names = sequence_names[: args.max_seqs]
    log_progress(
        f"[{args.dataset}] total_sequences={len(sequence_names)} "
        f"output_dir={output_dir} progress={progress_path}"
    )
    append_progress_jsonl(
        progress_path,
        {
            "event": "run_start",
            "dataset": args.dataset,
            "model_family": args.model_family,
            "layout": layout,
            "data_root": str(data_root),
            "output_dir": str(output_dir),
            "num_sequences": len(sequence_names),
            "time": datetime.now().isoformat(timespec="seconds"),
        },
    )

    # KITTI: metric eval at native GT depth resolution (no GT resize/interpolation; preds resized to GT H×W).
    kitti_native_gt_eval = spec.name == "kitti"
    if kitti_native_gt_eval:
        if args.eval_gt_shape != "official" or args.eval_resize_to_1036p:
            log_progress(
                "KITTI: metrics use native GT depth (unchanged from disk); predictions are resized to GT shape. "
                "Overriding --eval-gt-shape / --eval-resize-to-1036p."
            )
    eval_gt_shape = "official" if kitti_native_gt_eval else args.eval_gt_shape
    eval_resize_to_1036p = False if kitti_native_gt_eval else bool(args.eval_resize_to_1036p)
    eval_frame_indices_spec: Union[List[int], str]
    if str(args.eval_frame_indices).strip() == "tuple_meta":
        eval_frame_indices_spec = "tuple_meta"
    else:
        eval_frame_indices_spec = parse_eval_frame_indices(args.eval_frame_indices)
    if args.vggt_omega_query_view_sweep:
        if args.model_family != "vggt_omega":
            raise ValueError("--vggt-omega-query-view-sweep is only valid with --model-family=vggt_omega")
        if args.vggt_omega_global_attention_mode != "query_view_image_global":
            raise ValueError(
                "--vggt-omega-query-view-sweep requires "
                "--vggt-omega-global-attention-mode=query_view_image_global"
            )
        if eval_frame_indices_spec:
            raise ValueError("--eval-frame-indices is not compatible with --vggt-omega-query-view-sweep")
    eval_fov_policy_effective = resolve_eval_fov_policy(args)

    dataset_seq_metrics: List[Dict[str, float]] = []
    sequence_summaries: Dict[str, Dict[str, float]] = {}
    sequence_eval_frame_indices: Dict[str, List[int]] = {}
    query_view_depth_metrics: List[Dict[str, object]] = []
    sequence_runtime_topk: Dict[str, int] = {}
    vggt_omega_crop_boxes: Dict[str, Dict[str, int]] = {}
    total_infer_seconds = 0.0
    all_fps: List[float] = []
    infer_time_per_seq: List[float] = []
    current_vggt_runtime_topk: Optional[int] = None

    for seq_index, seq in enumerate(sequence_names, start=1):
        seq_wall_start = time.perf_counter()
        image_files, gt_files = list_sequence_files_for_layout(spec, data_root, layout, seq, args.camera)
        if args.max_frames_per_seq > 0:
            image_files = image_files[: args.max_frames_per_seq]
            gt_files = gt_files[: args.max_frames_per_seq]
        if not image_files or not gt_files:
            raise FileNotFoundError(
                f"[{seq}] empty image/gt files under dataset={args.dataset}, layout={layout}, root={data_root}"
            )
        if len(image_files) != len(gt_files):
            raise ValueError(
                f"[{seq}] image/gt count mismatch: {len(image_files)} vs {len(gt_files)} under {data_root} (layout={layout})"
            )

        seq_runtime_topk = resolve_vggt_runtime_topk(args, len(image_files))
        if args.model_family == "vggt" and model is not None and seq_runtime_topk > 0 and seq_runtime_topk != current_vggt_runtime_topk:
            set_vggt_runtime_topk(model, seq_runtime_topk)
            current_vggt_runtime_topk = seq_runtime_topk
        sequence_runtime_topk[seq] = int(seq_runtime_topk)
        topk_suffix = f" vggt_topk={seq_runtime_topk}" if args.model_family == "vggt" and seq_runtime_topk > 0 else ""
        seq_pred_dir = pred_root / seq
        pred_npy_files = sorted(seq_pred_dir.glob("*.npy")) if seq_pred_dir.is_dir() else []
        time_json = seq_pred_dir / "_time.json"
        need_infer = args.overwrite or not time_json.is_file() or len(pred_npy_files) != len(image_files)
        reused_predictions = not need_infer
        log_progress(
            f"[{args.dataset}] start {seq_index}/{len(sequence_names)} sequence={seq} "
            f"frames={len(image_files)} reuse_pred={reused_predictions}{topk_suffix}"
        )

        if args.vggt_omega_query_view_sweep:
            if model is None:
                raise RuntimeError("Model is not initialized while query-view sweep inference is required.")
            sequence_eval_frame_indices[seq] = list(range(len(image_files)))
            gt_depth_raw = stack_gt_depth_sequence(spec, image_files, gt_files, data_root, seq, args.camera)
            if args.model_family == "vggt_omega" and bool(args.vggt_omega_crop_gt_to_input_fov):
                gt_depth_raw, crop_info = crop_vggt_omega_gt_to_input_fov(
                    gt_depth=gt_depth_raw,
                    image_paths=image_files,
                )
                vggt_omega_crop_boxes[seq] = crop_info
            if eval_resize_to_1036p:
                gt_depth = resize_depth_to_1036p(
                    gt_depth_raw,
                    max_size=int(args.max_size),
                    align_size=int(args.align_size),
                    is_gt=True,
                )
            elif eval_gt_shape == "official":
                gt_depth = gt_depth_raw
            elif args.preprocess_style == "mvpd":
                _, gt_depth = preprocess_sequence_and_gt_mvpd(
                    image_paths=image_files,
                    gt_depth=gt_depth_raw,
                    proc_max_size=int(args.max_size),
                    proc_align_size=int(args.align_size),
                    safe_bound=int(args.safe_bound),
                    co3dv2_portrait_square=(args.dataset == "co3dv2"),
                )
            else:
                _, gt_depth = preprocess_sequence_and_gt(
                    image_paths=image_files,
                    gt_depth=gt_depth_raw,
                    max_size=int(args.max_size),
                    align_size=int(args.align_size),
                )

            for query_view_index in range(len(image_files)):
                query_wall_start = time.perf_counter()
                args.vggt_omega_query_view_index = int(query_view_index)
                q_pred_dir = pred_root / seq / f"query_view_{query_view_index:04d}"
                q_pred_path = q_pred_dir / f"query_view_{query_view_index:04d}.npy"
                time_json = q_pred_dir / "_time.json"
                need_query_infer = args.overwrite or not time_json.is_file() or not q_pred_path.is_file()
                if need_query_infer:
                    clear_sequence_predictions(q_pred_dir)
                    infer_time, depth_maps, conf_self = infer_vggt_omega_videodepth(model, image_files, args, device)
                    selected_depth = depth_maps[query_view_index].astype(np.float32)
                    q_pred_dir.mkdir(parents=True, exist_ok=True)
                    np.save(q_pred_path, selected_depth)
                    if args.save_png:
                        write_prediction_png(selected_depth, q_pred_dir / f"query_view_{query_view_index:04d}.png")
                        if conf_self is not None:
                            write_prediction_png(
                                np.log(np.clip(conf_self[query_view_index], a_min=1e-8, a_max=None)),
                                q_pred_dir / f"query_view_{query_view_index:04d}_conf.png",
                            )
                    with open(time_json, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "time": float(infer_time),
                                "input_frames": len(image_files),
                                "query_view_index": int(query_view_index),
                            },
                            f,
                            indent=2,
                        )
                else:
                    selected_depth = np.load(q_pred_path).astype(np.float32)
                    with open(time_json, "r", encoding="utf-8") as f:
                        timing_data = json.load(f)
                    infer_time = float(timing_data["time"])

                total_infer_seconds += float(infer_time) if need_query_infer else 0.0
                if eval_resize_to_1036p:
                    pr_depth_eval = resize_depth_to_1036p(
                        selected_depth,
                        max_size=int(args.max_size),
                        align_size=int(args.align_size),
                        is_gt=False,
                    )[None]
                else:
                    pr_depth_eval = resize_prediction(selected_depth, gt_depth.shape[1:])[None]
                gt_depth_eval = gt_depth[query_view_index : query_view_index + 1]

                q_metrics = evaluate_depth_sequence(
                    predicted_depth_original=pr_depth_eval,
                    ground_truth_depth_original=gt_depth_eval,
                    alignment=args.alignment,
                    max_depth=spec.max_depth,
                    post_clip_max=spec.post_clip_max,
                    device=eval_device,
                    min_depth=spec.min_depth,
                    scale_shift_fit_max_pixels=int(args.scale_shift_fit_max_pixels),
                )
                q_metrics.update(
                    {
                        "fps": float(1.0 / max(infer_time, 1e-8)),
                        "infer_time": float(infer_time),
                        "frames": 1.0,
                        "eval_frames": 1.0,
                        "input_frames": float(len(image_files)),
                    }
                )
                dataset_seq_metrics.append(q_metrics)
                all_fps.append(q_metrics["fps"])
                infer_time_per_seq.append(q_metrics["infer_time"])

                q_record: Dict[str, object] = dict(q_metrics)
                q_record.update(
                    {
                        "sequence": seq,
                        "query_view_index": int(query_view_index),
                        "image_name": image_files[query_view_index].name,
                        "reused_predictions": not need_query_infer,
                    }
                )
                query_view_depth_metrics.append(q_record)
                sequence_summaries[f"{seq}/query_view_{query_view_index:04d}"] = dict(q_record)
                append_progress_jsonl(
                    progress_path,
                    {
                        "event": "query_view_done",
                        "dataset": args.dataset,
                        "sequence": seq,
                        "sequence_index": seq_index,
                        "query_view_index": int(query_view_index),
                        "num_query_views": len(image_files),
                        "reused_predictions": not need_query_infer,
                        "infer_time": float(infer_time),
                        "wall_seconds": float(time.perf_counter() - query_wall_start),
                        "time": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                log_progress(
                    f"[{args.dataset}] query-view {seq_index}/{len(sequence_names)} sequence={seq} "
                    f"view={query_view_index + 1}/{len(image_files)} reuse={not need_query_infer} "
                    f"infer={infer_time:.3f}s AbsRel={q_metrics['Abs Rel']:.6f}"
                )

            wall_seconds = time.perf_counter() - seq_wall_start
            append_progress_jsonl(
                progress_path,
                {
                    "event": "sequence_done",
                    "dataset": args.dataset,
                    "sequence": seq,
                    "sequence_index": seq_index,
                    "num_sequences": len(sequence_names),
                    "frames": len(image_files),
                    "eval_frames": len(image_files),
                    "query_view_sweep": True,
                    "wall_seconds": float(wall_seconds),
                    "time": datetime.now().isoformat(timespec="seconds"),
                },
            )
            log_progress(
                f"[{args.dataset}] done {seq_index}/{len(sequence_names)} sequence={seq} "
                f"query_views={len(image_files)} wall={wall_seconds:.3f}s"
            )
            continue

        if need_infer:
            if model is None:
                raise RuntimeError("Model is not initialized while inference is required.")
            clear_sequence_predictions(seq_pred_dir)
            if args.model_family == "vggt_omega":
                infer_time, depth_maps, conf_self = infer_vggt_omega_videodepth(model, image_files, args, device)
                total_infer_seconds += infer_time
                save_sequence_predictions(depth_maps, seq_pred_dir, conf_self, args.save_png)
                with open(time_json, "w", encoding="utf-8") as f:
                    json.dump({"time": infer_time, "frames": len(image_files)}, f, indent=2)
                pred_npy_files = sorted(seq_pred_dir.glob("*.npy"))
                need_infer = False
            elif args.model_family == "depthanything3":
                infer_time, depth_maps, conf_self = infer_depthanything3_videodepth(model, image_files, args, device)
                total_infer_seconds += infer_time
                save_sequence_predictions(depth_maps, seq_pred_dir, conf_self, args.save_png)
                with open(time_json, "w", encoding="utf-8") as f:
                    json.dump({"time": infer_time, "frames": len(image_files)}, f, indent=2)
                pred_npy_files = sorted(seq_pred_dir.glob("*.npy"))
                need_infer = False
            # Keep the requested VGGT input preprocessing independent from the checkpoint source.
            if args.vggt_input_preprocess == "official" and args.model_family == "vggt":
                # Keep official VGGT preprocessing for inference (518/crop). Reuse legacy helper.
                if args.model_family == "pi3":
                    # PI3 path doesn't support this mode; fall back to current preprocessing.
                    pass
                else:
                    infer_time, depth_maps, conf_self = infer_vggt_videodepth(model, image_files, device)
                    total_infer_seconds += infer_time
                    save_sequence_predictions(depth_maps, seq_pred_dir, conf_self, args.save_png)
                    with open(time_json, "w", encoding="utf-8") as f:
                        json.dump({"time": infer_time, "frames": len(image_files)}, f, indent=2)
                    pred_npy_files = sorted(seq_pred_dir.glob("*.npy"))
                    need_infer = False

            if need_infer and eval_gt_shape == "official":
                if args.preprocess_style == "mvpd":
                    images_t3hw = preprocess_sequence_only_mvpd(
                        image_paths=image_files,
                        proc_max_size=int(args.max_size),
                        proc_align_size=int(args.align_size),
                        safe_bound=int(args.safe_bound),
                        co3dv2_portrait_square=(args.dataset == "co3dv2"),
                    )
                else:
                    images_t3hw = preprocess_sequence_only(
                        image_paths=image_files,
                        max_size=int(args.max_size),
                        align_size=int(args.align_size),
                    )
            else:
                gt_depth_raw_tmp = stack_gt_depth_sequence(
                    spec, image_files, gt_files, data_root, seq, args.camera
                )
                if args.preprocess_style == "mvpd":
                    images_t3hw, _ = preprocess_sequence_and_gt_mvpd(
                        image_paths=image_files,
                        gt_depth=gt_depth_raw_tmp,
                        proc_max_size=int(args.max_size),
                        proc_align_size=int(args.align_size),
                        safe_bound=int(args.safe_bound),
                        co3dv2_portrait_square=(args.dataset == "co3dv2"),
                    )
                else:
                    images_t3hw, _ = preprocess_sequence_and_gt(
                        image_paths=image_files,
                        gt_depth=gt_depth_raw_tmp,
                        max_size=int(args.max_size),
                        align_size=int(args.align_size),
                    )

            if need_infer:
                if args.model_family == "pi3":
                    infer_time, depth_maps, conf_self = infer_pi3_videodepth_from_tensor(model, images_t3hw, device)
                else:
                    infer_time, depth_maps, conf_self = infer_vggt_videodepth_from_tensor(model, images_t3hw, device)
                total_infer_seconds += infer_time
                save_sequence_predictions(depth_maps, seq_pred_dir, conf_self, args.save_png)
                with open(time_json, "w", encoding="utf-8") as f:
                    json.dump({"time": infer_time, "frames": len(image_files)}, f, indent=2)
                pred_npy_files = sorted(seq_pred_dir.glob("*.npy"))
        else:
            with open(time_json, "r", encoding="utf-8") as f:
                timing_data = json.load(f)
            infer_time = float(sum(timing_data["time"])) if isinstance(timing_data["time"], list) else float(timing_data["time"])

        gt_depth_raw_full = stack_gt_depth_sequence(spec, image_files, gt_files, data_root, seq, args.camera)
        gt_depth_raw = gt_depth_raw_full
        eval_fov_crop_info: Optional[Dict[str, int]] = None
        if eval_fov_policy_effective == "vggt_omega_center_crop":
            gt_depth_raw, eval_fov_crop_info = crop_vggt_omega_gt_to_input_fov(
                gt_depth=gt_depth_raw_full,
                image_paths=image_files,
            )
            vggt_omega_crop_boxes[seq] = eval_fov_crop_info
        if eval_resize_to_1036p:
            gt_depth = resize_depth_to_1036p(gt_depth_raw, max_size=int(args.max_size), align_size=int(args.align_size), is_gt=True)
        else:
            if eval_gt_shape == "official":
                gt_depth = gt_depth_raw
            else:
                # Preprocess GT to match 1036p evaluation space.
                if args.preprocess_style == "mvpd":
                    _, gt_depth = preprocess_sequence_and_gt_mvpd(
                        image_paths=image_files,
                        gt_depth=gt_depth_raw,
                        proc_max_size=int(args.max_size),
                        proc_align_size=int(args.align_size),
                        safe_bound=int(args.safe_bound),
                        co3dv2_portrait_square=(args.dataset == "co3dv2"),
                    )
                else:
                    _, gt_depth = preprocess_sequence_and_gt(
                        image_paths=image_files,
                        gt_depth=gt_depth_raw,
                        max_size=int(args.max_size),
                        align_size=int(args.align_size),
                    )
        # Load predictions and align to GT evaluation shape for metrics.
        pr_raw = [np.load(pd_path).astype(np.float32) for pd_path in pred_npy_files]
        if eval_resize_to_1036p:
            pr_depth = np.stack(
                [resize_depth_to_1036p(p, max_size=int(args.max_size), align_size=int(args.align_size), is_gt=False) for p in pr_raw],
                axis=0,
            )
        else:
            pr_depth = resize_predictions_for_eval_fov(
                pred_raw=pr_raw,
                gt_depth=gt_depth,
                gt_depth_full=gt_depth_raw_full,
                crop_info=eval_fov_crop_info if eval_gt_shape == "official" else None,
                model_family=args.model_family,
            )
        eval_frame_indices = resolve_eval_frame_indices_for_sequence(
            eval_frame_indices_spec=eval_frame_indices_spec,
            data_root=data_root,
            seq=seq,
        )
        sequence_eval_frame_indices[seq] = [int(index) for index in eval_frame_indices]
        pr_depth_eval = subset_depth_sequence_for_eval(pr_depth, eval_frame_indices, seq, "prediction")
        gt_depth_eval = subset_depth_sequence_for_eval(gt_depth, eval_frame_indices, seq, "ground truth")

        # Optional qualitative visualization in the evaluation space (e.g. 1036p).
        if args.save_gt_png or args.save_aligned_pred_png or args.save_image_png:
            vis_limit = int(args.vis_max_frames)
            n_vis = len(image_files) if vis_limit <= 0 else min(len(image_files), vis_limit)
            for idx in range(n_vis):
                stem = f"frame_{idx:04d}"
                if args.save_image_png:
                    # Save RGB in the same space as GT used for evaluation when possible.
                    if eval_resize_to_1036p:
                        img = np.asarray(Image.open(image_files[idx]).convert("RGB"), dtype=np.uint8)
                        img = resize_depth_to_1036p(img, max_size=int(args.max_size), align_size=int(args.align_size), is_gt=True)  # nearest for image-like
                        write_rgb_png(img, (seq_pred_dir / f"{stem}_rgb.png"))
                    elif eval_gt_shape == "official":
                        img = np.asarray(Image.open(image_files[idx]).convert("RGB"), dtype=np.uint8)
                        write_rgb_png(img, (seq_pred_dir / f"{stem}_rgb.png"))
                    else:
                        # preprocessed GT space
                        if args.preprocess_style == "mvpd":
                            img_t = preprocess_sequence_only_mvpd(
                                image_paths=[image_files[idx]],
                                proc_max_size=int(args.max_size),
                                proc_align_size=int(args.align_size),
                                safe_bound=int(args.safe_bound),
                                co3dv2_portrait_square=(args.dataset == "co3dv2"),
                            )[0]  # (3,H,W) float
                        else:
                            img_t = preprocess_sequence_only(
                                image_paths=[image_files[idx]],
                                max_size=int(args.max_size),
                                align_size=int(args.align_size),
                            )[0]
                        img = img_t.permute(1, 2, 0).detach().cpu().numpy()
                        write_rgb_png(img, (seq_pred_dir / f"{stem}_rgb.png"))
                if args.save_gt_png:
                    write_depth_vis_png(gt_depth[idx], (seq_pred_dir / f"{stem}_gt.png"))
                if args.save_aligned_pred_png:
                    write_depth_vis_png(pr_depth[idx], (seq_pred_dir / f"{stem}_pred_aligned.png"))
        seq_metrics = evaluate_depth_sequence(
            predicted_depth_original=pr_depth_eval,
            ground_truth_depth_original=gt_depth_eval,
            alignment=args.alignment,
            max_depth=spec.max_depth,
            post_clip_max=spec.post_clip_max,
            device=eval_device,
            min_depth=spec.min_depth,
            scale_shift_fit_max_pixels=int(args.scale_shift_fit_max_pixels),
        )
        seq_metrics.update(
            {
                "fps": float(len(image_files) / max(infer_time, 1e-8)),
                "infer_time": float(infer_time),
                "frames": float(len(image_files)),
                "eval_frames": float(pr_depth_eval.shape[0]),
                "sequence": seq,
            }
        )
        dataset_seq_metrics.append(seq_metrics)
        sequence_summaries[seq] = dict(seq_metrics)
        all_fps.append(seq_metrics["fps"])
        infer_time_per_seq.append(seq_metrics["infer_time"])
        wall_seconds = time.perf_counter() - seq_wall_start
        append_progress_jsonl(
            progress_path,
            {
                "event": "sequence_done",
                "dataset": args.dataset,
                "sequence": seq,
                "sequence_index": seq_index,
                "num_sequences": len(sequence_names),
                "frames": len(image_files),
                "eval_frames": int(pr_depth_eval.shape[0]),
                "reused_predictions": reused_predictions,
                "infer_time": float(infer_time),
                "wall_seconds": float(wall_seconds),
                "time": datetime.now().isoformat(timespec="seconds"),
            },
        )
        log_progress(
            f"[{args.dataset}] done {seq_index}/{len(sequence_names)} sequence={seq} "
            f"frames={len(image_files)} eval_frames={pr_depth_eval.shape[0]} "
            f"reuse_pred={reused_predictions} infer={infer_time:.3f}s wall={wall_seconds:.3f}s"
        )

    summary = weighted_average(dataset_seq_metrics)
    summary["fps"] = float(np.average(np.array(all_fps, dtype=np.float64))) if all_fps else 0.0
    summary["infer_time"] = float(np.average(np.array(infer_time_per_seq, dtype=np.float64))) if infer_time_per_seq else 0.0
    summary["total_infer_seconds"] = float(total_infer_seconds)

    paper_target = resolve_paper_target(args.model_family, args.vggt_model_tag, args.dataset)
    delta_to_paper = {key: (summary[key] - paper_target[key]) for key in paper_target if key in summary}

    result = {
        "dataset": args.dataset,
        "protocol": "pi3_table4_videodepth",
        "model_family": args.model_family,
        "alignment": args.alignment,
        "layout": layout,
        "camera": args.camera,
        "data_root": str(data_root),
        "pred_root": str(pred_root),
        "output_dir": str(output_dir),
        "ckpt": ckpt,
        "vggt_model_tag": args.vggt_model_tag if args.model_family == "vggt" else "",
        "vggt_config": args.vggt_config if args.model_family == "vggt" else "",
        "vggt_official_ckpt_root": args.vggt_official_ckpt_root if args.model_family == "vggt" else "",
        "vggt_pt34_ckpt": args.vggt_pt34_ckpt if args.model_family == "vggt" else "",
        "vggt_topk_override": int(args.vggt_topk_override) if args.model_family == "vggt" else 0,
        "vggt_short_topk_override": int(args.vggt_short_topk_override) if args.model_family == "vggt" else 0,
        "vggt_long_topk_override": int(args.vggt_long_topk_override) if args.model_family == "vggt" else 0,
        "vggt_long_topk_min_frames": int(args.vggt_long_topk_min_frames) if args.model_family == "vggt" else 0,
        "vggt_dtype_override": args.vggt_dtype_override if args.model_family == "vggt" else "",
        "vggt_omega_repo": str(Path(args.vggt_omega_repo).expanduser().resolve()) if args.model_family == "vggt_omega" else "",
        "vggt_omega_checkpoint": (
            str(Path(args.vggt_omega_checkpoint).expanduser().resolve()) if args.model_family == "vggt_omega" else ""
        ),
        "vggt_omega_resolution": int(args.vggt_omega_resolution) if args.model_family == "vggt_omega" else 0,
        "vggt_omega_mode": str(args.vggt_omega_mode) if args.model_family == "vggt_omega" else "",
        "vggt_omega_global_attention_mode": (
            str(args.vggt_omega_global_attention_mode) if args.model_family == "vggt_omega" else ""
        ),
        "vggt_omega_query_view_index": (
            (-1 if args.vggt_omega_query_view_sweep else int(args.vggt_omega_query_view_index))
            if args.model_family == "vggt_omega"
            else 0
        ),
        "vggt_omega_query_view_sweep": bool(args.vggt_omega_query_view_sweep),
        "vggt_omega_crop_gt_to_input_fov": (
            bool(args.vggt_omega_crop_gt_to_input_fov) if args.model_family == "vggt_omega" else False
        ),
        "eval_fov_policy": str(args.eval_fov_policy),
        "eval_fov_policy_effective": str(eval_fov_policy_effective),
        "scale_shift_fit_max_pixels": int(args.scale_shift_fit_max_pixels),
        "vggt_omega_crop_boxes": (
            vggt_omega_crop_boxes if eval_fov_policy_effective == "vggt_omega_center_crop" else {}
        ),
        "da3_repo": str(Path(args.da3_repo).expanduser().resolve()) if args.model_family == "depthanything3" else "",
        "da3_model": str(Path(args.da3_model).expanduser().resolve()) if args.model_family == "depthanything3" else "",
        "process_res": int(args.process_res) if args.model_family == "depthanything3" else 0,
        "process_res_method": str(args.process_res_method) if args.model_family == "depthanything3" else "",
        "ref_view_strategy": str(args.ref_view_strategy) if args.model_family == "depthanything3" else "",
        "use_ray_pose": bool(args.use_ray_pose) if args.model_family == "depthanything3" else False,
        "align_to_input_ext_scale": (
            bool(args.align_to_input_ext_scale) if args.model_family == "depthanything3" else False
        ),
        "load_img_size": args.load_img_size,
        "max_size": int(args.max_size),
        "align_size": int(args.align_size),
        "safe_bound": int(args.safe_bound),
        "preprocess_style": str(args.preprocess_style),
        "eval_gt_shape": str(args.eval_gt_shape),
        "eval_gt_shape_effective": str(eval_gt_shape),
        "kitti_native_gt_eval": bool(kitti_native_gt_eval),
        "vggt_input_preprocess": str(args.vggt_input_preprocess),
        "eval_resize_to_1036p": bool(args.eval_resize_to_1036p),
        "eval_resize_to_1036p_effective": bool(eval_resize_to_1036p),
        "eval_frame_indices": "tuple_meta" if eval_frame_indices_spec == "tuple_meta" else [int(index) for index in eval_frame_indices_spec],
        "sequence_eval_frame_indices": sequence_eval_frame_indices,
        "device": str(device),
        "eval_device": str(eval_device),
        "num_sequences": len(sequence_names),
        "total_infer_seconds": total_infer_seconds,
        "summary": summary,
        "paper_target": paper_target,
        "delta_to_paper": delta_to_paper,
        "sequence_runtime_topk": sequence_runtime_topk,
        "sequence_summaries": sequence_summaries,
        "query_view_depth_metrics": query_view_depth_metrics,
    }

    json_path = output_dir / f"{args.dataset}_videodepth_protocol_summary.json"
    csv_path = output_dir / f"{args.dataset}_videodepth_protocol_metrics.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    save_csv(csv_path, summary)

    append_progress_jsonl(
        progress_path,
        {
            "event": "run_done",
            "dataset": args.dataset,
            "summary_path": str(json_path),
            "csv_path": str(csv_path),
            "summary": summary,
            "time": datetime.now().isoformat(timespec="seconds"),
        },
    )
    print(
        json.dumps({"summary": summary, "paper_target": paper_target, "delta_to_paper": delta_to_paper}, indent=2, ensure_ascii=False),
        flush=True,
    )
    log_progress(f"Saved summary JSON to {json_path}")
    log_progress(f"Saved summary CSV to {csv_path}")


if __name__ == "__main__":
    main()
