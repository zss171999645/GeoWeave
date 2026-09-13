#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvf
from PIL import Image
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2

from easyvolcap.engine import Config, MODELS
from easyvolcap.models.official_vggt_model import OfficialVGGTModel
from easyvolcap.utils.pi3.models.pi3 import Pi3
from easyvolcap.utils.base_utils import dotdict
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
BONN_SEQUENCES = [
    "rgbd_bonn_balloon2",
    "rgbd_bonn_crowd2",
    "rgbd_bonn_crowd3",
    "rgbd_bonn_person_tracking2",
    "rgbd_bonn_synchronous",
]
MULTIVIEW_EVC_DATASETS = frozenset({"kitti", "eth3d", "vkitti2", "co3dv2"})
PAPER_TARGETS = {
    "sintel": {"Abs Rel": 0.277, "δ < 1.25": 0.614},
    "bonn": {"Abs Rel": 0.044, "δ < 1.25": 0.976},
    "kitti": {"Abs Rel": 0.060, "δ < 1.25": 0.971},
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
    use_fg_masks: bool = False


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
        if dataset_name == "co3dv2" and is_eth3d_evc_train_layout(scene_root):
            rels.append(scene_root.relative_to(root).as_posix())
            continue
        if (scene_root / "images" / camera).is_dir() and (scene_root / "depths" / camera).is_dir():
            rels.append(scene_root.relative_to(root).as_posix())
    return sorted(set(rels))


def _img_ext_candidates(primary: str) -> List[str]:
    primary = primary.lower().lstrip(".")
    out: List[str] = []
    for ext in (primary, "jpg", "jpeg", "png"):
        e = ext.lower().lstrip(".")
        if e and e not in out:
            out.append(e)
    return out


def list_paired_multiview_evc_frames(image_dir: Path, gt_dir: Path, img_ext: str, gt_ext: str) -> Tuple[List[Path], List[Path]]:
    gt_ext_l = gt_ext.lower().lstrip(".")
    gts = sorted(p for p in gt_dir.iterdir() if p.is_file() and p.suffix.lower() == f".{gt_ext_l}")
    gt_by_stem = {p.stem: p for p in gts}
    images: List[Path] = []
    for ext_try in _img_ext_candidates(img_ext):
        images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() == f".{ext_try}")
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


def depth_read_exr(filename: str) -> np.ndarray:
    return read_exr_depth(filename)


ETH3D_EVC_VIEW_DIR_RE = re.compile(r"^\d{6}$")
MVS_SYNTH_EVC_VIEW_DIR_RE = re.compile(r"^\d{4}$")


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


def is_mvs_synth_evc_layout(seq_path: Path) -> bool:
    return is_view_major_evc_layout(seq_path, 4)


def list_view_major_evc_frame_pairs(seq_path: Path, digit_width: int) -> Tuple[List[Path], List[Path]]:
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


def list_mvs_synth_evc_frame_pairs(seq_path: Path) -> Tuple[List[Path], List[Path]]:
    return list_view_major_evc_frame_pairs(seq_path, 4)


def blendedmvs_scene_relpaths(data_root: Path) -> List[str]:
    """BlendedMVS EVC: <root>/<scene> or <root>/<subset>/<scene> with view-major images/<view06d>/."""
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
    raise ValueError(f"Unsupported depth suffix: {depth_path}")


def load_gt_depth_path(gt_path: Path, spec: DatasetSpec) -> np.ndarray:
    suf = gt_path.suffix.lower()
    if suf in {".png", ".dpt"}:
        return spec.depth_reader(str(gt_path))
    if spec.name == "diode" and suf == ".npy":
        return spec.depth_reader(str(gt_path))
    return read_depth_file(gt_path)


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
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Pi3-style monodepth with the official PI3 Table 6 protocol.",
    )
    parser.add_argument(
        "--model-family",
        choices=["pi3", "vggt"],
        default="pi3",
        help="Model family used to generate depth predictions.",
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS.keys()), required=True)
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Dataset root matching PI3 gathered layout, KITTI official flat layout, or EVC seq/images+depths layout.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for predictions and metrics. Defaults to tmp/pi3_monodepth_eval_<dataset>_<timestamp>.",
    )
    parser.add_argument(
        "--pred-root",
        type=str,
        default="",
        help="Prediction root. Defaults to <output-dir>/predictions/<dataset>.",
    )
    parser.add_argument(
        "--alignment",
        choices=["median-scale", "scale"],
        default="median-scale",
        help="Depth alignment mode. PI3 Table 6 uses the official `median-scale` setting, which maps to `align_with_scale=True` in PI3 code.",
    )
    parser.add_argument(
        "--layout",
        choices=["auto", "official", "evc"],
        default="auto",
        help="Dataset layout. `official` accepts PI3 gathered layout or KITTI official flat layout; `evc` expects seq/images + seq/depths.",
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
        help="Resize both input image and GT depth to max(H,W)=this before inference/eval (1036p style).",
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
            "Note: `--dataset kitti` always uses native GT resolution for metrics; `--eval-resize-to-1036p` is disabled for KITTI."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
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
        "--eval-only",
        action="store_true",
        help="Skip inference and only evaluate existing .npy predictions under --pred-root.",
    )
    parser.add_argument(
        "--eval-resize-to-1036p",
        action="store_true",
        help=(
            "Evaluate in 1036p space: resize BOTH pred+GT to (max(H,W)=max_size, align=align_size) "
            "before computing metrics. Inference input is unchanged."
        ),
    )
    parser.add_argument(
        "--depthmodel-output-space",
        choices=["normalized", "metric"],
        default="normalized",
        help=(
            "Depth space produced by DepthModel/VGGTSampler. "
            "`normalized` means the model predicts dpt/scale like MultiviewPointDataset.get_xyz() supervision. "
            "`metric` means the model predicts metric depth directly."
        ),
    )
    parser.add_argument(
        "--eval-depth-space",
        choices=["metric", "normalized"],
        default="metric",
        help=(
            "Depth space used for evaluation. "
            "`metric` compares against dataset metric GT (PI3-style). "
            "`normalized` compares against gt/scale (easyvolcap training-style)."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions.")
    parser.add_argument("--save-png", action="store_true", help="Also save visualization PNGs like the official script.")
    parser.add_argument("--max-seqs", type=int, default=0, help="Optional sequence limit for smoke testing.")
    parser.add_argument("--max-frames-per-seq", type=int, default=0, help="Optional frame limit per sequence for smoke testing.")
    return parser.parse_args()


def _find_first_manual_scale(obj) -> Optional[Tuple[str, float]]:
    """
    Best-effort extraction of (scale_type, scale) from a config-like structure.
    We look for a dict that contains scale_type == MANUAL and a numeric scale.
    """
    if isinstance(obj, dict):
        st = obj.get("scale_type", None)
        sc = obj.get("scale", None)
        if isinstance(st, str) and st.upper() == "MANUAL" and isinstance(sc, (int, float)):
            return ("MANUAL", float(sc))
        for v in obj.values():
            found = _find_first_manual_scale(v)
            if found is not None:
                return found
        return None
    if isinstance(obj, (list, tuple)):
        for v in obj:
            found = _find_first_manual_scale(v)
            if found is not None:
                return found
    return None


def resolve_default_output_dir(dataset: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("tmp") / f"pi3_monodepth_eval_{dataset}_{timestamp}"


def resolve_default_ckpt() -> str:
    env_ckpt = os.environ.get("PI3_PRETRAIN_CKPT", "").strip()
    if env_ckpt and Path(env_ckpt).is_file():
        return env_ckpt

    user_name = os.environ.get("USER", "feng01.zhou")
    candidates = [
        Path(f"/horizon-bucket/saturn_v_dev/01_users/{user_name}/projects/meshx/baseline/pretrained/pi3/yyfz233_Pi3_model.safetensors"),
        Path("weights/pi3/model.safetensors"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return DEFAULT_HF_CKPT


def load_pi3_model(ckpt: str, device: torch.device) -> Pi3:
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
    if args.vggt_model_tag == "official":
        cfg_path = args.vggt_config or "configs/exps/vggt/vggt_official_eval_paper.yaml"
    else:
        cfg_path = args.vggt_config or "aidi/configs/vggt/finetune_5090_sparse_topk2048_l9_19_p34_record.yaml"

    cfg = load_resolved_config(cfg_path)
    raw_model_cfg = dotdict(cfg.model_cfg)
    # Best-effort: extract manual scale used by MultiviewPointDataset.get_xyz() (if present in config).
    manual_scale = _find_first_manual_scale(cfg._cfg_dict if hasattr(cfg, "_cfg_dict") else dict(cfg))

    # Build official model when config matches OfficialVGGTModel signature.
    # Otherwise build via easyvolcap registry (DepthModel / custom models with sampler_cfg, etc.).
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

        # pt34 / custom state dict that still targets OfficialVGGTModel
        model = OfficialVGGTModel(**model_cfg).to(device=device).eval()
        checkpoint = torch.load(str(Path(args.vggt_pt34_ckpt)), map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        state_dict, _ = OfficialVGGTModel._remap_special_token_keys(state_dict)
        model.load_state_dict(state_dict, strict=False)
        return model

    # Registry build path: supports DepthModel and other easyvolcap models.
    model = MODELS.build(raw_model_cfg).to(device=device).eval()
    if manual_scale is not None:
        model._eval_scale_type, model._eval_manual_scale = manual_scale[0], manual_scale[1]
    model._eval_depthmodel_output_space = str(args.depthmodel_output_space)
    model._eval_depth_space = str(args.eval_depth_space)
    ckpt_path = Path(args.vggt_pt34_ckpt) if args.vggt_model_tag != "official" else Path("")
    if ckpt_path and str(ckpt_path) and ckpt_path.is_file():
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    return model


def list_regular_files(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.iterdir() if p.is_file() and not p.name.startswith(".")])


def parse_kitti_official_flat_image_name(file_name: str) -> Tuple[str, str, str]:
    match = KITTI_OFFICIAL_FLAT_IMAGE_RE.fullmatch(file_name)
    if match is None:
        raise ValueError(f"Unsupported KITTI official flat image name: {file_name}")
    return match.group("drive"), match.group("frame"), match.group("camera")


def parse_kitti_official_flat_gt_name(file_name: str) -> Tuple[str, str, str]:
    match = KITTI_OFFICIAL_FLAT_GT_RE.fullmatch(file_name)
    if match is None:
        raise ValueError(f"Unsupported KITTI official flat GT name: {file_name}")
    return match.group("drive"), match.group("frame"), match.group("camera")


def build_kitti_official_flat_gt_name(drive: str, frame: str, camera: str) -> str:
    return f"{drive}_groundtruth_depth_{frame}_image_{camera}.png"


def has_kitti_official_gathered_layout(data_root: Path) -> bool:
    return (data_root / "image_gathered").is_dir() and (data_root / "groundtruth_depth_gathered").is_dir()


def has_kitti_official_flat_layout(data_root: Path) -> bool:
    return (data_root / "image").is_dir() and (data_root / "groundtruth_depth").is_dir()


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

    extra_gt_names = sorted(set(gt_by_name) - expected_gt_names)
    if extra_gt_names:
        raise FileNotFoundError(f"Missing KITTI images for GT files like {extra_gt_names[0]} under {image_root}")

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
        if has_kitti_official_gathered_layout(data_root) or has_kitti_official_flat_layout(data_root):
            return "official"
        seq_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()]) if data_root.is_dir() else []
        for seq_dir in seq_dirs:
            if (seq_dir / "images" / camera).is_dir() and (seq_dir / "depths" / camera).is_dir():
                return "evc"
    elif spec.name in ("eth3d", "7scenes", "blendedmvs", "scannetpp"):
        seq_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()]) if data_root.is_dir() else []
        for seq_dir in seq_dirs:
            if is_eth3d_evc_train_layout(seq_dir):
                return "evc"
        if spec.name == "blendedmvs" and blendedmvs_scene_relpaths(data_root):
            return "evc"
        cam_try = resolve_camera_for_multiview_evc(data_root, camera, "eth3d")
        for seq_dir in seq_dirs:
            if (seq_dir / "images" / cam_try).is_dir() and (seq_dir / "depths" / cam_try).is_dir():
                return "evc"
    elif spec.name == "diode":
        if diode_val_scene_relpaths(data_root):
            return "evc"
    elif spec.name == "mvs_synth":
        seq_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()]) if data_root.is_dir() else []
        for seq_dir in seq_dirs:
            if is_mvs_synth_evc_layout(seq_dir):
                return "evc"
        cam_try = resolve_camera_for_multiview_evc(data_root, camera, "eth3d")
        for seq_dir in seq_dirs:
            if (seq_dir / "images" / cam_try).is_dir() and (seq_dir / "depths" / cam_try).is_dir():
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
                    f"Cannot find KITTI official layout under {data_root}: expected image_gathered/groundtruth_depth_gathered or image/groundtruth_depth"
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
        if spec.name == "diode":
            names = diode_val_scene_relpaths(data_root)
        else:
            if spec.sequence_names is None:
                names = []
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
            if eth3d_vm:
                names = sorted(names)
            else:
                names = [
                    seq for seq in names
                    if (data_root / seq / "images" / camera).is_dir() and (data_root / seq / "depths" / camera).is_dir()
                ]
        elif spec.name == "mvs_synth":
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
        elif spec.name == "diode":
            pass
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


def list_sequence_files(data_root: Path, subdir_pattern: str, seq: str, ext: str) -> List[Path]:
    subdir = data_root / subdir_pattern.format(seq=seq)
    files = sorted(subdir.glob(f"*.{ext}"))
    return files


def load_and_resize14(file_path: Path, new_width: int, device: torch.device) -> torch.Tensor:
    img = Image.open(file_path).convert("RGB")
    w_orig, h_orig = img.size
    target_w = new_width
    target_h = max(14, round(h_orig * (new_width / max(w_orig, 1)) / 14) * 14)
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    tensor = tvf.ToTensor()(img).unsqueeze(0).to(device)
    patch_h, patch_w = tensor.shape[-2] // 14, tensor.shape[-1] // 14
    tensor = F.interpolate(
        tensor,
        (patch_h * 14, patch_w * 14),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return tensor.unsqueeze(0)


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


def preprocess_image_and_gt(
    image_path: Path,
    gt_depth: np.ndarray,
    max_size: int,
    align_size: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Apply easyvolcap-style resizing semantics:
    ratio = max_size / max(H, W), then round H/W to align_size.
    Both image and GT are transformed to the same (H, W).
    """
    img = Image.open(image_path).convert("RGB")
    w0, h0 = img.size
    h1, w1 = _compute_aligned_hw(h0, w0, max_size=max_size, align_size=align_size)
    if (h1, w1) != (h0, w0):
        img = img.resize((w1, h1), Image.Resampling.LANCZOS)
    image_tensor = tvf.ToTensor()(img).unsqueeze(0)  # (1, 3, H, W)

    gt = gt_depth.astype(np.float32, copy=False)
    if gt.shape[0] != h1 or gt.shape[1] != w1:
        gt = cv2.resize(gt, (w1, h1), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return image_tensor, gt


def preprocess_image_only(
    image_path: Path,
    max_size: int,
    align_size: int,
) -> torch.Tensor:
    """Same as preprocess_image_and_gt but without touching GT."""
    img = Image.open(image_path).convert("RGB")
    w0, h0 = img.size
    h1, w1 = _compute_aligned_hw(h0, w0, max_size=max_size, align_size=align_size)
    if (h1, w1) != (h0, w0):
        img = img.resize((w1, h1), Image.Resampling.LANCZOS)
    return tvf.ToTensor()(img).unsqueeze(0)  # (1, 3, H, W)


def _mvpd_target_hw(
    h0: int,
    w0: int,
    proc_max_size: int,
    proc_align_size: int,
    *,
    co3dv2_portrait_square: bool = False,
) -> Tuple[int, int]:
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


def preprocess_image_and_gt_mvpd(
    image_path: Path,
    gt_depth: np.ndarray,
    proc_max_size: int,
    proc_align_size: int,
    safe_bound: int,
    *,
    co3dv2_portrait_square: bool = False,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Approximate MultiviewPointDataset.get_sources() resizing for eval:
    - compute target (ht, wt) from aspect_ratio and proc_max_size
    - resize image/gt to be >= (ht+safe_bound, wt+safe_bound)
    - center-crop to (ht, wt)
    """
    img = Image.open(image_path).convert("RGB")
    w0, h0 = img.size
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

    img_np = np.asarray(img, dtype=np.uint8)
    if (h1, w1) != (h0, w0):
        img_np = cv2.resize(img_np, (w1, h1), interpolation=cv2.INTER_AREA)

    start_h = max(0, (h1 - ht) // 2)
    start_w = max(0, (w1 - wt) // 2)
    img_np = img_np[start_h:start_h + ht, start_w:start_w + wt, :]
    image_tensor = torch.from_numpy(img_np).to(dtype=torch.float32).permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # (1, 3, ht, wt)

    gt = gt_depth.astype(np.float32, copy=False)
    if gt.shape[0] != h1 or gt.shape[1] != w1:
        gt = cv2.resize(gt, (w1, h1), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    gt = gt[start_h:start_h + ht, start_w:start_w + wt]
    return image_tensor, gt


def preprocess_image_only_mvpd(
    image_path: Path,
    proc_max_size: int,
    proc_align_size: int,
    safe_bound: int,
    *,
    co3dv2_portrait_square: bool = False,
) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    w0, h0 = img.size
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
    img_np = np.asarray(img, dtype=np.uint8)
    if (h1, w1) != (h0, w0):
        img_np = cv2.resize(img_np, (w1, h1), interpolation=cv2.INTER_AREA)
    start_h = max(0, (h1 - ht) // 2)
    start_w = max(0, (w1 - wt) // 2)
    img_np = img_np[start_h:start_h + ht, start_w:start_w + wt, :]
    image_tensor = torch.from_numpy(img_np).to(dtype=torch.float32).permute(2, 0, 1) / 255.0
    return image_tensor.unsqueeze(0)


def infer_pi3_depth_from_tensor(model: Pi3, image_1x3_hw: torch.Tensor, device: torch.device) -> np.ndarray:
    imgs = image_1x3_hw.to(device=device).unsqueeze(0)  # (1, 1, 3, H, W)
    use_amp = device.type == "cuda"
    dtype = None
    if use_amp:
        major, _ = torch.cuda.get_device_capability(device=device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                pred = model(imgs)
        else:
            pred = model(imgs)
    return pred["local_points"][0, 0, ..., -1].detach().float().cpu().numpy()


def infer_vggt_depth_from_tensor(model: torch.nn.Module, image_1x3_hw: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    Supports both:
    - OfficialVGGTModel (expects batch.images and meta.H/W)
    - DepthModel / sampler-based models (expects batch.rgb flattened like dataset outputs)
    """
    images = image_1x3_hw.to(device=device)  # (1, 3, H, W)
    height, width = int(images.shape[-2]), int(images.shape[-1])

    # Provide both representations to maximize compatibility.
    rgb_flat = images.permute(0, 2, 3, 1).reshape(1, 1, height * width, 3)  # (B=1,S=1,P,3)
    batch = dotdict(
        images=images.unsqueeze(0),  # (1, 1, 3, H, W)
        rgb=rgb_flat,  # (1, 1, P, 3)
        scale=torch.tensor([1.0], device=images.device),
        loss_scaler=torch.tensor(1.0, device=images.device),
        meta=dotdict(
            iter=torch.tensor(0, device=images.device),
            H=torch.tensor([height], device=images.device),
            W=torch.tensor([width], device=images.device),
        ),
    )

    use_amp = device.type == "cuda"
    dtype = None
    if use_amp:
        major, _ = torch.cuda.get_device_capability(device=device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16

    def _is_depth_model(m: torch.nn.Module) -> bool:
        return m.__class__.__name__ == "DepthModel" and hasattr(m, "sampler")

    with torch.no_grad():
        if _is_depth_model(model):
            # DepthModel.prepare_data requires K/R/T etc. For evaluation-only depth maps, we can
            # bypass it and directly run the sampler which only needs rgb/meta (+ optional cam).
            sampler = model.sampler
            # If cam embedding is enabled but we do not have real camera params here,
            # disable it temporarily to avoid conditioning on all-zero cam.
            old_use_cam_emb = getattr(sampler, "use_cam_emb", None)
            patched_cam_emb = False
            if bool(old_use_cam_emb):
                setattr(sampler, "use_cam_emb", False)
                patched_cam_emb = True
            if not hasattr(batch, "cam"):
                batch.cam = torch.zeros((1, 1, 9), device=images.device, dtype=images.dtype)
            out_batch = sampler(batch=batch)
            if patched_cam_emb:
                setattr(sampler, "use_cam_emb", old_use_cam_emb)
            output = out_batch.output
        else:
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    output = model(batch)
            else:
                output = model(batch)

    if hasattr(output, "dpt_map"):
        dpt_map = output.dpt_map
    elif hasattr(output, "output") and hasattr(output.output, "dpt_map"):
        dpt_map = output.output.dpt_map
    else:
        raise ValueError("Model output does not contain dpt_map")

    return dpt_map[0, 0, :, 0].detach().float().reshape(height, width).cpu().numpy()


def _apply_depth_space_transform_for_depthmodel(
    model: torch.nn.Module,
    pred_depth: np.ndarray,
    gt_depth_metric: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align pred/gt depth spaces for DepthModel (trained with MultiviewPointDataset normalization).
    - If eval space is metric: compare metric GT with metric pred (pred_norm * scale).
    - If eval space is normalized: compare gt_metric/scale with pred_norm.
    If we cannot determine scale, fall back to scale=1.
    """
    if model.__class__.__name__ != "DepthModel":
        return pred_depth, gt_depth_metric

    out_space = getattr(model, "_eval_depthmodel_output_space", "normalized")
    eval_space = getattr(model, "_eval_depth_space", "metric")
    scale_type = getattr(model, "_eval_scale_type", "")
    scale = float(getattr(model, "_eval_manual_scale", 1.0)) if str(scale_type).upper() == "MANUAL" else 1.0
    scale = max(scale, 1e-8)

    pred = pred_depth.astype(np.float32, copy=False)
    gt_metric = gt_depth_metric.astype(np.float32, copy=False)

    if out_space == "metric":
        pred_metric = pred
        pred_norm = pred / scale
    else:
        pred_norm = pred
        pred_metric = pred * scale

    if eval_space == "normalized":
        gt_eval = gt_metric / scale
        pred_eval = pred_norm
    else:
        gt_eval = gt_metric
        pred_eval = pred_metric
    return pred_eval.astype(np.float32, copy=False), gt_eval.astype(np.float32, copy=False)


def infer_pi3_depth(model: Pi3, file_path: Path, load_img_size: int, device: torch.device) -> np.ndarray:
    imgs = load_and_resize14(file_path, load_img_size, device)
    use_amp = device.type == "cuda"
    dtype = None
    if use_amp:
        major, _ = torch.cuda.get_device_capability(device=device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16

    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                pred = model(imgs)
        else:
            pred = model(imgs)
    return pred["local_points"][0, 0, ..., -1].detach().float().cpu().numpy()


def infer_vggt_depth(model: OfficialVGGTModel, file_path: Path, device: torch.device) -> np.ndarray:
    # Legacy helper kept for backward-compatibility; evaluation now uses
    # infer_vggt_depth_from_tensor() so preprocessing can be aligned to 1036p.
    img = Image.open(file_path).convert("RGB")
    tensor = tvf.ToTensor()(img).unsqueeze(0)
    return infer_vggt_depth_from_tensor(model, tensor, device)


def infer_vggt_depth_official_preprocess(model: OfficialVGGTModel, file_path: Path, device: torch.device) -> np.ndarray:
    """Run OfficialVGGTModel with VGGT official 518 preprocessing."""
    from easyvolcap.official_vggt.utils.load_fn import load_and_preprocess_images as load_and_preprocess_vggt_images

    images = load_and_preprocess_vggt_images([str(file_path)]).to(device=device)
    height, width = int(images.shape[-2]), int(images.shape[-1])
    batch = dotdict(
        images=images.unsqueeze(0),
        meta=dotdict(
            iter=torch.tensor(0, device=images.device),
            H=torch.tensor([height], device=images.device),
            W=torch.tensor([width], device=images.device),
        ),
    )
    use_amp = device.type == "cuda"
    dtype = None
    if use_amp:
        major, _ = torch.cuda.get_device_capability(device=device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                output = model(batch)
        else:
            output = model(batch)
    return output.dpt_map[0, 0, :, 0].detach().float().reshape(height, width).cpu().numpy()


def resize_depth_to_1036p(depth: np.ndarray, max_size: int, align_size: int, is_gt: bool) -> np.ndarray:
    h0, w0 = int(depth.shape[0]), int(depth.shape[1])
    h1, w1 = _compute_aligned_hw(h0, w0, max_size=int(max_size), align_size=int(align_size))
    if (h1, w1) == (h0, w0):
        return depth.astype(np.float32, copy=False)
    interp = cv2.INTER_NEAREST if is_gt else cv2.INTER_CUBIC
    return cv2.resize(depth.astype(np.float32), (w1, h1), interpolation=interp).astype(np.float32)


def write_prediction_png(depth: np.ndarray, png_path: Path) -> None:
    normalized = depth - depth.min()
    denom = float(normalized.max())
    if denom > 0:
        normalized = normalized / denom
    png = (normalized * 255.0).astype(np.uint8)
    Image.fromarray(png).save(png_path)


def resize_prediction(prediction: np.ndarray, shape_hw: Sequence[int]) -> np.ndarray:
    height, width = int(shape_hw[0]), int(shape_hw[1])
    return cv2.resize(prediction.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)


def align_with_scale(predicted_depth: torch.Tensor, ground_truth_depth: torch.Tensor) -> torch.Tensor:
    dot_pred_gt = torch.nanmean(ground_truth_depth)
    dot_pred_pred = torch.nanmean(predicted_depth)
    scale = dot_pred_gt / dot_pred_pred
    for _ in range(10):
        residuals = scale * predicted_depth - ground_truth_depth
        weights = 1.0 / (residuals.abs() + 1e-8)
        weighted_dot_pred_gt = torch.sum(weights * predicted_depth * ground_truth_depth)
        weighted_dot_pred_pred = torch.sum(weights * predicted_depth**2)
        scale = weighted_dot_pred_gt / weighted_dot_pred_pred
    scale = scale.clamp(min=1e-3).detach()
    return predicted_depth * scale


def evaluate_depth_map(
    predicted_depth_original: np.ndarray,
    ground_truth_depth_original: np.ndarray,
    alignment: str,
    max_depth: Optional[float],
    post_clip_max: Optional[float],
    device: torch.device,
) -> Dict[str, float]:
    predicted_depth = torch.from_numpy(predicted_depth_original).to(device=device, dtype=torch.float32)
    ground_truth_depth = torch.from_numpy(ground_truth_depth_original).to(device=device, dtype=torch.float32)

    if max_depth is not None:
        mask = (ground_truth_depth > 0) & (ground_truth_depth < max_depth)
    else:
        mask = ground_truth_depth > 0

    predicted_depth = predicted_depth[mask]
    ground_truth_depth = ground_truth_depth[mask]

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

    # Match the PI3 official monodepth code exactly: `invariant=median-scale`
    # routes to `align_with_scale=True`, while the fallback branch uses
    # the literal median(gt)/median(pred) scaling.
    if alignment == "median-scale":
        predicted_depth = align_with_scale(predicted_depth, ground_truth_depth)
    else:
        scale_factor = torch.median(ground_truth_depth) / torch.median(predicted_depth)
        predicted_depth = predicted_depth * scale_factor

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


def main() -> None:
    args = parse_args()
    spec = DATASET_SPECS[args.dataset]
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir) if args.output_dir else resolve_default_output_dir(args.dataset)
    pred_root = Path(args.pred_root) if args.pred_root else output_dir / "predictions" / args.dataset
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
            "DIODE val: <split>/scene_*/scan_* with <stem>.png + <stem>_depth.npy). Use --layout evc or --layout auto."
        )

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    model: Optional[object] = None
    ckpt = (args.ckpt or resolve_default_ckpt()) if args.model_family == "pi3" else ""
    if not args.eval_only:
        if args.model_family == "pi3":
            model = load_pi3_model(ckpt, device)
        else:
            model = load_vggt_model(args, device)

    layout = args.layout if args.layout != "auto" else detect_layout(spec, data_root, args.camera)
    print(f"Using dataset layout: {layout}")
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
            print(
                "EVC view-major: each images/<view##>/ is one frame (`--camera` ignored); "
                "6-digit (eth3d/7scenes/blendedmvs/scannetpp) or 4-digit (mvs_synth). "
                "BlendedMVS may use <root>/<subset>/<scene> paths."
            )
        elif co3d_view_major:
            print(
                "CO3Dv2 EVC (co3d_to_easyvolcap layout): each images/<view06d>/ is one evaluation frame; "
                "`--camera` does not subset views."
            )
            resolved_cam = resolve_camera_for_multiview_evc(data_root, args.camera, spec.name)
            if resolved_cam != args.camera:
                print(f"Multiview EVC (probe): using camera directory '{resolved_cam}' (--camera was '{args.camera}')")
            args.camera = resolved_cam
        else:
            resolved_cam = resolve_camera_for_multiview_evc(data_root, args.camera, spec.name)
            if resolved_cam != args.camera:
                print(f"Multiview EVC: using camera directory '{resolved_cam}' (--camera was '{args.camera}')")
            args.camera = resolved_cam

    sequence_names = get_sequence_names_for_layout(spec, data_root, layout, args.camera)
    if args.max_seqs > 0:
        sequence_names = sequence_names[: args.max_seqs]

    kitti_native_gt_eval = spec.name == "kitti"
    if kitti_native_gt_eval:
        if args.eval_gt_shape != "official" or args.eval_resize_to_1036p:
            print(
                "KITTI: metrics use native GT depth (unchanged from disk); predictions are resized to GT shape. "
                "Overriding --eval-gt-shape / --eval-resize-to-1036p."
            )
    eval_gt_shape = "official" if kitti_native_gt_eval else args.eval_gt_shape
    eval_resize_to_1036p = False if kitti_native_gt_eval else bool(args.eval_resize_to_1036p)

    dataset_frame_metrics: List[Dict[str, float]] = []
    sequence_summaries: Dict[str, Dict[str, float]] = {}
    total_infer_seconds = 0.0
    total_frames = 0

    for seq in sequence_names:
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
                f"[{seq}] image/gt count mismatch: {len(image_files)} vs {len(gt_files)} "
                f"under {data_root} (layout={layout})"
            )

        print(f"[{args.dataset}] sequence={seq} frames={len(image_files)}")
        seq_metrics: List[Dict[str, float]] = []
        seq_pred_dir = pred_root / seq
        seq_pred_dir.mkdir(parents=True, exist_ok=True)
        for image_path, gt_path in zip(image_files, gt_files):
            if spec.name in ("eth3d", "7scenes", "blendedmvs", "scannetpp") and ETH3D_EVC_VIEW_DIR_RE.fullmatch(
                image_path.parent.name
            ):
                pred_stem = f"{image_path.parent.name}_{image_path.stem}"
            elif spec.name == "mvs_synth" and MVS_SYNTH_EVC_VIEW_DIR_RE.fullmatch(image_path.parent.name):
                pred_stem = f"{image_path.parent.name}_{image_path.stem}"
            else:
                pred_stem = image_path.stem
            pred_npy = seq_pred_dir / f"{pred_stem}.npy"
            pred_png = seq_pred_dir / f"{pred_stem}.png"

            gt_depth_raw = load_gt_depth_path(gt_path, spec)
            if spec.use_fg_masks:
                gt_depth_raw = apply_fg_mask_to_depth(
                    gt_depth_raw,
                    evc_fg_mask_path(data_root, seq, image_path, args.camera),
                )

            if eval_gt_shape == "official":
                # Inference uses 1036p preprocessing, but evaluation stays on raw GT shape.
                if args.preprocess_style == "mvpd":
                    image_tensor_1x3_hw = preprocess_image_only_mvpd(
                        image_path=image_path,
                        proc_max_size=int(args.max_size),
                        proc_align_size=int(args.align_size),
                        safe_bound=int(args.safe_bound),
                        co3dv2_portrait_square=(args.dataset == "co3dv2"),
                    )
                else:
                    image_tensor_1x3_hw = preprocess_image_only(
                        image_path=image_path,
                        max_size=int(args.max_size),
                        align_size=int(args.align_size),
                    )
                gt_depth = gt_depth_raw
            else:
                # Default: image + GT are both preprocessed to 1036p space.
                if args.preprocess_style == "mvpd":
                    image_tensor_1x3_hw, gt_depth = preprocess_image_and_gt_mvpd(
                        image_path=image_path,
                        gt_depth=gt_depth_raw,
                        proc_max_size=int(args.max_size),
                        proc_align_size=int(args.align_size),
                        safe_bound=int(args.safe_bound),
                        co3dv2_portrait_square=(args.dataset == "co3dv2"),
                    )
                else:
                    image_tensor_1x3_hw, gt_depth = preprocess_image_and_gt(
                        image_path=image_path,
                        gt_depth=gt_depth_raw,
                        max_size=int(args.max_size),
                        align_size=int(args.align_size),
                    )

            if not pred_npy.is_file() or args.overwrite:
                if model is None:
                    raise RuntimeError("Model is not initialized while inference is required.")
                start = datetime.now()
                if args.model_family == "pi3":
                    pred_depth = infer_pi3_depth_from_tensor(model, image_tensor_1x3_hw, device)
                else:
                    if (
                        args.vggt_model_tag == "official"
                        and args.vggt_input_preprocess == "official"
                        and hasattr(model, "__class__")
                        and model.__class__.__name__ == "OfficialVGGTModel"
                    ):
                        pred_depth = infer_vggt_depth_official_preprocess(model, image_path, device)
                    else:
                        pred_depth = infer_vggt_depth_from_tensor(model, image_tensor_1x3_hw, device)
                total_infer_seconds += (datetime.now() - start).total_seconds()
                np.save(pred_npy, pred_depth)
                if args.save_png:
                    write_prediction_png(pred_depth, pred_png)
            else:
                pred_depth = np.load(pred_npy)

            # For DepthModel: align pred/gt depth spaces (metric vs normalized) before resizing for eval.
            pred_depth, gt_depth_raw_aligned = _apply_depth_space_transform_for_depthmodel(
                model=model if model is not None else torch.nn.Identity(),
                pred_depth=pred_depth,
                gt_depth_metric=gt_depth_raw,
            )

            if eval_resize_to_1036p:
                gt_eval = resize_depth_to_1036p(gt_depth_raw_aligned, max_size=int(args.max_size), align_size=int(args.align_size), is_gt=True)
                pred_eval = resize_depth_to_1036p(pred_depth, max_size=int(args.max_size), align_size=int(args.align_size), is_gt=False)
            else:
                if kitti_native_gt_eval:
                    gt_eval = gt_depth_raw_aligned
                    pred_eval = resize_prediction(pred_depth, gt_eval.shape)
                else:
                    gt_eval = gt_depth
                    pred_eval = resize_prediction(pred_depth, gt_depth.shape)
            frame_metrics = evaluate_depth_map(
                predicted_depth_original=pred_eval,
                ground_truth_depth_original=gt_eval,
                alignment=args.alignment,
                max_depth=spec.max_depth,
                post_clip_max=spec.post_clip_max,
                device=device,
            )
            frame_metrics.update(
                {
                    "path": str(image_path),
                    "gt_path": str(gt_path),
                    "prediction_path": str(pred_npy),
                    "sequence": seq,
                }
            )
            seq_metrics.append(frame_metrics)
            dataset_frame_metrics.append(frame_metrics)
            total_frames += 1

        sequence_summaries[seq] = weighted_average(seq_metrics)

    summary = weighted_average(dataset_frame_metrics)
    paper_target = PAPER_TARGETS.get(args.dataset, {})
    delta_to_paper = {
        key: (summary[key] - paper_target[key]) for key in paper_target if key in summary
    }

    result = {
        "dataset": args.dataset,
        "protocol": "pi3_table6_monodepth",
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
        "depthmodel_output_space": str(args.depthmodel_output_space),
        "eval_depth_space": str(args.eval_depth_space),
        "device": str(device),
        "num_sequences": len(sequence_names),
        "num_frames": total_frames,
        "total_infer_seconds": total_infer_seconds,
        "summary": summary,
        "paper_target": paper_target,
        "delta_to_paper": delta_to_paper,
        "sequence_summaries": sequence_summaries,
        "frame_metrics": dataset_frame_metrics,
    }

    json_path = output_dir / f"{args.dataset}_monodepth_protocol_summary.json"
    csv_path = output_dir / f"{args.dataset}_monodepth_protocol_metrics.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    save_csv(csv_path, summary)

    print(json.dumps({"summary": summary, "paper_target": paper_target, "delta_to_paper": delta_to_paper}, indent=2, ensure_ascii=False))
    print(f"Saved summary JSON to {json_path}")
    print(f"Saved summary CSV to {csv_path}")


if __name__ == "__main__":
    main()
