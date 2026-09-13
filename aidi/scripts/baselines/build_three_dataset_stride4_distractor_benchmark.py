#!/usr/bin/env python3
"""Build a unified stride-4 distractor benchmark for driving datasets.

Protocol:
  clean-10 = 10 frames from the target scene with stride 4.
  noise-10 = the same first 6 target frames plus 4 frames from another scene.
  If --fixed-distractor-camera is set, noise frames come from the same scene
  and the fixed camera at the clean tuple tail timestamps.

Only frames 0..5 are intended for evaluation.  The output uses the PI3
relpose "official" layout so eval_pi3_relpose_distance_protocol.py can be
reused with --datasets vkitti2 and --eval-frame-indices 0,1,2,3,4,5.
"""

from __future__ import annotations

import argparse
import struct
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

DEFAULT_WAYMO_ROOT = (
    "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/tao02.xie/datasets/"
    "scatt3r_evaluation/waymo_open_dataset_v1_4_3"
)
DEFAULT_KITTI_ROOT = (
    "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/tao02.xie/datasets/"
    "kitti/odometry"
)
DEFAULT_VKITTI_ROOT = "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/datasets/vkitti2_resolved"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_repo_root_on_syspath() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


ensure_repo_root_on_syspath()

from easyvolcap.utils.easy_utils import read_camera


SUPPORTED_DATASETS = {"waymo", "kitti", "vkitti2"}


@dataclass(frozen=True)
class SceneSpec:
    dataset: str
    name: str
    root: Path
    camera: str
    image_exts: Tuple[str, ...]
    has_depth: bool = False
    depth_ext: str = "exr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="waymo,kitti,vkitti2")
    parser.add_argument("--waymo-root", default=DEFAULT_WAYMO_ROOT)
    parser.add_argument("--kitti-root", default=DEFAULT_KITTI_ROOT)
    parser.add_argument("--vkitti-root", default=DEFAULT_VKITTI_ROOT)
    parser.add_argument("--waymo-camera", default="cycle")
    parser.add_argument("--kitti-camera", default="2")
    parser.add_argument("--vkitti-camera", default="00")
    parser.add_argument(
        "--vkitti-scene-specs",
        default="Scene01/clone,Scene06/clone,Scene18/clone,Scene20/clone",
        help="Comma-separated VKITTI scene/weather specs. Use 'all' to discover all.",
    )
    parser.add_argument("--output-root", default="")
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--max-anchors-per-scene", type=int, default=5)
    parser.add_argument("--anchor-quantiles", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--total-views", type=int, default=10)
    parser.add_argument("--eval-views", type=int, default=6)
    parser.add_argument("--noise-views", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument(
        "--distractor-scene-offset",
        type=int,
        default=1,
        help="Deterministic offset into the scene list for picking a different distractor scene.",
    )
    parser.add_argument(
        "--match-distractor-resolution",
        action="store_true",
        help="Only pick distractor frames whose image resolution matches the target eval frames.",
    )
    parser.add_argument(
        "--match-distractor-camera",
        action="store_true",
        help="Only pick distractor frames from the same camera id as the target tuple.",
    )
    parser.add_argument(
        "--fixed-distractor-camera",
        default="",
        help=(
            "Use this camera from the same scene as deterministic distractor context. "
            "When set, the noise tail uses clean tail frame names from this camera and "
            "does not search other scenes."
        ),
    )
    parser.add_argument(
        "--require-fixed-distractor-frame-match",
        action="store_true",
        help=(
            "When --fixed-distractor-camera is set, require the distractor camera to contain "
            "the exact clean tail frame names. If not, fail instead of falling back to the "
            "same local start index on the distractor camera."
        ),
    )
    parser.add_argument(
        "--output-image-ext",
        default=".jpg",
        help="Filename suffix used in output color_90. Keep .jpg for the reused vkitti2 relpose spec.",
    )
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of creating symlinks.")
    return parser.parse_args()


def parse_datasets(value: str) -> List[str]:
    aliases = {"vkitti": "vkitti2", "virtual_kitti": "vkitti2"}
    items = [aliases.get(item.strip().lower(), item.strip().lower()) for item in value.split(",") if item.strip()]
    bad = [item for item in items if item not in SUPPORTED_DATASETS]
    if bad:
        raise ValueError(f"Unsupported datasets: {bad}; choose from {sorted(SUPPORTED_DATASETS)}")
    return items


def parse_anchor_quantiles(value: str) -> List[float]:
    quantiles = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not quantiles:
        raise ValueError("anchor quantiles cannot be empty")
    return quantiles


def default_output_root() -> Path:
    return repo_root() / "tmp" / f"driving_stride4_distractor_{time.strftime('%Y%m%d_%H%M%S')}"


def discover_vkitti_scenes(args: argparse.Namespace) -> List[SceneSpec]:
    specs = [item.strip() for item in str(args.vkitti_scene_specs).split(",") if item.strip()]
    if len(specs) == 1 and specs[0].lower() == "all":
        root = Path(args.vkitti_root)
        scenes: List[SceneSpec] = []
        for scene_group in list_subdirs(root):
            for scene_root in list_subdirs(scene_group):
                scenes.append(
                    SceneSpec(
                        dataset="vkitti2",
                        name=f"{scene_group.name}/{scene_root.name}",
                        root=scene_root,
                        camera=args.vkitti_camera,
                        image_exts=("jpg", "png"),
                        has_depth=False,
                    )
                )
        if args.limit_scenes > 0:
            scenes = scenes[: args.limit_scenes]
        return scenes

    scenes: List[SceneSpec] = []
    root = Path(args.vkitti_root)
    for spec in specs:
        scene_root = root / spec
        scenes.append(
            SceneSpec(
                dataset="vkitti2",
                name=spec,
                root=scene_root,
                camera=args.vkitti_camera,
                image_exts=("jpg", "png"),
                has_depth=False,
            )
        )
    if args.limit_scenes > 0:
        scenes = scenes[: args.limit_scenes]
    return scenes


def discover_dataset_scenes(dataset: str, args: argparse.Namespace) -> List[SceneSpec]:
    if dataset == "vkitti2":
        return discover_vkitti_scenes(args)
    scenes: List[SceneSpec] = []
    if dataset == "waymo":
        root = Path(args.waymo_root)
        for idx, scene_root in enumerate(list_subdirs(root)):
            cams = camera_choices(args.waymo_camera, scene_root / "images")
            if not cams:
                continue
            cam = cams[idx % len(cams)] if args.waymo_camera.strip().lower() == "cycle" else cams[0]
            scenes.append(SceneSpec(dataset, scene_root.name, scene_root, cam, ("png",)))
    elif dataset == "kitti":
        root = Path(args.kitti_root)
        for scene_root in list_subdirs(root):
            scenes.append(SceneSpec(dataset, scene_root.name, scene_root, args.kitti_camera, ("png",)))
    else:
        raise ValueError(f"Unsupported dataset={dataset}")
    if args.limit_scenes > 0:
        scenes = scenes[: args.limit_scenes]
    return scenes


def list_subdirs(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda p: p.name)


def camera_choices(camera_spec: str, image_root: Path) -> List[str]:
    cams = sorted([p.name for p in image_root.iterdir() if p.is_dir()]) if image_root.is_dir() else []
    requested = [item.strip() for item in camera_spec.split(",") if item.strip()]
    if camera_spec.strip().lower() == "cycle":
        return cams
    if requested:
        return [cam for cam in requested if cam in cams] or requested
    return cams[:1]


def _numeric_key(text: str) -> Tuple[int, str]:
    try:
        return int(text), text
    except ValueError:
        return 0, text


def image_map(scene: SceneSpec) -> Dict[str, Path]:
    img_dir = scene.root / "images" / scene.camera
    out: Dict[str, Path] = {}
    for ext in scene.image_exts:
        for path in img_dir.glob(f"*.{ext}"):
            out[path.stem] = path
    return out


def depth_map(scene: SceneSpec) -> Dict[str, Path]:
    if not scene.has_depth:
        return {}
    depth_dir = scene.root / "depths" / scene.camera
    return {path.stem: path for path in depth_dir.glob(f"*.{scene.depth_ext}")}


def normalize_camera_key(key: Any) -> str:
    if isinstance(key, (int, np.integer)):
        return str(int(key))
    return str(key)


def load_scene_cameras(scene: SceneSpec) -> Dict[str, Any]:
    cam_dir = scene.root / "cameras" / scene.camera
    intri = cam_dir / "intri.yml"
    extri = cam_dir / "extri.yml"
    cameras = read_camera(str(intri), str(extri), use_dict=False)
    return {normalize_camera_key(key): value for key, value in cameras.items()}


def pose3x4_to_c2w(rt: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :4] = np.asarray(rt, dtype=np.float64)
    return np.linalg.inv(pose)


def available_frame_names(scene: SceneSpec) -> List[str]:
    imgs = image_map(scene)
    cams = load_scene_cameras(scene)
    common = set(imgs) & set(cams)
    if scene.has_depth:
        common &= set(depth_map(scene))
    return sorted(common, key=_numeric_key)


def quantile_starts(num_frames: int, total_views: int, stride: int, max_anchors: int, quantiles: Sequence[float]) -> List[int]:
    max_start = num_frames - 1 - (int(total_views) - 1) * int(stride)
    if max_start < 0 or max_anchors <= 0:
        return []
    selected: List[int] = []
    used: set[int] = set()
    for quantile in list(quantiles)[:max_anchors]:
        base = int(np.floor(float(quantile) * (max_start + 1)))
        base = min(max(base, 0), max_start)
        candidates = [base]
        for radius in range(1, max_start + 1):
            left = base - radius
            right = base + radius
            if left >= 0:
                candidates.append(left)
            if right <= max_start:
                candidates.append(right)
        pick = next((item for item in candidates if item not in used), None)
        if pick is None:
            break
        used.add(int(pick))
        selected.append(int(pick))
    return selected


def normalize_image_ext(value: str) -> str:
    ext = str(value or "").strip().lower()
    if not ext:
        return ""
    return ext if ext.startswith(".") else f".{ext}"


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(str(src), str(dst))
    except FileExistsError:
        raise
    except OSError:
        shutil.copy2(src, dst)


def read_image_size(image_path: Path) -> Tuple[int, int]:
    """Read PNG/JPEG dimensions without adding an image library dependency."""
    path = Path(image_path)
    with path.open("rb") as f:
        header = f.read(24)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            width, height = struct.unpack(">II", header[16:24])
            return int(width), int(height)

        f.seek(0)
        if f.read(2) != b"\xff\xd8":
            raise ValueError(f"Unsupported image format for resolution check: {path}")

        sof_markers = {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }
        while True:
            marker_prefix = f.read(1)
            while marker_prefix and marker_prefix != b"\xff":
                marker_prefix = f.read(1)
            if not marker_prefix:
                break

            marker = f.read(1)
            while marker == b"\xff":
                marker = f.read(1)
            if not marker:
                break

            code = marker[0]
            if code in (0xD8, 0xD9):
                continue

            length_bytes = f.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = struct.unpack(">H", length_bytes)[0]
            if segment_length < 2:
                raise ValueError(f"Invalid JPEG segment length in {path}")

            if code in sof_markers:
                segment = f.read(segment_length - 2)
                if len(segment) < 5:
                    break
                height, width = struct.unpack(">HH", segment[1:5])
                return int(width), int(height)

            f.seek(segment_length - 2, os.SEEK_CUR)

    raise ValueError(f"Could not read image resolution from {path}")


def resolution_for_frame(payload: Dict[str, Any], frame_name: str) -> Tuple[int, int]:
    return read_image_size(Path(payload["images"][frame_name]))


def materialize_pose_tuple(
    tuple_root: Path,
    image_paths: Sequence[Path],
    poses_c2w: Sequence[np.ndarray],
    copy_images: bool,
    output_image_ext: str,
    metadata: Dict[str, Any],
) -> None:
    color_dir = tuple_root / "color_90"
    color_dir.mkdir(parents=True, exist_ok=False)
    pose_lines: List[str] = []
    suffix_override = normalize_image_ext(output_image_ext)
    for out_idx, (image_path, pose) in enumerate(zip(image_paths, poses_c2w)):
        suffix = suffix_override or Path(image_path).suffix.lower() or ".jpg"
        link_or_copy(Path(image_path), color_dir / f"frame_{out_idx:04d}{suffix}", copy_images=copy_images)
        pose_lines.append(" ".join(f"{float(value):.8f}" for value in np.asarray(pose).reshape(-1)))
    (tuple_root / "pose_90.txt").write_text("\n".join(pose_lines) + "\n", encoding="utf-8")
    (tuple_root / "tuple_meta.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_scene_payload(scene: SceneSpec) -> Dict[str, Any]:
    frames = available_frame_names(scene)
    return {
        "scene": scene,
        "frames": frames,
        "images": image_map(scene),
        "cameras": load_scene_cameras(scene),
    }


def scene_with_camera(scene: SceneSpec, camera: str) -> SceneSpec:
    return SceneSpec(
        dataset=scene.dataset,
        name=scene.name,
        root=scene.root,
        camera=str(camera),
        image_exts=scene.image_exts,
        has_depth=scene.has_depth,
        depth_ext=scene.depth_ext,
    )


def select_stride_frame_names(frames: Sequence[str], start: int, count: int, stride: int) -> List[str]:
    indices = [int(start) + idx * int(stride) for idx in range(int(count))]
    if not indices or indices[-1] >= len(frames):
        raise ValueError("stride selection exceeds frame list")
    return [frames[index] for index in indices]


def select_fixed_camera_distractor(
    target_payload: Dict[str, Any],
    fixed_camera: str,
    desired_frame_names: Sequence[str],
    start: int,
    eval_views: int,
    noise_views: int,
    stride: int,
    require_frame_match: bool = False,
) -> Tuple[Dict[str, Any], List[str], int, bool]:
    scene = target_payload["scene"]
    distractor_payload = load_scene_payload(scene_with_camera(scene, fixed_camera))
    available = set(distractor_payload["frames"])
    if all(frame_name in available for frame_name in desired_frame_names):
        return distractor_payload, list(desired_frame_names), int(start) + int(eval_views) * int(stride), True

    if bool(require_frame_match):
        missing = [frame_name for frame_name in desired_frame_names if frame_name not in available]
        raise ValueError(
            "fixed distractor camera is missing clean tail frame names: "
            f"camera={fixed_camera}, missing={missing[:4]}"
        )

    distractor_start = int(start) + int(eval_views) * int(stride)
    distractor_frame_names = select_stride_frame_names(
        distractor_payload["frames"],
        distractor_start,
        int(noise_views),
        int(stride),
    )
    return distractor_payload, distractor_frame_names, int(distractor_start), False


def find_distractor_selection(
    scene_payloads: Sequence[Dict[str, Any]],
    target_scene_index: int,
    anchor_index: int,
    quantile: float,
    noise_views: int,
    stride: int,
    scene_offset: int,
    target_camera: str,
    target_resolution: Tuple[int, int] | None,
    match_resolution: bool,
    match_camera: bool,
) -> Tuple[Dict[str, Any], List[str], int]:
    if len(scene_payloads) < 2:
        raise ValueError("at least two scenes are required to pick distractor frames")
    num_scenes = len(scene_payloads)
    for shift in range(num_scenes):
        source_index = (target_scene_index + int(scene_offset) + int(anchor_index) + shift) % num_scenes
        if source_index == target_scene_index:
            continue
        source = scene_payloads[source_index]
        if match_camera and str(source["scene"].camera) != str(target_camera):
            continue
        starts = quantile_starts(
            num_frames=len(source["frames"]),
            total_views=int(noise_views),
            stride=int(stride),
            max_anchors=1,
            quantiles=[float(quantile)],
        )
        if not starts:
            continue
        frame_names = select_stride_frame_names(source["frames"], starts[0], noise_views, stride)
        if match_resolution:
            if target_resolution is None:
                raise ValueError("target resolution is required when match_resolution=True")
            candidate_resolutions = {resolution_for_frame(source, frame_name) for frame_name in frame_names}
            if candidate_resolutions != {target_resolution}:
                continue
        return source, frame_names, int(starts[0])
    constraints = []
    if match_camera:
        constraints.append(f"camera={target_camera}")
    if match_resolution:
        constraints.append(f"resolution={target_resolution}")
    suffix = f" matching {' and '.join(constraints)}" if constraints else ""
    raise ValueError(f"no valid distractor scene with enough frames{suffix}")


def pose_for_frame(payload: Dict[str, Any], frame_name: str) -> np.ndarray:
    return pose3x4_to_c2w(np.asarray(payload["cameras"][frame_name].RT))


def paths_and_poses_for_frames(payload: Dict[str, Any], frame_names: Sequence[str]) -> Tuple[List[Path], List[np.ndarray]]:
    return [payload["images"][name] for name in frame_names], [pose_for_frame(payload, name) for name in frame_names]


def scene_slug(dataset: str, scene: SceneSpec) -> str:
    return f"{dataset}-{scene.name.replace('/', '-')}-cam{scene.camera}"


def build_scene_tuples(
    dataset: str,
    scene_index: int,
    scene_payloads: Sequence[Dict[str, Any]],
    output_root: Path,
    args: argparse.Namespace,
    anchor_quantiles: Sequence[float],
) -> Dict[str, Any]:
    payload = scene_payloads[scene_index]
    scene = payload["scene"]
    frames = payload["frames"]
    starts = quantile_starts(
        num_frames=len(frames),
        total_views=args.total_views,
        stride=args.frame_stride,
        max_anchors=args.max_anchors_per_scene,
        quantiles=anchor_quantiles,
    )

    rows: List[Dict[str, Any]] = []
    clean_count = 0
    noise_count = 0
    for anchor_index, start in enumerate(starts):
        quantile = float(anchor_quantiles[min(anchor_index, len(anchor_quantiles) - 1)])
        clean_frame_names = select_stride_frame_names(frames, start, args.total_views, args.frame_stride)
        eval_frame_names = clean_frame_names[: int(args.eval_views)]
        target_resolutions = {resolution_for_frame(payload, frame_name) for frame_name in eval_frame_names}
        if len(target_resolutions) != 1:
            rows.append(
                {
                    "start": int(start),
                    "skipped": f"target eval frames have mixed resolutions: {sorted(target_resolutions)}",
                }
            )
            continue
        target_resolution = next(iter(target_resolutions))
        fixed_distractor_camera = str(getattr(args, "fixed_distractor_camera", "") or "").strip()
        distractor_frame_names_exact_match = False
        try:
            if fixed_distractor_camera:
                (
                    distractor_payload,
                    distractor_frame_names,
                    distractor_start,
                    distractor_frame_names_exact_match,
                ) = select_fixed_camera_distractor(
                    target_payload=payload,
                    fixed_camera=fixed_distractor_camera,
                    desired_frame_names=clean_frame_names[int(args.eval_views) :],
                    start=int(start),
                    eval_views=int(args.eval_views),
                    noise_views=int(args.noise_views),
                    stride=int(args.frame_stride),
                    require_frame_match=bool(getattr(args, "require_fixed_distractor_frame_match", False)),
                )
                if bool(args.match_distractor_resolution):
                    candidate_resolutions = {
                        resolution_for_frame(distractor_payload, frame_name) for frame_name in distractor_frame_names
                    }
                    if candidate_resolutions != {target_resolution}:
                        raise ValueError(
                            "fixed distractor camera has resolution mismatch: "
                            f"target={target_resolution}, distractor={sorted(candidate_resolutions)}"
                        )
            else:
                distractor_payload, distractor_frame_names, distractor_start = find_distractor_selection(
                    scene_payloads=scene_payloads,
                    target_scene_index=scene_index,
                    anchor_index=anchor_index,
                    quantile=quantile,
                    noise_views=args.noise_views,
                    stride=args.frame_stride,
                    scene_offset=args.distractor_scene_offset,
                    target_camera=str(scene.camera),
                    target_resolution=target_resolution,
                    match_resolution=bool(args.match_distractor_resolution),
                    match_camera=bool(args.match_distractor_camera),
                )
        except ValueError as exc:
            rows.append({"start": int(start), "skipped": str(exc)})
            continue

        noise_frame_names = [*eval_frame_names, *distractor_frame_names]
        if len(noise_frame_names) != int(args.total_views):
            raise RuntimeError(f"Noise tuple length mismatch: {len(noise_frame_names)} != {args.total_views}")

        clean_images, clean_poses = paths_and_poses_for_frames(payload, clean_frame_names)
        prefix_images, prefix_poses = paths_and_poses_for_frames(payload, eval_frame_names)
        distractor_images, distractor_poses = paths_and_poses_for_frames(distractor_payload, distractor_frame_names)
        noise_images = [*prefix_images, *distractor_images]
        noise_poses = [*prefix_poses, *distractor_poses]

        base = f"{scene_slug(dataset, scene)}__anchor{int(start):04d}"
        tuple_specs = [
            ("clean", clean_frame_names, clean_images, clean_poses),
            ("noise", noise_frame_names, noise_images, noise_poses),
        ]
        for tuple_kind, tuple_frames, tuple_images, tuple_poses in tuple_specs:
            tuple_name = f"{base}__{tuple_kind}"
            metadata = {
                "protocol": "three_dataset_stride4_distractor_v1",
                "dataset": dataset,
                "tuple_kind": tuple_kind,
                "tuple_name": tuple_name,
                "target_scene_name": scene.name,
                "target_camera": scene.camera,
                "target_source_root": str(scene.root),
                "anchor_start": int(start),
                "frame_stride": int(args.frame_stride),
                "eval_frame_indices": list(range(int(args.eval_views))),
                "eval_frame_names": list(eval_frame_names),
                "ordered_frame_names": list(tuple_frames),
                "distractor_scene_name": distractor_payload["scene"].name if tuple_kind == "noise" else "",
                "distractor_camera": distractor_payload["scene"].camera if tuple_kind == "noise" else "",
                "distractor_source_root": str(distractor_payload["scene"].root) if tuple_kind == "noise" else "",
                "distractor_start": int(distractor_start) if tuple_kind == "noise" else None,
                "distractor_frame_names": list(distractor_frame_names) if tuple_kind == "noise" else [],
                "target_resolution": list(target_resolution),
                "distractor_resolution": list(resolution_for_frame(distractor_payload, distractor_frame_names[0]))
                if tuple_kind == "noise"
                else [],
                "match_distractor_resolution": bool(args.match_distractor_resolution),
                "match_distractor_camera": bool(args.match_distractor_camera),
                "fixed_distractor_camera": fixed_distractor_camera,
                "require_fixed_distractor_frame_match": bool(
                    getattr(args, "require_fixed_distractor_frame_match", False)
                ),
                "distractor_frames_exact_match": bool(distractor_frame_names_exact_match),
            }
            materialize_pose_tuple(
                tuple_root=output_root / tuple_name,
                image_paths=tuple_images,
                poses_c2w=tuple_poses,
                copy_images=bool(args.copy_images),
                output_image_ext=str(args.output_image_ext),
                metadata=metadata,
            )
            if tuple_kind == "clean":
                clean_count += 1
            else:
                noise_count += 1

        rows.append(
            {
                "start": int(start),
                "quantile": quantile,
                "clean_frames": list(clean_frame_names),
                "noise_frames": list(noise_frame_names),
                "eval_frames": list(eval_frame_names),
                "distractor_scene_name": distractor_payload["scene"].name,
                "distractor_camera": distractor_payload["scene"].camera,
                "distractor_start": int(distractor_start),
                "distractor_frames": list(distractor_frame_names),
                "target_resolution": list(target_resolution),
                "distractor_resolution": list(resolution_for_frame(distractor_payload, distractor_frame_names[0])),
                "distractor_frames_exact_match": bool(distractor_frame_names_exact_match),
            }
        )

    return {
        "dataset": dataset,
        "scene_name": scene.name,
        "camera": scene.camera,
        "source_root": str(scene.root),
        "num_frames": int(len(frames)),
        "num_clean_tuples": int(clean_count),
        "num_noise_tuples": int(noise_count),
        "anchors": rows,
    }


def build_dataset(dataset: str, output_root: Path, args: argparse.Namespace, anchor_quantiles: Sequence[float]) -> Path:
    dataset_output_root = output_root / dataset
    dataset_output_root.mkdir(parents=True, exist_ok=True)
    scenes = discover_dataset_scenes(dataset, args)
    scene_payloads = [load_scene_payload(scene) for scene in scenes]
    scene_rows = [
        build_scene_tuples(
            dataset=dataset,
            scene_index=scene_index,
            scene_payloads=scene_payloads,
            output_root=dataset_output_root,
            args=args,
            anchor_quantiles=anchor_quantiles,
        )
        for scene_index in range(len(scene_payloads))
    ]
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "three_dataset_stride4_distractor_v1",
        "dataset": dataset,
        "output_root": str(dataset_output_root),
        "config": {
            "total_views": int(args.total_views),
            "eval_views": int(args.eval_views),
            "noise_views": int(args.noise_views),
            "frame_stride": int(args.frame_stride),
            "distractor_scene_offset": int(args.distractor_scene_offset),
            "match_distractor_resolution": bool(args.match_distractor_resolution),
            "match_distractor_camera": bool(args.match_distractor_camera),
            "fixed_distractor_camera": str(args.fixed_distractor_camera or ""),
            "require_fixed_distractor_frame_match": bool(args.require_fixed_distractor_frame_match),
            "max_anchors_per_scene": int(args.max_anchors_per_scene),
            "anchor_quantiles": [float(item) for item in anchor_quantiles],
            "copy_images": bool(args.copy_images),
            "output_image_ext": normalize_image_ext(str(args.output_image_ext)) or "source",
            "vkitti_scene_specs": str(args.vkitti_scene_specs),
        },
        "num_scenes_scanned": int(len(scene_rows)),
        "num_scenes_kept": int(sum(1 for row in scene_rows if int(row["num_clean_tuples"]) > 0)),
        "num_clean_tuples": int(sum(int(row["num_clean_tuples"]) for row in scene_rows)),
        "num_noise_tuples": int(sum(int(row["num_noise_tuples"]) for row in scene_rows)),
        "scenes": scene_rows,
    }
    summary_path = dataset_output_root / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    args = parse_args()
    if int(args.total_views) <= int(args.eval_views):
        raise ValueError("--total-views must be greater than --eval-views")
    if int(args.total_views) != int(args.eval_views) + int(args.noise_views):
        raise ValueError("--total-views must equal --eval-views + --noise-views")
    if int(args.frame_stride) < 1:
        raise ValueError("--frame-stride must be >= 1")

    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else default_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    anchor_quantiles = parse_anchor_quantiles(args.anchor_quantiles)
    summaries = {
        dataset: str(build_dataset(dataset, output_root, args, anchor_quantiles))
        for dataset in parse_datasets(args.datasets)
    }
    print(json.dumps({"output_root": str(output_root), "summaries": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
