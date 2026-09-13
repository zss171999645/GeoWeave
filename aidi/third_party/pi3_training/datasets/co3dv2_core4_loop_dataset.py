from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image

from aidi.utils.core4_main_val import (
    CO3DV2_RAW_ROOT_DEFAULT,
    _filter_co3dv2_frames_with_existing_images,
    _filter_co3dv2_seq_names_with_image_dirs,
    _load_selected_co3dv2_annotations,
    _select_by_stride,
    _stable_int_seed,
)
from datasets.base.base_dataset import BaseDataset
from datasets.co3dv2_dataset import opencv_from_cameras_projection


def _parse_categories(categories: str | Sequence[str], data_root: str) -> List[str]:
    if isinstance(categories, str):
        items = [item.strip() for item in categories.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in categories if str(item).strip()]
    if items:
        return items

    root = Path(data_root)
    return sorted(path.name for path in root.iterdir() if path.is_dir()) if root.is_dir() else []


def _legacy_np_choice(seed: int, size: int, num: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.choice(size, num, replace=False)


class CO3DV2Core4LoopDataset(BaseDataset):
    """CO3Dv2 subset that mirrors the native core4 CO3D validation sequence/frame selection.

    This dataset is meant for closed-loop diagnostics: train Pi3 on exactly the same
    CO3D raw root, set-list selection, category subset, per-sequence seed, and frame
    count used by core4 CO3D validation.
    """

    def __init__(
        self,
        data_root: str = CO3DV2_RAW_ROOT_DEFAULT,
        categories: str | Sequence[str] = "apple",
        split: str = "test",
        set_list_tag: str = "fewview_dev",
        selection_source: str = "co3d_setlists",
        min_quality: float = 0.5,
        min_num_images: int = 50,
        num_frames: int = 10,
        sample_stride: int = 10,
        max_sequences: int = 0,
        core4_seed: int = 0,
        mask_bg: bool | str = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.dataset_label = "CO3Dv2"
        self.data_root = str(data_root)
        self.categories = _parse_categories(categories, self.data_root)
        self.split = str(split)
        self.set_list_tag = str(set_list_tag)
        self.selection_source = str(selection_source)
        self.min_quality = float(min_quality)
        self.min_num_images = int(min_num_images)
        self.num_frames = int(num_frames)
        self.sample_stride = max(1, int(sample_stride))
        self.max_sequences = max(0, int(max_sequences))
        self.core4_seed = int(core4_seed)
        self.mask_bg = mask_bg

        if self.mask_bg not in (True, False, "rand"):
            raise ValueError(f"mask_bg must be one of True, False, 'rand', got {self.mask_bg!r}")

        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(
                "CO3DV2Core4LoopDataset found no samples. "
                f"data_root={self.data_root}, categories={self.categories}, split={self.split}"
            )

        print(
            f"[{self.dataset_label}Core4Loop] Built {len(self.samples)} sequences "
            f"from categories={self.categories}, sample_stride={self.sample_stride}, "
            f"max_sequences={self.max_sequences}, num_frames={self.num_frames}.",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _build_samples(self) -> List[Dict[str, Any]]:
        ds_cfg = {
            "selection_source": self.selection_source,
            "split": self.split,
            "setlist_root": self.data_root,
            "co3d_v2_dir": self.data_root,
            "set_list_tag": self.set_list_tag,
            "min_quality": self.min_quality,
        }
        selected_annotations = _load_selected_co3dv2_annotations(ds_cfg, self.categories)

        samples: List[Dict[str, Any]] = []
        for category in self.categories:
            annotation = selected_annotations.get(category)
            if not annotation:
                continue

            seq_names = _filter_co3dv2_seq_names_with_image_dirs(
                sorted(annotation.keys()),
                image_root=self.data_root,
                category=category,
            )
            seq_names = _select_by_stride(seq_names, self.sample_stride)
            if self.max_sequences > 0:
                seq_names = seq_names[: self.max_sequences]

            for seq_name in seq_names:
                seq_data = _filter_co3dv2_frames_with_existing_images(
                    annotation[seq_name],
                    image_root=self.data_root,
                )
                if len(seq_data) < max(self.min_num_images, self.num_frames):
                    continue

                seed = _stable_int_seed(self.core4_seed, category, seq_name)
                ids = _legacy_np_choice(seed, len(seq_data), self.num_frames)
                frames = [seq_data[int(idx)] for idx in ids]
                samples.append(
                    {
                        "category": category,
                        "seq_name": seq_name,
                        "frames": frames,
                        "selection_seed": seed,
                        "path": f"{category}/{seq_name}",
                    }
                )

        return samples

    def _select_runtime_frames(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        frames = list(sample["frames"])
        if self.frame_num == len(frames):
            return frames

        seed = _stable_int_seed(sample["selection_seed"], "runtime", self.frame_num)
        rng = np.random.RandomState(seed)
        replace = len(frames) < self.frame_num
        ids = rng.choice(len(frames), self.frame_num, replace=replace)
        return [frames[int(idx)] for idx in ids]

    def _get_views(self, index: int, resolution: Sequence[int], rng: np.random.Generator) -> List[Dict[str, Any]]:
        sample = self.samples[index]
        frames = self._select_runtime_frames(sample)

        mask_bg = self.mask_bg is True or (self.mask_bg == "rand" and bool(rng.choice(2)))
        self.this_views_info = {
            "path": sample["path"],
            "frame_numbers": [int(frame.get("frame_number", -1)) for frame in frames],
        }

        views: List[Dict[str, Any]] = []
        for frame in frames:
            image_path = os.path.join(self.data_root, frame["filepath"])
            depth_path = image_path.replace("/images/", "/depths/") + ".geometric.png"
            mask_path = image_path.replace("/images/", "/masks/").replace(".jpg", ".png")

            image = Image.open(image_path).convert("RGB")
            rgb_image = np.array(image)
            depthmap = self._load_depth(depth_path)

            if mask_bg and os.path.exists(mask_path):
                maskmap = np.array(Image.open(mask_path), dtype=np.float32)
                if maskmap.ndim == 3:
                    maskmap = maskmap[..., 0]
                depthmap *= (maskmap / 255.0) > 0.1

            image_size = np.array([rgb_image.shape[0], rgb_image.shape[1]])
            r_mat = np.array(frame["R"], dtype=np.float32)
            t_vec = np.array(frame["T"], dtype=np.float32)
            focal_length = np.array(frame["focal_length"], dtype=np.float32)
            principal_point = np.array(frame["principal_point"], dtype=np.float32)

            r_cv, t_cv, camera_intrinsics = opencv_from_cameras_projection(
                r_mat,
                t_vec,
                focal_length,
                principal_point,
                image_size,
            )
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3, :3] = r_cv
            camera_pose[:3, 3] = t_cv
            camera_pose = np.linalg.inv(camera_pose).astype(np.float32)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image,
                depthmap,
                camera_intrinsics.copy(),
                resolution,
                rng=rng,
                info=image_path,
            )

            views.append(
                {
                    "img": rgb_image,
                    "depthmap": depthmap.astype(np.float32),
                    "camera_pose": camera_pose,
                    "camera_intrinsics": intrinsics.astype(np.float32),
                    "dataset": self.dataset_label,
                    "label": sample["path"],
                    "instance": Path(image_path).name,
                }
            )

        return views

    @staticmethod
    def _load_depth(path: str) -> np.ndarray:
        raw = np.array(Image.open(path), dtype=np.uint16)
        depthmap = raw.view(np.float16).astype(np.float32)
        return np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
