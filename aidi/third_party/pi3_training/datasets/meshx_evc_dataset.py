import hashlib
import json
import os
import os.path as osp
import time
from collections import OrderedDict
from glob import glob
from typing import Dict, Iterable, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image

from datasets.base.base_dataset import BaseDataset
from easyvolcap.utils.easy_utils import read_camera


def _as_int(value, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be int-compatible, got {value!r}") from exc


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("1", "true", "yes", "on"):
            return True
        if value in ("0", "false", "no", "off", "", "none", "null"):
            return False
    return bool(value)


def _none_if_empty(value):
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    return value


class MeshXEvcDataset(BaseDataset):
    """Adapter for MeshX/EasyVolcap-style processed geometry datasets.

    A valid sequence is any directory below ``data_root`` containing:
      - intri.yml
      - extri.yml
      - images/
      - depths/

    The common layout is ``images/<frame_id>/<camera>.(jpg|png)`` and
    ``depths/<frame_id>/<camera>.(exr|npy|png)``. The scanner is recursive so
    the same adapter can cover BlendedMVS, HyperSim, MegaDepth, ScanNetPP,
    Taskonomy, WildRGBD, CO3Dv2 and TarTanAir exports without hardcoding their
    directory depth.
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        dataset_label: str = "MeshXEVC",
        verbose: bool = False,
        max_distance: int = 24,
        seq_num: int = -1,
        use_masks: bool = False,
        masks_dir: str = "masks",
        mask_threshold: float = 0.1,
        min_depth_quantile: float = 0.0,
        max_depth_quantile: float = 1.0,
        max_depth: float = 0.0,
        camera_ids: Optional[Iterable[str]] = None,
        use_camera_pkl_cache: bool = False,
        use_index_cache: bool = False,
        rebuild_index_cache: bool = False,
        index_cache_dir: Optional[str] = None,
        index_cache_wait_sec: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if data_root is None:
            raise ValueError("data_root is required for MeshXEvcDataset")
        seq_num = _as_int(seq_num, "seq_num")
        max_distance = _as_int(max_distance, "max_distance")
        index_cache_dir = _none_if_empty(index_cache_dir)

        self.data_root = data_root
        self.dataset_label = dataset_label
        self.verbose = verbose
        self.max_distance = max_distance
        self.use_masks = _as_bool(use_masks)
        self.masks_dir = str(masks_dir or "masks")
        self.mask_threshold = float(mask_threshold)
        self.min_depth_quantile = float(min_depth_quantile)
        self.max_depth_quantile = float(max_depth_quantile)
        self.max_depth = float(max_depth)
        self.camera_ids = self._normalize_camera_ids(camera_ids)
        self.use_camera_pkl_cache = _as_bool(use_camera_pkl_cache)
        self.use_index_cache = _as_bool(use_index_cache)
        self.rebuild_index_cache = _as_bool(rebuild_index_cache)
        self.index_cache_dir = index_cache_dir
        self.index_cache_wait_sec = int(index_cache_wait_sec)
        self.lazy_camera_load = os.environ.get("PI3_LAZY_SEQUENCE_INDEX", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

        self.sequences: List[str] = []
        self.frame_ids: Dict[str, List[str]] = {}
        self.num_imgs: Dict[str, int] = {}
        self.camera_files: Dict[str, Tuple[str, str]] = {}
        self.image_roots: Dict[str, str] = {}
        self.depth_roots: Dict[str, str] = {}
        self.mask_roots: Dict[str, str] = {}
        self.sequence_labels: Dict[str, str] = {}
        self.cameras = {}

        sequence_index = self._load_or_build_sequence_index(data_root, seq_num)
        for entry in sequence_index:
            spec = self._normalize_sequence_entry(entry)
            seq_dir = spec["path"]
            frames = spec["frames"]
            self.sequences.append(seq_dir)
            self.frame_ids[seq_dir] = frames
            self.num_imgs[seq_dir] = len(frames) if frames else -1
            self.camera_files[seq_dir] = (spec["intri_path"], spec["extri_path"])
            self.image_roots[seq_dir] = spec["image_root"]
            self.depth_roots[seq_dir] = spec["depth_root"]
            self.mask_roots[seq_dir] = spec["mask_root"]
            self.sequence_labels[seq_dir] = spec["label"]

        self.sequences = sorted(self.sequences)
        if seq_num > 0:
            self.sequences = self.sequences[:seq_num]
        if not self.lazy_camera_load:
            for seq_dir in self.sequences:
                self._get_cameras(seq_dir)

        if self.verbose:
            print(f"[{self.dataset_label}] sequences: {self.sequences}")
        print(f"[{self.dataset_label}] Found {len(self.sequences)} valid sequences in {data_root}", flush=True)

    def _load_or_build_sequence_index(self, data_root: str, seq_num: int) -> List[dict]:
        cache_path = self._index_cache_path(data_root)
        sequence_index = None
        if self.use_index_cache and not self.rebuild_index_cache:
            sequence_index = self._try_load_sequence_index(cache_path)
            if sequence_index is None and seq_num <= 0 and self._distributed_rank() > 0:
                sequence_index = self._wait_for_sequence_index(cache_path)

        if sequence_index is None:
            sequence_index = self._scan_sequence_index(data_root, seq_num)
            if self.use_index_cache and seq_num <= 0 and self._distributed_rank() == 0:
                self._write_sequence_index(cache_path, data_root, sequence_index)
        elif seq_num > 0:
            sequence_index = sequence_index[:seq_num]
        return sequence_index

    def _index_cache_path(self, data_root: str) -> str:
        cache_dir = self.index_cache_dir or osp.join(data_root, ".meshx_pi3_index_cache")
        digest_source = osp.abspath(data_root)
        if self.camera_ids is not None:
            digest_source = f"{digest_source}|camera_ids={','.join(sorted(self.camera_ids))}"
        digest = hashlib.md5(digest_source.encode("utf-8")).hexdigest()[:16]
        label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in self.dataset_label)
        return osp.join(cache_dir, f"{label}_{digest}_v2.json")

    @staticmethod
    def _normalize_camera_ids(camera_ids: Optional[Iterable[str]]) -> Optional[Set[str]]:
        if camera_ids is None:
            return None
        if isinstance(camera_ids, str):
            value = camera_ids.strip()
            if value.lower() in ("", "none", "null", "all", "*"):
                return None
            items = [item.strip() for item in value.split(",")]
        else:
            items = [str(item).strip() for item in camera_ids]
        normalized = {item for item in items if item}
        return normalized or None

    @staticmethod
    def _distributed_rank() -> int:
        for key in ("RANK", "LOCAL_RANK", "NODE_RANK"):
            value = os.environ.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except ValueError:
                continue
        return 0

    def _try_load_sequence_index(self, cache_path: str) -> Optional[List[dict]]:
        if not cache_path:
            return None
        candidate_paths = [cache_path]
        if not osp.isfile(cache_path):
            basename = osp.basename(cache_path)
            if basename.endswith("_v2.json"):
                digest = basename.rsplit("_", 2)[-2]
                candidate_paths.extend(sorted(glob(osp.join(osp.dirname(cache_path), f"*_{digest}_v2.json"))))
        try:
            for candidate_path in candidate_paths:
                if not osp.isfile(candidate_path):
                    continue
                with open(candidate_path, "r") as f:
                    payload = json.load(f)
                entries = payload.get("sequences", [])
                sequence_index = [self._normalize_sequence_entry(item) for item in entries]
                if sequence_index:
                    print(f"[{self.dataset_label}] Loaded sequence index cache: {candidate_path}", flush=True)
                    return sequence_index
        except Exception as exc:
            print(f"[{self.dataset_label}] Failed to load index cache {cache_path}: {exc}", flush=True)
        return None

    def _wait_for_sequence_index(self, cache_path: str) -> Optional[List[dict]]:
        if not cache_path or self.index_cache_wait_sec <= 0:
            return None
        deadline = time.time() + self.index_cache_wait_sec
        while time.time() < deadline:
            sequence_index = self._try_load_sequence_index(cache_path)
            if sequence_index is not None:
                return sequence_index
            time.sleep(10.0)
        print(f"[{self.dataset_label}] Timed out waiting for index cache: {cache_path}", flush=True)
        return None

    def _write_sequence_index(
        self,
        cache_path: str,
        data_root: str,
        sequence_index: List[dict],
    ) -> None:
        if not cache_path:
            return
        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        payload = dict(
            version=2,
            data_root=data_root,
            dataset_label=self.dataset_label,
            camera_ids=sorted(self.camera_ids) if self.camera_ids is not None else None,
            sequences=[self._normalize_sequence_entry(entry) for entry in sequence_index],
        )
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        try:
            with open(tmp_path, "w") as f:
                json.dump(payload, f)
            try:
                os.replace(tmp_path, cache_path)
            except OSError as exc:
                print(
                    f"[{self.dataset_label}] Atomic cache replace failed ({exc}); writing cache non-atomically",
                    flush=True,
                )
                with open(cache_path, "w") as f:
                    json.dump(payload, f)
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            print(f"[{self.dataset_label}] Wrote sequence index cache: {cache_path}", flush=True)
        except Exception as exc:
            print(f"[{self.dataset_label}] Failed to write index cache {cache_path}: {exc}", flush=True)

    def _scan_sequence_index(self, data_root: str, seq_num: int) -> List[dict]:
        sequence_index: List[dict] = []
        for spec in self._iter_sequence_specs(data_root, self.masks_dir, self.camera_ids):
            if spec.get("lazy_frames", False):
                frames = []
            else:
                frames = self._collect_frame_ids(spec["image_root"], spec["depth_root"])
                if not frames:
                    continue
            entry = dict(spec)
            entry["frames"] = frames
            sequence_index.append(entry)
            if seq_num > 0 and len(sequence_index) >= seq_num:
                break
        return sequence_index

    @staticmethod
    def _iter_sequence_specs(data_root: str, masks_dir: str, camera_ids: Optional[Set[str]] = None) -> Iterable[dict]:
        data_roots_path = osp.join(data_root, "data_roots.txt")
        data_roots_specs = []
        if osp.isfile(data_roots_path):
            try:
                with open(data_roots_path, "r", encoding="utf-8", errors="ignore") as f:
                    rel_roots = [line.strip() for line in f if line.strip()]
                if len(rel_roots) > 16:
                    data_roots_specs = MeshXEvcDataset._infer_sequence_specs_from_data_roots(
                        data_root,
                        rel_roots,
                        masks_dir,
                        camera_ids,
                    )
                else:
                    for rel_root in rel_roots:
                        seq_root = rel_root if osp.isabs(rel_root) else osp.join(data_root, rel_root)
                        data_roots_specs.extend(MeshXEvcDataset._iter_sequence_specs_at_root(
                            seq_root,
                            masks_dir,
                            camera_ids=camera_ids,
                            lazy_frames=True,
                        ))
                if data_roots_specs:
                    print(
                        f"[MeshXEvcDataset] Loaded {len(data_roots_specs)} sequence roots from {data_roots_path}",
                        flush=True,
                    )
                    yield from data_roots_specs
                    return
                print(
                    f"[MeshXEvcDataset] No valid sequence roots in {data_roots_path}; falling back to recursive scan",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[MeshXEvcDataset] Failed reading {data_roots_path}: {exc}; falling back to recursive scan",
                    flush=True,
                )

        for root, dirs, files in os.walk(data_root):
            yielded_specs = list(MeshXEvcDataset._iter_sequence_specs_at_root(root, masks_dir, camera_ids=camera_ids))
            if yielded_specs:
                yield from yielded_specs
                dirs[:] = []

    @staticmethod
    def _infer_sequence_specs_from_data_roots(
        data_root: str,
        rel_roots: List[str],
        masks_dir: str,
        camera_ids: Optional[Set[str]] = None,
    ) -> List[dict]:
        sample_root = None
        sample_specs: List[dict] = []
        for rel_root in rel_roots[:128]:
            root = rel_root if osp.isabs(rel_root) else osp.join(data_root, rel_root)
            sample_specs = MeshXEvcDataset._iter_sequence_specs_at_root(
                root,
                masks_dir,
                lazy_frames=True,
                camera_ids=camera_ids,
            )
            if sample_specs:
                sample_root = root
                break
        if sample_root is None:
            return []

        direct_layout = len(sample_specs) == 1 and sample_specs[0]["path"] == sample_root
        if direct_layout:
            specs = []
            for rel_root in rel_roots:
                root = rel_root if osp.isabs(rel_root) else osp.join(data_root, rel_root)
                specs.append(MeshXEvcDataset._make_sequence_entry(
                    path=root,
                    image_root=osp.join(root, "images"),
                    depth_root=osp.join(root, "depths"),
                    mask_root=osp.join(root, masks_dir),
                    intri_path=osp.join(root, "intri.yml"),
                    extri_path=osp.join(root, "extri.yml"),
                    label=root,
                    lazy_frames=True,
                ))
            return specs

        camera_ids = [osp.basename(spec["path"]) for spec in sample_specs]
        specs = []
        for rel_root in rel_roots:
            root = rel_root if osp.isabs(rel_root) else osp.join(data_root, rel_root)
            for camera_id in camera_ids:
                specs.append(MeshXEvcDataset._make_sequence_entry(
                    path=osp.join(root, "cameras", camera_id),
                    image_root=osp.join(root, "images", camera_id),
                    depth_root=osp.join(root, "depths", camera_id),
                    mask_root=osp.join(root, masks_dir, camera_id),
                    intri_path=osp.join(root, "cameras", camera_id, "intri.yml"),
                    extri_path=osp.join(root, "cameras", camera_id, "extri.yml"),
                    label=osp.join(root, camera_id),
                    lazy_frames=True,
                ))
        return specs

    @staticmethod
    def _iter_sequence_specs_at_root(
        root: str,
        masks_dir: str,
        lazy_frames: bool = False,
        camera_ids: Optional[Set[str]] = None,
    ) -> List[dict]:
        specs: List[dict] = []
        if (
            osp.isfile(osp.join(root, "intri.yml"))
            and osp.isfile(osp.join(root, "extri.yml"))
            and osp.isdir(osp.join(root, "images"))
            and osp.isdir(osp.join(root, "depths"))
        ):
            specs.append(MeshXEvcDataset._make_sequence_entry(
                path=root,
                image_root=osp.join(root, "images"),
                depth_root=osp.join(root, "depths"),
                mask_root=osp.join(root, masks_dir),
                intri_path=osp.join(root, "intri.yml"),
                extri_path=osp.join(root, "extri.yml"),
                label=root,
                lazy_frames=lazy_frames,
            ))
            return specs

        camera_root = osp.join(root, "cameras")
        image_root = osp.join(root, "images")
        depth_root = osp.join(root, "depths")
        if not (osp.isdir(camera_root) and osp.isdir(image_root) and osp.isdir(depth_root)):
            return specs

        for camera_id in sorted(os.listdir(camera_root)):
            if camera_ids is not None and camera_id not in camera_ids:
                continue
            camera_dir = osp.join(camera_root, camera_id)
            if not osp.isdir(camera_dir):
                continue
            intri_path = osp.join(camera_dir, "intri.yml")
            extri_path = osp.join(camera_dir, "extri.yml")
            cam_image_root = osp.join(image_root, camera_id)
            cam_depth_root = osp.join(depth_root, camera_id)
            if (
                osp.isfile(intri_path)
                and osp.isfile(extri_path)
                and osp.isdir(cam_image_root)
                and osp.isdir(cam_depth_root)
            ):
                specs.append(MeshXEvcDataset._make_sequence_entry(
                    path=camera_dir,
                    image_root=cam_image_root,
                    depth_root=cam_depth_root,
                    mask_root=osp.join(root, masks_dir, camera_id),
                    intri_path=intri_path,
                    extri_path=extri_path,
                    label=osp.join(root, camera_id),
                    lazy_frames=lazy_frames,
                ))
        return specs

    @staticmethod
    def _make_sequence_entry(
        path: str,
        image_root: str,
        depth_root: str,
        mask_root: str,
        intri_path: str,
        extri_path: str,
        label: str,
        frames: Optional[List[str]] = None,
        lazy_frames: bool = False,
    ) -> dict:
        return dict(
            path=path,
            image_root=image_root,
            depth_root=depth_root,
            mask_root=mask_root,
            intri_path=intri_path,
            extri_path=extri_path,
            label=label,
            frames=list(frames or []),
            lazy_frames=bool(lazy_frames),
        )

    def _normalize_sequence_entry(self, entry) -> dict:
        if isinstance(entry, dict):
            path = entry["path"]
            return self._make_sequence_entry(
                path=path,
                image_root=entry.get("image_root", osp.join(path, "images")),
                depth_root=entry.get("depth_root", osp.join(path, "depths")),
                mask_root=entry.get("mask_root", osp.join(path, self.masks_dir)),
                intri_path=entry.get("intri_path", osp.join(path, "intri.yml")),
                extri_path=entry.get("extri_path", osp.join(path, "extri.yml")),
                label=entry.get("label", path),
                frames=list(entry.get("frames", [])),
                lazy_frames=entry.get("lazy_frames", False),
            )
        seq_dir, frames = entry
        return self._make_sequence_entry(
            path=seq_dir,
            image_root=osp.join(seq_dir, "images"),
            depth_root=osp.join(seq_dir, "depths"),
            mask_root=osp.join(seq_dir, self.masks_dir),
            intri_path=osp.join(seq_dir, "intri.yml"),
            extri_path=osp.join(seq_dir, "extri.yml"),
            label=seq_dir,
            frames=list(frames),
            lazy_frames=False,
        )

    @staticmethod
    def _collect_frame_ids(img_root: str, depth_root: str) -> List[str]:
        img_frames = {d for d in os.listdir(img_root) if osp.isdir(osp.join(img_root, d))}
        depth_frames = {d for d in os.listdir(depth_root) if osp.isdir(osp.join(depth_root, d))}
        nested = sorted(img_frames & depth_frames)
        if nested:
            return nested

        img_stems = {
            osp.splitext(f)[0]
            for f in os.listdir(img_root)
            if osp.isfile(osp.join(img_root, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        depth_stems = {
            osp.splitext(f)[0]
            for f in os.listdir(depth_root)
            if osp.isfile(osp.join(depth_root, f)) and f.lower().endswith((".exr", ".npy", ".png"))
        }
        return sorted(img_stems & depth_stems)

    def __len__(self):
        return len(self.sequences)

    @staticmethod
    def _pick_file(root: str, frame_id: str, preferred_exts: Tuple[str, ...]) -> Optional[str]:
        frame_dir = osp.join(root, frame_id)
        if osp.isdir(frame_dir):
            files = sorted([f for f in os.listdir(frame_dir) if osp.isfile(osp.join(frame_dir, f))])
            for ext in preferred_exts:
                for filename in files:
                    if filename.lower().endswith(ext):
                        return osp.join(frame_dir, filename)
            return osp.join(frame_dir, files[0]) if files else None

        for ext in preferred_exts:
            candidate = osp.join(root, f"{frame_id}{ext}")
            if osp.isfile(candidate):
                return candidate
        return None

    @staticmethod
    def _w2c_to_c2w(R: np.ndarray, T: np.ndarray) -> np.ndarray:
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R.astype(np.float32)
        w2c[:3, 3] = T.reshape(3).astype(np.float32)
        return np.linalg.inv(w2c).astype(np.float32)

    @staticmethod
    def _lookup_camera(cameras, frame_id: str):
        if frame_id in cameras:
            return cameras[frame_id]
        try:
            numeric = f"{int(frame_id):06d}"
        except ValueError:
            numeric = None
        if numeric is not None and numeric in cameras:
            return cameras[numeric]
        raise KeyError(f"Frame {frame_id} not found in camera dict")

    def _get_cameras(self, seq_dir: str):
        if seq_dir not in self.cameras:
            intri_path, extri_path = self.camera_files[seq_dir]
            self.cameras[seq_dir] = read_camera(
                intri_path,
                extri_path,
                use_pkl=self.use_camera_pkl_cache,
            )
        return self.cameras[seq_dir]

    def _get_frame_ids(self, seq_dir: str) -> List[str]:
        frames = self.frame_ids.get(seq_dir, [])
        if not frames:
            frames = self._collect_frame_ids(
                self.image_roots.get(seq_dir, osp.join(seq_dir, "images")),
                self.depth_roots.get(seq_dir, osp.join(seq_dir, "depths")),
            )
            if not frames:
                raise FileNotFoundError(f"No image/depth frame pairs found in {seq_dir}")
            self.frame_ids[seq_dir] = frames
            self.num_imgs[seq_dir] = len(frames)
        return frames

    def _sample_indices(self, rng, num_imgs: int) -> List[int]:
        if self.frame_num > 20 and rng.random() < self.random_sample_thres:
            return list(rng.choice(num_imgs, size=self.frame_num, replace=num_imgs < self.frame_num))

        idxs = [int(rng.integers(0, num_imgs))]
        max_distance = int(self.max_distance / 8 * self.frame_num)
        start_idx = max(0, idxs[-1] - max_distance)
        end_idx = min(num_imgs - 1, start_idx + 2 * max_distance)
        start_idx = max(0, end_idx - 2 * max_distance)
        valid_indices = np.arange(start_idx, end_idx + 1)
        idxs.extend(list(rng.choice(valid_indices, self.frame_num - 1, replace=len(valid_indices) < self.frame_num - 1)))
        return [int(i) for i in idxs]

    @staticmethod
    def _read_depth(depth_path: str) -> np.ndarray:
        if depth_path.lower().endswith(".npy"):
            return np.load(depth_path).astype(np.float32)

        depthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depthmap is None:
            raise ValueError(f"Failed to read depth map: {depth_path}")
        if depthmap.ndim == 3:
            depthmap = depthmap[..., 0]
        depthmap = depthmap.astype(np.float32)
        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
        if depth_path.lower().endswith(".png"):
            depthmap = depthmap / 1000.0
        return depthmap

    @staticmethod
    def _read_mask(mask_path: str, threshold: float) -> np.ndarray:
        if mask_path.lower().endswith(".npy"):
            mask = np.load(mask_path)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to read mask map: {mask_path}")
            mask = mask.astype(np.float32) / 255.0
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask.astype(np.float32) > threshold

    def _apply_mask(self, depthmap: np.ndarray, mask_path: Optional[str]) -> np.ndarray:
        if not self.use_masks:
            return depthmap
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for masked dataset {self.dataset_label}")
        mask = self._read_mask(mask_path, self.mask_threshold)
        if mask.shape != depthmap.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (depthmap.shape[1], depthmap.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return np.where(mask, depthmap, 0.0).astype(np.float32)

    def _apply_depth_filters(self, depthmap: np.ndarray) -> np.ndarray:
        depthmap = depthmap.astype(np.float32, copy=True)
        valid = np.isfinite(depthmap) & (depthmap > 0.0)
        if self.max_depth > 0:
            depthmap[valid & (depthmap > self.max_depth)] = 0.0
            valid = np.isfinite(depthmap) & (depthmap > 0.0)
        if valid.any() and (self.min_depth_quantile > 0.0 or self.max_depth_quantile < 1.0):
            values = depthmap[valid]
            if self.min_depth_quantile > 0.0:
                min_depth = float(np.quantile(values, self.min_depth_quantile))
                depthmap[valid & (depthmap < min_depth)] = 0.0
            valid = np.isfinite(depthmap) & (depthmap > 0.0)
            if valid.any() and self.max_depth_quantile < 1.0:
                max_depth = float(np.quantile(depthmap[valid], self.max_depth_quantile))
                depthmap[valid & (depthmap > max_depth)] = 0.0
        return depthmap

    def _get_views(self, index, resolution, rng):
        seq_dir = self.sequences[index]
        img_root = self.image_roots.get(seq_dir, osp.join(seq_dir, "images"))
        depth_root = self.depth_roots.get(seq_dir, osp.join(seq_dir, "depths"))
        mask_root = self.mask_roots.get(seq_dir, osp.join(seq_dir, self.masks_dir))
        frames = self._get_frame_ids(seq_dir)
        cameras = self._get_cameras(seq_dir)

        sampled = self._sample_indices(rng, len(frames))
        self.this_views_info = dict(
            scene=osp.relpath(seq_dir, self.data_root),
            sampled=sampled,
        )

        views = []
        for idx in sampled:
            frame_id = frames[idx]
            image_path = self._pick_file(img_root, frame_id, (".jpg", ".jpeg", ".png"))
            depth_path = self._pick_file(depth_root, frame_id, (".exr", ".npy", ".png"))
            mask_path = self._pick_file(mask_root, frame_id, (".png", ".jpg", ".jpeg", ".npy")) if self.use_masks else None
            if image_path is None or depth_path is None:
                raise FileNotFoundError(f"Missing image/depth for frame {frame_id} in {seq_dir}")
            if self.use_masks and mask_path is None:
                raise FileNotFoundError(f"Missing mask for frame {frame_id} in {seq_dir}")

            cam = self._lookup_camera(cameras, frame_id)
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
            depthmap = self._read_depth(depth_path)
            depthmap = self._apply_mask(depthmap, mask_path)
            depthmap = self._apply_depth_filters(depthmap)
            camera_pose = self._w2c_to_c2w(cam.R, cam.T)
            intrinsics = cam.K.astype(np.float32)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image,
                depthmap,
                intrinsics,
                resolution,
                rng=rng,
                info=image_path,
            )

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset=self.dataset_label,
                label=osp.relpath(self.sequence_labels.get(seq_dir, seq_dir), self.data_root),
                instance=str(frame_id),
            ))

        return views


class MeshXBusinessMultiviewDataset(BaseDataset):
    """Business scene-level multi-camera adapter for native Pi3 training.

    This keeps Pi3's native dynamic sampler contract: the sampler passes a
    requested view count through ``frame_num`` and the dataset returns exactly
    that many symmetric views. Unlike ``MeshXEvcDataset``, one sample is a
    business scene/subscene root and views are sampled across camera directories
    and nearby frame ids. There is no target/source role in the returned views.
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        data_roots_file: str = "data_roots.txt",
        dataset_label: str = "BusinessMultiview",
        verbose: bool = False,
        max_distance: int = 24,
        seq_num: int = -1,
        camera_ids: Optional[Iterable[str]] = None,
        min_cameras: int = 2,
        intri_file: str = "intri.yml",
        extri_file: str = "extri.yml",
        pose_prior_extri_file: Optional[str] = None,
        pose_prior_required: bool = False,
        use_masks: bool = False,
        masks_dir: str = "masks",
        mask_threshold: float = 0.1,
        min_depth_quantile: float = 0.0,
        max_depth_quantile: float = 1.0,
        max_depth: float = 0.0,
        use_camera_pkl_cache: bool = False,
        use_index_cache: bool = False,
        rebuild_index_cache: bool = False,
        index_cache_dir: Optional[str] = None,
        index_cache_wait_sec: int = 0,
        camera_cache_size: int = 0,
        vggt_val_protocol: bool = False,
        vggt_val_extra_src_pool: int = 5,
        vggt_val_frame_sample: Optional[Iterable[Optional[int]]] = None,
        vggt_val_target_metrics_file: Optional[str] = None,
        vggt_val_target_list_file: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if data_root is None:
            raise ValueError("data_root is required for MeshXBusinessMultiviewDataset")
        seq_num = _as_int(seq_num, "seq_num")
        max_distance = _as_int(max_distance, "max_distance")
        index_cache_dir = _none_if_empty(index_cache_dir)

        self.data_root = data_root
        self.data_roots_file = str(data_roots_file or "data_roots.txt")
        self.dataset_label = dataset_label
        self.verbose = verbose
        self.max_distance = max_distance
        self.camera_ids = MeshXEvcDataset._normalize_camera_ids(camera_ids)
        self.min_cameras = max(1, _as_int(min_cameras, "min_cameras"))
        self.intri_file = str(intri_file or "intri.yml")
        self.extri_file = str(extri_file or "extri.yml")
        self.pose_prior_extri_file = _none_if_empty(pose_prior_extri_file)
        self.pose_prior_required = _as_bool(pose_prior_required)
        self.use_masks = _as_bool(use_masks)
        self.masks_dir = str(masks_dir or "masks")
        self.mask_threshold = float(mask_threshold)
        self.min_depth_quantile = float(min_depth_quantile)
        self.max_depth_quantile = float(max_depth_quantile)
        self.max_depth = float(max_depth)
        self.use_camera_pkl_cache = _as_bool(use_camera_pkl_cache)
        self.use_index_cache = _as_bool(use_index_cache)
        self.rebuild_index_cache = _as_bool(rebuild_index_cache)
        self.index_cache_dir = index_cache_dir
        self.index_cache_wait_sec = int(index_cache_wait_sec)
        self.camera_cache_size = max(0, _as_int(camera_cache_size, "camera_cache_size"))
        self.vggt_val_protocol = _as_bool(vggt_val_protocol)
        self.vggt_val_extra_src_pool = max(0, _as_int(vggt_val_extra_src_pool, "vggt_val_extra_src_pool"))
        self.vggt_val_frame_sample = self._normalize_frame_sample(vggt_val_frame_sample)
        self.vggt_val_target_metrics_file = _none_if_empty(vggt_val_target_metrics_file)
        self.vggt_val_target_list_file = _none_if_empty(vggt_val_target_list_file)

        self.scenes: List[str] = []
        self.scene_specs: Dict[str, dict] = {}
        self.val_samples: List[Tuple[str, str, str]] = []
        self.cameras: Dict[Tuple[str, str], object] = {}
        self._camera_cache_lru: OrderedDict[Tuple[str, str], None] = OrderedDict()

        scene_index = self._load_or_build_scene_index(data_root, seq_num)
        for entry in scene_index:
            spec = self._normalize_scene_entry(entry)
            self.scenes.append(spec["path"])
            self.scene_specs[spec["path"]] = spec

        self.scenes = sorted(self.scenes)
        if seq_num > 0:
            self.scenes = self.scenes[:seq_num]

        if self.vggt_val_protocol:
            self.val_samples = self._build_vggt_val_samples()

        if self.verbose:
            print(f"[{self.dataset_label}] scenes: {self.scenes}")
        print(f"[{self.dataset_label}] Found {len(self.scenes)} valid multiview scenes in {data_root}", flush=True)

    def __len__(self):
        if self.vggt_val_protocol:
            return len(self.val_samples)
        return len(self.scenes)

    @staticmethod
    def _normalize_frame_sample(frame_sample: Optional[Iterable[Optional[int]]]) -> List[Optional[int]]:
        if frame_sample is None:
            return [0, None, 1500]
        values = list(frame_sample)
        if len(values) != 3:
            raise ValueError(f"vggt_val_frame_sample must have 3 items, got {values!r}")
        start = 0 if values[0] is None else int(values[0])
        stop = None if values[1] is None else int(values[1])
        step = 1 if values[2] is None else int(values[2])
        if step <= 0:
            raise ValueError(f"vggt_val_frame_sample step must be positive, got {values!r}")
        return [start, stop, step]

    @staticmethod
    def _normalize_vggt_camera_id(value) -> str:
        if value is None:
            return "00"
        value = str(value)
        if value.isdigit():
            return f"{int(value):02d}"
        return value

    @staticmethod
    def _normalize_vggt_frame_id(value) -> str:
        if isinstance(value, (int, np.integer)):
            return f"{int(value):05d}"
        value = str(value)
        stem = osp.splitext(osp.basename(value))[0]
        if stem.isdigit():
            return f"{int(stem):05d}"
        return stem

    def _scene_path_from_vggt_target(self, path: str) -> str:
        path = str(path)
        if not osp.isabs(path):
            path = osp.join(self.data_root, path)
        if f"{osp.sep}images{osp.sep}" in path:
            parts = path.split(osp.sep)
            image_idx = parts.index("images")
            path = osp.sep.join(parts[:image_idx])
        return osp.realpath(path)

    def _parse_vggt_target_image_path(self, image_path: str) -> Tuple[str, str, str]:
        image_path = str(image_path).strip()
        if not image_path:
            raise ValueError("empty VGGT target image path")
        path = image_path if osp.isabs(image_path) else osp.join(self.data_root, image_path)
        parts = path.split(osp.sep)
        if "images" not in parts:
            raise ValueError(f"target image path must contain /images/<camera>/<frame>: {image_path}")
        image_idx = parts.index("images")
        if image_idx + 2 >= len(parts):
            raise ValueError(f"target image path must contain camera and frame: {image_path}")
        scene_path = osp.realpath(osp.sep.join(parts[:image_idx]))
        camera_id = self._normalize_vggt_camera_id(parts[image_idx + 1])
        frame_id = self._normalize_vggt_frame_id(parts[image_idx + 2])
        return scene_path, camera_id, frame_id

    def _load_vggt_val_targets(self) -> Optional[List[Tuple[str, str, str]]]:
        target_file = self.vggt_val_target_metrics_file or self.vggt_val_target_list_file
        if not target_file:
            return None
        target_file = osp.expanduser(str(target_file))
        if not osp.isfile(target_file):
            raise FileNotFoundError(f"VGGT val target file does not exist: {target_file}")

        targets: List[Tuple[str, str, str]] = []
        if self.vggt_val_target_metrics_file:
            with open(target_file, "r", encoding="utf-8", errors="ignore") as f:
                payload = json.load(f)
            metrics = payload.get("metrics", payload) if isinstance(payload, dict) else payload
            if not isinstance(metrics, list):
                raise ValueError(f"VGGT metrics target file must contain a metrics list: {target_file}")
            for item in metrics:
                if not isinstance(item, dict) or "path" not in item or "frame" not in item:
                    continue
                scene_path = self._scene_path_from_vggt_target(item["path"])
                camera_id = self._normalize_vggt_camera_id(item.get("camera", 0))
                frame_id = self._normalize_vggt_frame_id(item["frame"])
                targets.append((scene_path, camera_id, frame_id))
        else:
            with open(target_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    targets.append(self._parse_vggt_target_image_path(line))

        return targets

    def _build_vggt_val_samples(self) -> List[Tuple[str, str, str]]:
        target_samples = self._load_vggt_val_targets()
        if target_samples is not None:
            scene_paths = {osp.realpath(path): path for path in self.scene_specs.keys()}
            samples: List[Tuple[str, str, str]] = []
            skipped = 0
            for scene_path, camera_id, frame_id in target_samples:
                canonical_scene_path = scene_paths.get(osp.realpath(scene_path))
                if canonical_scene_path is None:
                    skipped += 1
                    continue
                if camera_id not in self.scene_specs[canonical_scene_path]["camera_specs"]:
                    skipped += 1
                    continue
                samples.append((canonical_scene_path, camera_id, frame_id))
            if not samples:
                raise ValueError(
                    f"No VGGT val targets from {self.vggt_val_target_metrics_file or self.vggt_val_target_list_file} "
                    f"matched scenes under {self.data_root}"
                )
            print(
                f"[{self.dataset_label}] Built {len(samples)} VGGT-style val samples "
                f"from target file {self.vggt_val_target_metrics_file or self.vggt_val_target_list_file} "
                f"(skipped={skipped})",
                flush=True,
            )
            return samples

        samples: List[Tuple[str, str, str]] = []
        for scene_path in self.scenes:
            scene_spec = self.scene_specs[scene_path]
            common_frames = self._common_vggt_val_frames(scene_spec, camera_cache={})
            start, stop, step = self.vggt_val_frame_sample
            end = len(common_frames) if stop is None else min(stop, len(common_frames))
            frame_positions = list(range(start, end, step))
            samples.extend((scene_path, "00", common_frames[pos]) for pos in frame_positions)
        print(
            f"[{self.dataset_label}] Built {len(samples)} VGGT-style val samples "
            f"from {len(self.scenes)} scenes with frame_sample={self.vggt_val_frame_sample}",
            flush=True,
        )
        return samples

    def _common_vggt_val_frames(self, scene_spec: dict, camera_cache: Optional[Dict[Tuple[str, str], object]] = None) -> List[str]:
        frames_by_camera = {}
        frame_cache: Dict[Tuple[str, str], List[str]] = {}
        for camera_id in sorted(scene_spec["camera_specs"].keys()):
            frames = self._get_camera_frames(
                scene_spec,
                camera_id,
                frame_cache=frame_cache,
                camera_cache=camera_cache,
            )
            if frames:
                frames_by_camera[camera_id] = frames
        if len(frames_by_camera) < self.min_cameras:
            raise RuntimeError(f"Scene {scene_spec['path']} has fewer than {self.min_cameras} valid cameras")
        common_frames = sorted(set.intersection(*[set(frames) for frames in frames_by_camera.values()]))
        if not common_frames:
            raise RuntimeError(f"Scene {scene_spec['path']} has no common frames across selected cameras")
        return common_frames

    def convert_attributes(self):
        cameras = self.cameras
        camera_cache_lru = self._camera_cache_lru
        super().convert_attributes()
        self.cameras = cameras
        self._camera_cache_lru = camera_cache_lru

    def _load_or_build_scene_index(self, data_root: str, seq_num: int) -> List[dict]:
        cache_path = self._index_cache_path(data_root)
        scene_index = None
        if self.use_index_cache and not self.rebuild_index_cache:
            scene_index = self._try_load_scene_index(cache_path)
            if scene_index is None and seq_num <= 0 and MeshXEvcDataset._distributed_rank() > 0:
                scene_index = self._wait_for_scene_index(cache_path)

        if scene_index is None:
            scene_index = self._scan_scene_index(data_root, seq_num)
            if self.use_index_cache and seq_num <= 0 and MeshXEvcDataset._distributed_rank() == 0:
                self._write_scene_index(cache_path, data_root, scene_index)
        elif seq_num > 0:
            scene_index = scene_index[:seq_num]
        return scene_index

    def _index_cache_path(self, data_root: str) -> str:
        cache_dir = self.index_cache_dir or osp.join(data_root, ".meshx_pi3_index_cache")
        digest_source = f"{osp.abspath(data_root)}|multiview"
        digest_source = f"{digest_source}|data_roots_file={self.data_roots_file}"
        digest_source = f"{digest_source}|intri_file={self.intri_file}|extri_file={self.extri_file}"
        digest_source = (
            f"{digest_source}|pose_prior_extri_file={self.pose_prior_extri_file}"
            f"|pose_prior_required={int(self.pose_prior_required)}"
        )
        if self.camera_ids is not None:
            digest_source = f"{digest_source}|camera_ids={','.join(sorted(self.camera_ids))}"
        digest = hashlib.md5(digest_source.encode("utf-8")).hexdigest()[:16]
        label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in self.dataset_label)
        return osp.join(cache_dir, f"{label}_{digest}_mv1.json")

    def _try_load_scene_index(self, cache_path: str) -> Optional[List[dict]]:
        if not cache_path or not osp.isfile(cache_path):
            return None
        try:
            with open(cache_path, "r") as f:
                payload = json.load(f)
            scenes = [self._normalize_scene_entry(item) for item in payload.get("scenes", [])]
            if scenes:
                print(f"[{self.dataset_label}] Loaded multiview scene index cache: {cache_path}", flush=True)
                return scenes
        except Exception as exc:
            print(f"[{self.dataset_label}] Failed to load multiview index cache {cache_path}: {exc}", flush=True)
        return None

    def _wait_for_scene_index(self, cache_path: str) -> Optional[List[dict]]:
        if not cache_path or self.index_cache_wait_sec <= 0:
            return None
        deadline = time.time() + self.index_cache_wait_sec
        while time.time() < deadline:
            scene_index = self._try_load_scene_index(cache_path)
            if scene_index is not None:
                return scene_index
            time.sleep(10.0)
        print(f"[{self.dataset_label}] Timed out waiting for multiview index cache: {cache_path}", flush=True)
        return None

    def _write_scene_index(self, cache_path: str, data_root: str, scene_index: List[dict]) -> None:
        if not cache_path:
            return
        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        payload = dict(
            version=1,
            data_root=data_root,
            dataset_label=self.dataset_label,
            data_roots_file=self.data_roots_file,
            intri_file=self.intri_file,
            extri_file=self.extri_file,
            camera_ids=sorted(self.camera_ids) if self.camera_ids is not None else None,
            scenes=[self._normalize_scene_entry(entry) for entry in scene_index],
        )
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        try:
            with open(tmp_path, "w") as f:
                json.dump(payload, f)
            os.replace(tmp_path, cache_path)
            print(f"[{self.dataset_label}] Wrote multiview scene index cache: {cache_path}", flush=True)
        except Exception as exc:
            print(f"[{self.dataset_label}] Failed to write multiview index cache {cache_path}: {exc}", flush=True)
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _scan_scene_index(self, data_root: str, seq_num: int) -> List[dict]:
        scene_index: List[dict] = []
        for spec in self._iter_scene_specs(
            data_root,
            self.data_roots_file,
            self.masks_dir,
            self.camera_ids,
            self.min_cameras,
            self.intri_file,
            self.extri_file,
            self.pose_prior_extri_file,
            self.pose_prior_required,
        ):
            scene_index.append(spec)
            if seq_num > 0 and len(scene_index) >= seq_num:
                break
        return scene_index

    @staticmethod
    def _iter_scene_specs(
        data_root: str,
        data_roots_file: str,
        masks_dir: str,
        camera_ids: Optional[Set[str]],
        min_cameras: int,
        intri_file: str,
        extri_file: str,
        pose_prior_extri_file: Optional[str],
        pose_prior_required: bool,
    ) -> Iterable[dict]:
        data_roots_path = osp.join(data_root, data_roots_file)
        if osp.isfile(data_roots_path):
            try:
                with open(data_roots_path, "r", encoding="utf-8", errors="ignore") as f:
                    rel_roots = [line.strip() for line in f if line.strip()]
                if len(rel_roots) > 16:
                    specs = MeshXBusinessMultiviewDataset._infer_scene_specs_from_data_roots(
                        data_root, rel_roots, masks_dir, camera_ids, min_cameras, intri_file, extri_file,
                        pose_prior_extri_file, pose_prior_required,
                    )
                else:
                    specs = []
                    for rel_root in rel_roots:
                        root = rel_root if osp.isabs(rel_root) else osp.join(data_root, rel_root)
                        scene_spec = MeshXBusinessMultiviewDataset._scene_spec_at_root(
                            root, masks_dir, camera_ids, min_cameras, lazy_frames=True,
                            intri_file=intri_file, extri_file=extri_file,
                            pose_prior_extri_file=pose_prior_extri_file,
                            pose_prior_required=pose_prior_required,
                        )
                        if scene_spec:
                            specs.append(scene_spec)
                if specs:
                    print(
                        f"[MeshXBusinessMultiviewDataset] Loaded {len(specs)} scene roots from {data_roots_path}",
                        flush=True,
                    )
                    yield from specs
                    return
                print(
                    f"[MeshXBusinessMultiviewDataset] No valid scene roots in {data_roots_path}; falling back to recursive scan",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[MeshXBusinessMultiviewDataset] Failed reading {data_roots_path}: {exc}; falling back to recursive scan",
                    flush=True,
                )

        for root, dirs, _ in os.walk(data_root):
            scene_spec = MeshXBusinessMultiviewDataset._scene_spec_at_root(
                root, masks_dir, camera_ids, min_cameras, lazy_frames=True,
                intri_file=intri_file, extri_file=extri_file,
                pose_prior_extri_file=pose_prior_extri_file,
                pose_prior_required=pose_prior_required,
            )
            if scene_spec:
                yield scene_spec
                dirs[:] = []

    @staticmethod
    def _infer_scene_specs_from_data_roots(
        data_root: str,
        rel_roots: List[str],
        masks_dir: str,
        camera_ids: Optional[Set[str]],
        min_cameras: int,
        intri_file: str,
        extri_file: str,
        pose_prior_extri_file: Optional[str],
        pose_prior_required: bool,
    ) -> List[dict]:
        sample_spec = None
        for rel_root in rel_roots[:128]:
            root = rel_root if osp.isabs(rel_root) else osp.join(data_root, rel_root)
            sample_spec = MeshXBusinessMultiviewDataset._scene_spec_at_root(
                root, masks_dir, camera_ids, min_cameras, lazy_frames=True,
                intri_file=intri_file, extri_file=extri_file,
                pose_prior_extri_file=pose_prior_extri_file,
                pose_prior_required=pose_prior_required,
            )
            if sample_spec:
                break
        if sample_spec is None:
            return []

        inferred_camera_ids = list(sample_spec["camera_specs"].keys())
        specs = []
        for rel_root in rel_roots:
            root = rel_root if osp.isabs(rel_root) else osp.join(data_root, rel_root)
            specs.append(MeshXBusinessMultiviewDataset._make_scene_entry(
                root, inferred_camera_ids, masks_dir, True, intri_file, extri_file,
                pose_prior_extri_file=pose_prior_extri_file,
            ))
        return specs

    @staticmethod
    def _scene_spec_at_root(
        root: str,
        masks_dir: str,
        camera_ids: Optional[Set[str]],
        min_cameras: int,
        lazy_frames: bool,
        intri_file: str,
        extri_file: str,
        pose_prior_extri_file: Optional[str] = None,
        pose_prior_required: bool = False,
    ) -> Optional[dict]:
        camera_root = osp.join(root, "cameras")
        image_root = osp.join(root, "images")
        depth_root = osp.join(root, "depths")
        if not (osp.isdir(camera_root) and osp.isdir(image_root) and osp.isdir(depth_root)):
            return None

        valid_camera_ids = []
        for camera_id in sorted(os.listdir(camera_root)):
            if camera_ids is not None and camera_id not in camera_ids:
                continue
            camera_dir = osp.join(camera_root, camera_id)
            if not osp.isdir(camera_dir):
                continue
            if (
                osp.isfile(osp.join(camera_dir, intri_file))
                and osp.isfile(osp.join(camera_dir, extri_file))
                and (
                    not pose_prior_required
                    or (
                        pose_prior_extri_file is not None
                        and osp.isfile(osp.join(camera_dir, pose_prior_extri_file))
                    )
                )
                and osp.isdir(osp.join(image_root, camera_id))
                and osp.isdir(osp.join(depth_root, camera_id))
            ):
                valid_camera_ids.append(camera_id)

        if len(valid_camera_ids) < min_cameras:
            return None
        return MeshXBusinessMultiviewDataset._make_scene_entry(
            root, valid_camera_ids, masks_dir, lazy_frames, intri_file, extri_file,
            pose_prior_extri_file=pose_prior_extri_file,
        )

    @staticmethod
    def _make_scene_entry(
        path: str,
        camera_ids: List[str],
        masks_dir: str,
        lazy_frames: bool,
        intri_file: str = "intri.yml",
        extri_file: str = "extri.yml",
        pose_prior_extri_file: Optional[str] = None,
    ) -> dict:
        camera_specs = {}
        for camera_id in camera_ids:
            camera_specs[camera_id] = dict(
                image_root=osp.join(path, "images", camera_id),
                depth_root=osp.join(path, "depths", camera_id),
                mask_root=osp.join(path, masks_dir, camera_id),
                intri_path=osp.join(path, "cameras", camera_id, intri_file),
                extri_path=osp.join(path, "cameras", camera_id, extri_file),
                pose_prior_extri_path=(
                    osp.join(path, "cameras", camera_id, pose_prior_extri_file)
                    if pose_prior_extri_file is not None
                    else ""
                ),
                frames=[],
                lazy_frames=bool(lazy_frames),
            )
        return dict(path=path, label=path, camera_specs=camera_specs)

    def _normalize_scene_entry(self, entry: dict) -> dict:
        path = entry["path"]
        camera_specs = {}
        for camera_id, spec in entry.get("camera_specs", {}).items():
            camera_specs[str(camera_id)] = dict(
                image_root=spec.get("image_root", osp.join(path, "images", str(camera_id))),
                depth_root=spec.get("depth_root", osp.join(path, "depths", str(camera_id))),
                mask_root=spec.get("mask_root", osp.join(path, self.masks_dir, str(camera_id))),
                intri_path=spec.get("intri_path", osp.join(path, "cameras", str(camera_id), self.intri_file)),
                extri_path=spec.get("extri_path", osp.join(path, "cameras", str(camera_id), self.extri_file)),
                pose_prior_extri_path=spec.get(
                    "pose_prior_extri_path",
                    (
                        osp.join(path, "cameras", str(camera_id), self.pose_prior_extri_file)
                        if self.pose_prior_extri_file is not None
                        else ""
                    ),
                ),
                frames=list(spec.get("frames", [])),
                lazy_frames=spec.get("lazy_frames", True),
            )
        return dict(path=path, label=entry.get("label", path), camera_specs=camera_specs)

    def _get_camera_frames(
        self,
        scene_spec: dict,
        camera_id: str,
        frame_cache: Optional[Dict[Tuple[str, str], List[str]]] = None,
        camera_cache: Optional[Dict[Tuple[str, str], object]] = None,
    ) -> List[str]:
        key = (scene_spec["path"], camera_id)
        if frame_cache is not None and key in frame_cache:
            return frame_cache[key]

        camera_spec = scene_spec["camera_specs"][camera_id]
        frames = camera_spec.get("frames", [])
        if not frames:
            if camera_spec.get("lazy_frames", True):
                cameras = self._get_cameras(scene_spec["path"], camera_id, scene_spec, camera_cache=camera_cache)
                frames = sorted(str(frame_id) for frame_id in cameras.keys())
            else:
                frames = MeshXEvcDataset._collect_frame_ids(camera_spec["image_root"], camera_spec["depth_root"])
            if not frames:
                raise FileNotFoundError(f"No image/depth frame pairs found in {scene_spec['path']} camera {camera_id}")
            if frame_cache is not None:
                frame_cache[key] = frames
        return frames

    def _remember_cached_camera(self, key: Tuple[str, str]) -> None:
        if self.camera_cache_size <= 0:
            return
        self._camera_cache_lru[key] = None
        self._camera_cache_lru.move_to_end(key, last=True)
        while len(self._camera_cache_lru) > self.camera_cache_size:
            evict_key, _ = self._camera_cache_lru.popitem(last=False)
            self.cameras.pop(evict_key, None)

    def _get_cameras(
        self,
        scene_path: str,
        camera_id: str,
        scene_spec: dict,
        camera_cache: Optional[Dict[Tuple[str, str], object]] = None,
    ):
        key = (scene_path, camera_id)
        if camera_cache is not None and key in camera_cache:
            return camera_cache[key]
        if self.camera_cache_size > 0 and key in self.cameras:
            self._remember_cached_camera(key)
            return self.cameras[key]

        camera_spec = scene_spec["camera_specs"][camera_id]
        cameras = read_camera(
            camera_spec["intri_path"],
            camera_spec["extri_path"],
            use_pkl=self.use_camera_pkl_cache,
        )
        if self.camera_cache_size > 0:
            self.cameras[key] = cameras
            self._remember_cached_camera(key)
        elif camera_cache is not None:
            camera_cache[key] = cameras
        return cameras

    def _get_pose_prior_cameras(
        self,
        scene_path: str,
        camera_id: str,
        scene_spec: dict,
        camera_cache: Optional[Dict[Tuple[str, str], object]] = None,
    ):
        camera_spec = scene_spec["camera_specs"][camera_id]
        prior_extri_path = camera_spec.get("pose_prior_extri_path", "")
        if not prior_extri_path:
            if self.pose_prior_required:
                raise FileNotFoundError(f"Missing pose prior extri path for {scene_path} camera {camera_id}")
            return None
        if not osp.isfile(prior_extri_path):
            if self.pose_prior_required:
                raise FileNotFoundError(f"Missing pose prior extri file: {prior_extri_path}")
            return None

        key = (scene_path, camera_id, "pose_prior")
        if camera_cache is not None and key in camera_cache:
            return camera_cache[key]
        cameras = read_camera(
            camera_spec["intri_path"],
            prior_extri_path,
            use_pkl=self.use_camera_pkl_cache,
        )
        if camera_cache is not None:
            camera_cache[key] = cameras
        return cameras

    def _sample_view_keys(
        self,
        scene_spec: dict,
        rng,
        camera_cache: Optional[Dict[Tuple[str, str], object]] = None,
    ) -> List[Tuple[str, str]]:
        frames_by_camera = {}
        frame_cache: Dict[Tuple[str, str], List[str]] = {}
        for camera_id in sorted(scene_spec["camera_specs"].keys()):
            frames = self._get_camera_frames(
                scene_spec,
                camera_id,
                frame_cache=frame_cache,
                camera_cache=camera_cache,
            )
            if frames:
                frames_by_camera[camera_id] = frames
        camera_ids = sorted(frames_by_camera.keys())
        if len(camera_ids) < self.min_cameras:
            raise RuntimeError(f"Scene {scene_spec['path']} has fewer than {self.min_cameras} valid cameras")

        common_frames = sorted(set.intersection(*[set(frames_by_camera[camera_id]) for camera_id in camera_ids]))
        if common_frames:
            anchor_pos = int(rng.integers(0, len(common_frames)))
            max_distance = int(self.max_distance / 8 * self.frame_num)
            start = max(0, anchor_pos - max_distance)
            end = min(len(common_frames) - 1, start + 2 * max_distance)
            start = max(0, end - 2 * max_distance)
            window_by_camera = {camera_id: common_frames[start:end + 1] for camera_id in camera_ids}
        else:
            window_by_camera = {}
            for camera_id, frames in frames_by_camera.items():
                anchor_pos = int(rng.integers(0, len(frames)))
                max_distance = int(self.max_distance / 8 * self.frame_num)
                start = max(0, anchor_pos - max_distance)
                end = min(len(frames) - 1, start + 2 * max_distance)
                start = max(0, end - 2 * max_distance)
                window_by_camera[camera_id] = frames[start:end + 1]

        shuffled_cameras = camera_ids.copy()
        rng.shuffle(shuffled_cameras)
        view_keys: List[Tuple[str, str]] = []
        used = set()
        for i in range(self.frame_num):
            camera_id = shuffled_cameras[i % len(shuffled_cameras)]
            frame_pool = window_by_camera[camera_id] or frames_by_camera[camera_id]
            frame_id = str(rng.choice(frame_pool))
            for _ in range(8):
                candidate = (camera_id, frame_id)
                if candidate not in used:
                    break
                frame_id = str(rng.choice(frame_pool))
            view_keys.append((camera_id, frame_id))
            used.add((camera_id, frame_id))
        return view_keys

    def _vggt_val_view_keys(
        self,
        scene_spec: dict,
        target_camera_id: str,
        target_frame_id: str,
        camera_cache: Optional[Dict[Tuple[str, str], object]] = None,
    ) -> List[Tuple[str, str]]:
        camera_ids = sorted(scene_spec["camera_specs"].keys())
        common_frames = self._common_vggt_val_frames(scene_spec, camera_cache=camera_cache)
        target_frame_id = str(target_frame_id)
        if target_frame_id not in common_frames:
            raise FileNotFoundError(f"Target frame {target_frame_id} is not common to all cameras in {scene_spec['path']}")
        target_pos = common_frames.index(target_frame_id)
        if target_camera_id in camera_ids:
            camera_ids = [target_camera_id] + [camera_id for camera_id in camera_ids if camera_id != target_camera_id]

        total_frames = self.vggt_val_extra_src_pool + 1
        start = max(0, target_pos - self.vggt_val_extra_src_pool // 2)
        frame_positions = list(range(start, min(start + total_frames, len(common_frames))))
        if target_pos not in frame_positions:
            frame_positions.insert(0, target_pos)
        frame_positions = [target_pos] + [pos for pos in frame_positions if pos != target_pos]

        view_keys: List[Tuple[str, str]] = []
        for camera_id in camera_ids:
            for pos in frame_positions:
                view_keys.append((camera_id, common_frames[pos]))
        return view_keys

    def _get_views(self, index, resolution, rng):
        if self.vggt_val_protocol:
            scene_path, target_camera_id, target_frame_id = self.val_samples[index]
        else:
            scene_path = self.scenes[index]
            target_camera_id, target_frame_id = "", ""
        scene_spec = self.scene_specs[scene_path]
        camera_cache: Dict[Tuple[str, str], object] = {}
        if self.vggt_val_protocol:
            view_keys = self._vggt_val_view_keys(
                scene_spec,
                target_camera_id,
                target_frame_id,
                camera_cache=camera_cache,
            )
        else:
            view_keys = self._sample_view_keys(scene_spec, rng, camera_cache=camera_cache)
        sample_id = (
            f"{osp.relpath(scene_path, self.data_root)}|{target_camera_id}|{target_frame_id}"
            if self.vggt_val_protocol
            else ""
        )
        self.this_views_info = dict(
            scene=osp.relpath(scene_path, self.data_root),
            sampled=[f"{camera_id}/{frame_id}" for camera_id, frame_id in view_keys],
            target_camera=target_camera_id,
            target_frame=target_frame_id,
            protocol="vggt_val" if self.vggt_val_protocol else "pi3_native",
        )

        views = []
        for camera_id, frame_id in view_keys:
            camera_spec = scene_spec["camera_specs"][camera_id]
            image_path = MeshXEvcDataset._pick_file(camera_spec["image_root"], frame_id, (".jpg", ".jpeg", ".png"))
            depth_path = MeshXEvcDataset._pick_file(camera_spec["depth_root"], frame_id, (".exr", ".npy", ".png"))
            mask_path = (
                MeshXEvcDataset._pick_file(camera_spec["mask_root"], frame_id, (".png", ".jpg", ".jpeg", ".npy"))
                if self.use_masks else None
            )
            if image_path is None or depth_path is None:
                raise FileNotFoundError(f"Missing image/depth for {camera_id}/{frame_id} in {scene_path}")
            if self.use_masks and mask_path is None:
                raise FileNotFoundError(f"Missing mask for {camera_id}/{frame_id} in {scene_path}")

            cameras = self._get_cameras(scene_path, camera_id, scene_spec, camera_cache=camera_cache)
            cam = MeshXEvcDataset._lookup_camera(cameras, frame_id)
            pose_prior_cameras = self._get_pose_prior_cameras(
                scene_path,
                camera_id,
                scene_spec,
                camera_cache=camera_cache,
            )
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
            depthmap = self._read_depth(depth_path)
            depthmap = self._apply_mask(depthmap, mask_path)
            depthmap = self._apply_depth_filters(depthmap)
            camera_pose = MeshXEvcDataset._w2c_to_c2w(cam.R, cam.T)
            pose_prior = None
            if pose_prior_cameras is not None:
                pose_prior_cam = MeshXEvcDataset._lookup_camera(pose_prior_cameras, frame_id)
                pose_prior = MeshXEvcDataset._w2c_to_c2w(pose_prior_cam.R, pose_prior_cam.T)
            intrinsics = cam.K.astype(np.float32)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image,
                depthmap,
                intrinsics,
                resolution,
                rng=rng,
                info=image_path,
            )

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset=self.dataset_label,
                label=osp.relpath(scene_spec.get("label", scene_path), self.data_root),
                instance=f"{camera_id}/{frame_id}",
                sample_id=sample_id,
                camera_id=camera_id,
                frame_id=frame_id,
            ))
            if pose_prior is not None:
                views[-1]["pose_prior"] = pose_prior.astype(np.float32)

        return views

    _read_depth = staticmethod(MeshXEvcDataset._read_depth)
    _read_mask = staticmethod(MeshXEvcDataset._read_mask)
    _apply_mask = MeshXEvcDataset._apply_mask
    _apply_depth_filters = MeshXEvcDataset._apply_depth_filters
