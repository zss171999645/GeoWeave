import sys
sys.path.append('.')

import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image

from datasets.base.base_dataset import BaseDataset
from easyvolcap.utils.easy_utils import read_camera


class MeshXTartanAirDataset(BaseDataset):
    """Adapter for meshx-processed TarTanAir sequences.

    Expected per-sequence layout (current meshx training data):
      <data_root>/<scene>/<difficulty>/<track>/
        intri.yml
        extri.yml
        images/<frame_id>/<cam_id>.png
        depths/<frame_id>/<cam_id>.exr
    """

    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=24,
        seq_num=-1,
        use_camera_pkl_cache=False,
        use_index_cache=False,
        rebuild_index_cache=False,
        index_cache_dir=None,
        index_cache_wait_sec=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert data_root is not None

        self.verbose = verbose
        self.dataset_label = 'TarTanAir'
        self.max_distance = max_distance
        self.data_root = data_root
        self.use_camera_pkl_cache = bool(use_camera_pkl_cache)
        self.use_index_cache = bool(use_index_cache)
        self.rebuild_index_cache = bool(rebuild_index_cache)
        self.index_cache_dir = index_cache_dir
        self.index_cache_wait_sec = int(index_cache_wait_sec)

        self.sequences = []
        self.num_imgs = {}
        self.frame_ids = {}
        self.cameras = {}

        scene_names = sorted([d for d in os.listdir(data_root) if osp.isdir(osp.join(data_root, d))])
        for scene in scene_names:
            for difficulty in ('Easy', 'Hard'):
                diff_dir = osp.join(data_root, scene, difficulty)
                if not osp.isdir(diff_dir):
                    continue
                for track in sorted(os.listdir(diff_dir)):
                    seq_dir = osp.join(diff_dir, track)
                    if not osp.isdir(seq_dir):
                        continue
                    intri = osp.join(seq_dir, 'intri.yml')
                    extri = osp.join(seq_dir, 'extri.yml')
                    img_root = osp.join(seq_dir, 'images')
                    dpt_root = osp.join(seq_dir, 'depths')
                    if not (osp.isfile(intri) and osp.isfile(extri) and osp.isdir(img_root) and osp.isdir(dpt_root)):
                        continue
                    frames = sorted([f for f in os.listdir(img_root) if osp.isdir(osp.join(img_root, f))])
                    if not frames:
                        continue
                    seq_key = (scene, difficulty, track)
                    self.sequences.append(seq_key)
                    self.frame_ids[seq_key] = frames
                    self.num_imgs[seq_key] = len(frames)
                    self.cameras[seq_key] = read_camera(
                        intri,
                        extri,
                        use_pkl=self.use_camera_pkl_cache,
                    )

        if seq_num > 0:
            self.sequences = self.sequences[:seq_num]

        if self.verbose:
            print(f'[{self.dataset_label}] sequences: {self.sequences}')
        print(f'[{self.dataset_label}] Found {len(self.sequences)} valid sequences in {data_root}', flush=True)

    def __len__(self):
        return len(self.sequences)

    @staticmethod
    def _pick_cam_file(frame_dir: str, preferred_exts):
        if not osp.isdir(frame_dir):
            return None
        files = sorted([f for f in os.listdir(frame_dir) if osp.isfile(osp.join(frame_dir, f))])
        if not files:
            return None
        for ext in preferred_exts:
            for f in files:
                if f.lower().endswith(ext):
                    return osp.join(frame_dir, f)
        return osp.join(frame_dir, files[0])

    @staticmethod
    def _w2c_to_c2w(R: np.ndarray, T: np.ndarray) -> np.ndarray:
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R.astype(np.float32)
        w2c[:3, 3] = T.reshape(3).astype(np.float32)
        c2w = np.linalg.inv(w2c).astype(np.float32)
        return c2w

    def _sample_indices(self, rng, num_imgs):
        if self.frame_num > 20 and rng.random() < self.random_sample_thres:
            should_replace = num_imgs < self.frame_num
            return list(rng.choice(num_imgs, size=self.frame_num, replace=should_replace))

        idxs = [rng.integers(0, num_imgs)]
        max_distance = int(self.max_distance / 8 * self.frame_num)
        start_idx = max(0, idxs[-1] - max_distance)
        end_idx = min(num_imgs - 1, start_idx + 2 * max_distance)
        start_idx = max(0, end_idx - 2 * max_distance)
        valid_indices = np.arange(start_idx, end_idx + 1)
        should_replace = len(valid_indices) < self.frame_num - 1
        idxs.extend(list(rng.choice(valid_indices, self.frame_num - 1, replace=should_replace)))
        return idxs

    def _get_views(self, index, resolution, rng):
        scene = self.sequences[index]
        seq_dir = osp.join(self.data_root, scene[0], scene[1], scene[2])
        img_root = osp.join(seq_dir, 'images')
        dpt_root = osp.join(seq_dir, 'depths')
        cams = self.cameras[scene]
        frames = self.frame_ids[scene]
        num_imgs = len(frames)

        sampled = self._sample_indices(rng, num_imgs)
        self.this_views_info = dict(scene=scene, sampled=sampled)

        views = []
        for idx in sampled:
            frame_id = frames[int(idx)]
            img_frame_dir = osp.join(img_root, frame_id)
            dpt_frame_dir = osp.join(dpt_root, frame_id)

            impath = self._pick_cam_file(img_frame_dir, preferred_exts=('.png', '.jpg', '.jpeg'))
            depthpath = self._pick_cam_file(dpt_frame_dir, preferred_exts=('.exr', '.npy', '.png'))
            if impath is None or depthpath is None:
                raise FileNotFoundError(f'Missing image/depth for frame {frame_id} in {scene}')

            rgb_image = np.array(Image.open(impath))

            if depthpath.lower().endswith('.npy'):
                depthmap = np.load(depthpath).astype(np.float32)
            else:
                depthmap = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
                if depthmap is None:
                    raise ValueError(f'Failed to read depth map: {depthpath}')
                if depthmap.ndim == 3:
                    depthmap = depthmap[..., 0]
                depthmap = depthmap.astype(np.float32)
                # Typical uint16 millimeter depth.
                if depthpath.lower().endswith('.png'):
                    depthmap = depthmap / 1000.0

            if frame_id not in cams:
                raise KeyError(f'Frame {frame_id} not found in camera dict for {scene}')
            cam = cams[frame_id]
            camera_pose = self._w2c_to_c2w(cam.R, cam.T)
            intrinsics = cam.K.astype(np.float32)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath
            )

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset=self.dataset_label,
                label=f'{scene[0]}_{scene[1]}_{scene[2]}',
                instance=str(frame_id),
            ))

        return views
