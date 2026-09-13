from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tests.pi3_meshx_evc_dataset_tests import write_camera_files, write_depth, write_image


REPO_ROOT = Path(__file__).resolve().parents[1]
PI3_ROOT = REPO_ROOT / "aidi" / "third_party" / "pi3_training"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PI3_ROOT))

from datasets.meshx_evc_dataset import MeshXEvcDataset  # noqa: E402
from pi3.models.loss import Pi3Loss  # noqa: E402
from pi3.utils.geometry import depthmap_to_absolute_camera_coordinates  # noqa: E402


class Pi3LazyGtPointsTests(unittest.TestCase):
    def _make_view_pair(self):
        depth0 = np.array(
            [
                [1.0, 2.0, 0.0],
                [3.0, 4.0, 5.0],
            ],
            dtype=np.float32,
        )
        depth1 = np.array(
            [
                [2.0, 0.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
        intrinsic = np.array(
            [
                [10.0, 0.0, 1.0],
                [0.0, 10.0, 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        pose0 = np.eye(4, dtype=np.float32)
        pose1 = np.eye(4, dtype=np.float32)
        pose1[:3, 3] = np.array([0.5, -0.25, 1.0], dtype=np.float32)

        old_views = []
        lazy_views = []
        for depth, pose, instance in ((depth0, pose0, "0"), (depth1, pose1, "1")):
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(
                depth,
                intrinsic,
                pose,
                z_far=5.5,
            )
            valid_mask = valid_mask & np.isfinite(pts3d).all(axis=-1)
            common = dict(
                img=torch.zeros((1, 3, 2, 3), dtype=torch.float32),
                depthmap=torch.from_numpy(depth[None]),
                camera_intrinsics=torch.from_numpy(intrinsic[None]),
                camera_pose=torch.from_numpy(pose[None]),
                valid_mask=torch.from_numpy(valid_mask[None]),
                dataset=["BusinessDriving"],
                label=["scene"],
                instance=[instance],
            )
            old_views.append({**common, "pts3d": torch.from_numpy(pts3d[None])})
            lazy_views.append(common)
        return old_views, lazy_views

    def test_pi3_loss_lazy_gt_points_match_precomputed_pts3d_path(self):
        old_views, lazy_views = self._make_view_pair()
        loss = Pi3Loss(normal_loss_weight=0.0)

        old_gt = loss.prepare_gt(old_views)
        lazy_gt = loss.prepare_gt(lazy_views)

        self.assertTrue(torch.equal(old_gt["valid_masks"], lazy_gt["valid_masks"]))
        mask = old_gt["valid_masks"]
        torch.testing.assert_close(old_gt["global_points"][mask], lazy_gt["global_points"][mask], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(old_gt["local_points"][mask], lazy_gt["local_points"][mask], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(old_gt["camera_poses"], lazy_gt["camera_poses"], rtol=1e-5, atol=1e-6)

    def test_meshx_dataset_lazy_gt_points_omits_dense_pts3d_in_worker_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seq = Path(tmpdir) / "scene"
            write_camera_files(seq, ["000000"])
            write_image(seq / "images" / "000000" / "000000.jpg", size=(16, 16))
            depth = np.ones((16, 16), dtype=np.float32)
            depth[:, 8:] = 100.0
            write_depth(seq / "depths" / "000000" / "000000.npy", depth)

            dataset = MeshXEvcDataset(
                data_root=str(tmpdir),
                dataset_label="BusinessDriving",
                resolution=[[16, 16]],
                frame_num=1,
                z_far=50.0,
                lazy_gt_points=True,
                shuffle=False,
            )
            view = dataset[0][0]

            self.assertNotIn("pts3d", view)
            self.assertIn("valid_mask", view)
            self.assertEqual(view["valid_mask"].shape, (16, 16))
            self.assertTrue(np.all(view["depthmap"][:, :8] > 0))
            self.assertTrue(np.all(view["depthmap"][:, 8:] == 0))


if __name__ == "__main__":
    unittest.main()
