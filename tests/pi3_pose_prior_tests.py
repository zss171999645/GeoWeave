from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PI3_ROOT = REPO_ROOT / "aidi" / "third_party" / "pi3_training"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PI3_ROOT))

from pi3.models.pi3_training import pose_prior_to_vggt_like_features  # noqa: E402


class Pi3PosePriorTests(unittest.TestCase):
    def test_pose_prior_features_are_relative_to_first_view_and_scale_free(self):
        poses = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1)
        poses[0, 1, 0, 3] = 10.0
        poses[0, 2, 0, 3] = 20.0
        intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 3, 1, 1)
        intrinsics[..., 0, 0] = 10.0
        intrinsics[..., 1, 1] = 20.0

        features, valid = pose_prior_to_vggt_like_features(
            poses,
            intrinsics=intrinsics,
            image_hw=(40, 20),
            return_valid_mask=True,
        )

        self.assertEqual(tuple(features.shape), (1, 3, 10))
        self.assertTrue(torch.allclose(valid, torch.ones(1, 3, dtype=torch.bool)))
        self.assertTrue(torch.allclose(features[0, 0, :3], torch.zeros(3), atol=1e-6))
        self.assertLess(features[0, 1, 0].item(), 0.0)
        self.assertLess(features[0, 2, 0].item(), features[0, 1, 0].item())
        self.assertTrue(torch.allclose(features[..., 3:7].norm(dim=-1), torch.ones(1, 3), atol=1e-6))
        self.assertTrue(torch.allclose(features[..., -1], torch.zeros(1, 3)))
        self.assertTrue(torch.isfinite(features).all())

    def test_c2w_pose_prior_format_keeps_relative_c2w_direction(self):
        poses = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
        poses[0, 1, 0, 3] = 10.0

        w2c_features = pose_prior_to_vggt_like_features(
            poses,
            image_hw=(16, 16),
            pose_prior_format="vggt_w2c_quat10",
        )
        c2w_features = pose_prior_to_vggt_like_features(
            poses,
            image_hw=(16, 16),
            pose_prior_format="c2w_quat10",
        )

        self.assertLess(w2c_features[0, 1, 0].item(), 0.0)
        self.assertGreater(c2w_features[0, 1, 0].item(), 0.0)
        self.assertTrue(torch.allclose(w2c_features[..., 3:7].norm(dim=-1), torch.ones(1, 2), atol=1e-6))
        self.assertTrue(torch.allclose(c2w_features[..., 3:7].norm(dim=-1), torch.ones(1, 2), atol=1e-6))

    def test_pose_prior_features_zero_invalid_views_and_clear_flag(self):
        poses = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
        poses[0, 1, 0, 0] = float("nan")

        features, valid = pose_prior_to_vggt_like_features(
            poses,
            image_hw=(16, 16),
            return_valid_mask=True,
        )

        self.assertTrue(valid[0, 0].item())
        self.assertFalse(valid[0, 1].item())
        self.assertEqual(features[0, 1, -1].item(), 0.0)
        self.assertTrue(torch.allclose(features[0, 1], torch.zeros(10)))


if __name__ == "__main__":
    unittest.main()
