from __future__ import annotations

import importlib.util
import unittest
from unittest import mock


def has_torch_stack() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("torchvision") is not None
    )


@unittest.skipIf(not has_torch_stack(), "torch/torchvision stack not installed")
class PoseSupervisorMetricTests(unittest.TestCase):
    def test_camera_accuracy_float_metrics_are_recorded(self):
        import torch

        from easyvolcap.models.supervisors.pose_supervisor import PoseSupervisor
        from easyvolcap.utils.base_utils import dotdict
        from easyvolcap.utils.loss_utils import PoseLossType

        supervisor = object.__new__(PoseSupervisor)
        supervisor.pose_loss_type = PoseLossType.L1
        supervisor.pose_loss_huber_delta = 0.1
        supervisor.pose_loss_weight = 0.0
        supervisor.seq_pose_loss_translation_weight = 1.0
        supervisor.seq_pose_loss_rotation_weight = 1.0
        supervisor.seq_pose_loss_focal_weight = 0.5
        supervisor.seq_pose_loss_gamma = 0.6
        supervisor.seq_pose_loss_translation_max = 100.0

        output = dotdict(
            cam_map=torch.zeros(1, 2, 9),
            cam_maps=[torch.zeros(1, 2, 9), torch.ones(1, 2, 9)],
        )
        batch = dotdict(
            cam=torch.zeros(1, 2, 9),
            msk=torch.ones(1, 2, 128, 1),
        )
        scalar_stats = dotdict()

        with mock.patch(
            "easyvolcap.models.supervisors.pose_supervisor.camera_accuracy_auc",
            return_value=dotdict(
                pose_auc_03=0.5,
                rotation_accuracy_15=torch.tensor([0.25, 0.75]),
            ),
        ):
            loss = supervisor.compute_loss(
                output,
                batch,
                torch.tensor(0.0),
                scalar_stats,
                dotdict(),
            )

        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(torch.is_tensor(scalar_stats.pose_auc_03))
        self.assertAlmostEqual(float(scalar_stats.pose_auc_03.item()), 0.5)
        self.assertAlmostEqual(float(scalar_stats.rotation_accuracy_15.item()), 0.5)


if __name__ == "__main__":
    unittest.main()
