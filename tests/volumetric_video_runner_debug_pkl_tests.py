import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

import torch

from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner
from easyvolcap.utils.base_utils import dotdict


class VolumetricVideoRunnerDebugPklTests(unittest.TestCase):
    def make_runner(self):
        runner = VolumetricVideoRunner.__new__(VolumetricVideoRunner)
        return runner

    def make_batch_and_output(self):
        H, W, N = 2, 2, 1
        batch = dotdict(
            rgb=torch.rand(1, N, H, W, 3),
            c2ws=torch.eye(4).reshape(1, 1, 4, 4),
            ixts=torch.tensor([[[[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]]]),
            dpt=torch.ones(1, N, H, W),
            msk=torch.ones(1, N, H, W),
            meta=dotdict(
                H=torch.tensor([H]),
                W=torch.tensor([W]),
                aspect_ratio=torch.tensor([1.0]),
                original_aspect_ratio=torch.tensor([1.0]),
                data_root=["/tmp/eth3d/courtyard"],
                dataset_name=["eth3d"],
            ),
        )
        output = dotdict(
            scalar_stats={
                "tensor_metric": torch.tensor(1.25),
                "float_metric": 2.5,
            },
            dpt_map=torch.ones(1, N, H, W),
            xyz_map=torch.ones(1, N, H, W, 3),
            cam_map=torch.zeros(1, 1),
            dpt_cnf=torch.ones(1, N, H, W),
        )
        return batch, output

    def test_save_predictions_to_pkl_accepts_python_float_scalar_stats(self):
        runner = self.make_runner()
        batch, output = self.make_batch_and_output()

        fake_w2cs = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0, 0.0],
                                   [0.0, 0.0, 1.0, 0.0]]])
        fake_ixts = batch.ixts[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("easyvolcap.runners.volumetric_video_runner.decode_camera_params", return_value=(fake_w2cs, fake_ixts)):
                runner.save_predictions_to_pkl(0, 0, batch, output, tmpdir, metrics={"rmse": 0.1})

            pkl_path = os.path.join(tmpdir, "log_iter_0.pkl")
            self.assertTrue(os.path.isfile(pkl_path))

            with open(pkl_path, "rb") as handle:
                data = pickle.load(handle)

            self.assertEqual(data["meta"]["scalar_stats"]["tensor_metric"], 1.25)
            self.assertEqual(data["meta"]["scalar_stats"]["float_metric"], 2.5)
            self.assertEqual(data["meta"]["metrics"]["rmse"], 0.1)


if __name__ == "__main__":
    unittest.main()
