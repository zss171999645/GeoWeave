import unittest
from pathlib import Path

from aidi.scripts.baselines.eval_config_utils import load_resolved_config


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestAdded6DepthEvalConfigs(unittest.TestCase):
    def _load_cfg(self, rel_path: str):
        return load_resolved_config(REPO_ROOT / rel_path)

    def test_existing_added5_configs_match_sparse_depth_protocol(self):
        expectations = {
            "configs/exps/vggt/evaluation/eth3d.yaml": {
                "meta_root_suffix": "datasets/eth3d/train",
                "dataset_name": "eth3d/train",
                "scene_stride": 1,
            },
            "configs/exps/vggt/evaluation/7scenes.yaml": {
                "meta_root_suffix": "datasets/7scenes/test",
                "dataset_name": "7scenes/test",
                "scene_stride": 1,
            },
            "configs/exps/vggt/evaluation/blendedmvs.yaml": {
                "meta_root_suffix": "evc_data/blendedmvs/test",
                "dataset_name": None,
                "scene_stride": 10,
            },
            "configs/exps/vggt/evaluation/mvs_synth.yaml": {
                "meta_root_suffix": "datasets/mvs_synth/GTAV_1080",
                "dataset_name": None,
                "scene_stride": 10,
            },
            "configs/exps/vggt/evaluation/vkitti2.yaml": {
                "meta_root_suffix": "datasets/vkitti2",
                "dataset_name": None,
                "scene_stride": 10,
            },
        }
        for rel_path, expected in expectations.items():
            with self.subTest(config=rel_path):
                cfg = self._load_cfg(rel_path)
                dataset_cfg = cfg["val_dataloader_cfg"]["dataset_cfg"]
                sampler_cfg = cfg["val_dataloader_cfg"]["sampler_cfg"]
                point_cfg = dataset_cfg["metaset_cfgs"][-1]

                self.assertEqual(dataset_cfg["type"], "GeneralizableDataset")
                self.assertNotIn("view_sample", dataset_cfg)
                self.assertNotIn("frame_sample", dataset_cfg)
                self.assertTrue(dataset_cfg["meta_roots"][0].endswith(expected["meta_root_suffix"]))

                if expected["dataset_name"] is not None:
                    self.assertEqual(dataset_cfg["metaset_cfgs"][0]["dataset_name"], expected["dataset_name"])

                self.assertEqual(point_cfg["type"], "MultiviewPointDataset")
                self.assertEqual(point_cfg["max_size"], 518)
                self.assertEqual(point_cfg["align_size"], 14)
                self.assertEqual(point_cfg["view_sample"], [0, None, 1])
                self.assertEqual(point_cfg["frame_sample"], [0, None, 1])
                self.assertEqual(point_cfg["n_srcs_list"], [9])

                self.assertEqual(sampler_cfg["view_sample"], [0, None, 1])
                self.assertEqual(sampler_cfg["frame_sample"], [0, None, expected["scene_stride"]])

    def test_scannetpp_config_exists_and_matches_large_scene_protocol(self):
        cfg = self._load_cfg("configs/exps/vggt/evaluation/scannetpp.yaml")
        dataset_cfg = cfg["val_dataloader_cfg"]["dataset_cfg"]
        sampler_cfg = cfg["val_dataloader_cfg"]["sampler_cfg"]
        point_cfg = dataset_cfg["metaset_cfgs"][-1]

        self.assertEqual(dataset_cfg["type"], "GeneralizableDataset")
        self.assertNotIn("view_sample", dataset_cfg)
        self.assertNotIn("frame_sample", dataset_cfg)
        self.assertTrue(dataset_cfg["meta_roots"][0].endswith("evc_data/scannetpp"))

        scene_cfg = dataset_cfg["metaset_cfgs"][0]
        self.assertEqual(scene_cfg["dataset_name"], "scannetpp")
        self.assertEqual(scene_cfg["min_depth_quantile"], 0.02)
        self.assertEqual(scene_cfg["max_depth_quantile"], 0.95)

        self.assertEqual(point_cfg["type"], "MultiviewPointDataset")
        self.assertEqual(point_cfg["max_size"], 518)
        self.assertEqual(point_cfg["align_size"], 14)
        self.assertEqual(point_cfg["view_sample"], [0, None, 1])
        self.assertEqual(point_cfg["frame_sample"], [0, None, 1])
        self.assertEqual(point_cfg["n_srcs_list"], [9])

        self.assertEqual(sampler_cfg["view_sample"], [0, None, 1])
        self.assertEqual(sampler_cfg["frame_sample"], [0, None, 10])


if __name__ == "__main__":
    unittest.main()
