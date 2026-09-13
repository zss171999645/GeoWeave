from pathlib import Path
import unittest

from aidi.scripts.baselines.eval_config_utils import load_resolved_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/exps/vggt/saturnv/evaluation/vggt_business_10view_badcase_top100_lio.yaml"
TARGET_LIST_PATH = REPO_ROOT / "aidi/benchmarks/business_10view_badcase_top100/target_img_list.txt"
TARGET_LIST_CFG = "aidi/benchmarks/business_10view_badcase_top100/target_img_list.txt"
CONFIG_TOP300_PATH = REPO_ROOT / "configs/exps/vggt/saturnv/evaluation/vggt_business_10view_badcase_top300_lio.yaml"
TARGET_LIST_TOP300_PATH = REPO_ROOT / "aidi/benchmarks/business_10view_badcase_top300/target_img_list.txt"
TARGET_LIST_TOP300_CFG = "aidi/benchmarks/business_10view_badcase_top300/target_img_list.txt"
BUSINESS_VAL_ROOT = "/horizon-bucket/perception-dataprocess/4dlabel/static_visual/002_vggt_data/evc_data/business_data/driving_data/distort_2000h_v2/eval1000clip_202603171836"
CONFIG_STD160K74_PATH = REPO_ROOT / "configs/exps/vggt/saturnv/evaluation/vggt_business_2000hv2_std160k74_lio.yaml"
TARGET_LIST_STD160K74_PATH = REPO_ROOT / "aidi/benchmarks/business_std160k74/target_img_list.txt"
TARGET_LIST_STD160K74_CFG = "aidi/benchmarks/business_std160k74/target_img_list.txt"
BUSINESS_STD160K_VAL_ROOT = "/horizon-bucket/perception-dataprocess/4dlabel/static_visual/002_vggt_data/evc_data/business_data/driving_data/distort_2000h_v1/eval1000clip_202603051529"


class VggtBusiness10ViewBadcaseBenchmarkTests(unittest.TestCase):
    def _assert_target_list(self, path: Path, expected_len: int):
        targets = [line.strip() for line in path.read_text().splitlines() if line.strip()]

        self.assertEqual(len(targets), expected_len)
        self.assertEqual(len(set(targets)), expected_len)
        self.assertTrue(all(target.startswith(BUSINESS_VAL_ROOT) for target in targets))
        self.assertTrue(all("/images/" in target and target.endswith(".jpg") for target in targets))
        return targets

    def _assert_10view_config(self, config_path: Path, target_list_cfg: str, metrics_file: str):
        cfg = load_resolved_config(config_path)
        val_loader = cfg["val_dataloader_cfg"]
        val_dataset = val_loader["dataset_cfg"]
        scene_cfg = val_dataset["metaset_cfgs"][0]
        sample_cfg = val_dataset["metaset_cfgs"][1]

        self.assertEqual(val_dataset["meta_roots"], [BUSINESS_VAL_ROOT])
        self.assertEqual(scene_cfg["target_img_list_file"], target_list_cfg)
        self.assertEqual(sample_cfg["target_img_list_file"], target_list_cfg)
        self.assertEqual(sample_cfg["source_type"], "SEQUENTIAL")
        self.assertEqual(sample_cfg["sequential_stride"], 1)
        self.assertEqual(sample_cfg["extra_src_pool"], 0)
        self.assertTrue(sample_cfg["closest_using_t"])
        self.assertTrue(sample_cfg["use_3ddr"])
        self.assertEqual(val_loader["batch_sampler_cfg"]["n_srcs_list"], [9])
        self.assertEqual(val_loader["batch_sampler_cfg"]["n_srcs_prob"], [1.0])
        self.assertEqual(val_loader["sampler_cfg"]["type"], "DistributedSequentialSampler")
        self.assertEqual(cfg["runner_cfg"]["evaluator_cfg"]["metrics_file"], metrics_file)

    def test_target_list_freezes_top100_business_badcases(self):
        self._assert_target_list(TARGET_LIST_PATH, 100)

    def test_eval_config_reproduces_10view_same_camera_protocol(self):
        self._assert_10view_config(CONFIG_PATH, TARGET_LIST_CFG, "metrics_business_10view_badcase_top100.json")

    def test_target_list_freezes_top300_business_badcases(self):
        top100 = set(self._assert_target_list(TARGET_LIST_PATH, 100))
        top300 = set(self._assert_target_list(TARGET_LIST_TOP300_PATH, 300))

        self.assertTrue(top100.issubset(top300))

    def test_top300_eval_config_reproduces_10view_same_camera_protocol(self):
        self._assert_10view_config(CONFIG_TOP300_PATH, TARGET_LIST_TOP300_CFG, "metrics_business_10view_badcase_top300.json")

    def test_target_img_list_sampler_has_direct_index_fast_path(self):
        source = (REPO_ROOT / "easyvolcap/dataloaders/datasamplers.py").read_text()

        self.assertIn("_fast_filter_subdataset_by_img_list", source)
        self.assertIn("_load_img_filter_realpaths_ordered", source)
        self.assertIn("_load_img_filter_paths_ordered", source)
        self.assertIn("_parse_img_path", source)
        self.assertIn("target_parent not in (raw_data_parent, data_parent)", source)
        self.assertIn("raw_data_parent", source)
        self.assertIn("parse_dirs.append('images')", source)
        self.assertIn("camera_names.index(camera_name)", source)
        self.assertIn("candidate_name == frame_name", source)
        self.assertIn("target/img list filtering matched no samples", source)
        self.assertIn("view_ind * n_latents + frame_ind", source)

    def test_std160k74_target_list_freezes_standard_vggt_reference_targets(self):
        targets = [line.strip() for line in TARGET_LIST_STD160K74_PATH.read_text().splitlines() if line.strip()]

        self.assertEqual(len(targets), 74)
        self.assertEqual(len(set(targets)), 74)
        self.assertTrue(all(target.startswith(BUSINESS_STD160K_VAL_ROOT) for target in targets))
        self.assertTrue(all("/images/00/" in target and target.endswith(".jpg") for target in targets))

    def test_std160k74_eval_config_matches_pi3_standard_val_protocol(self):
        cfg = load_resolved_config(CONFIG_STD160K74_PATH)
        val_loader = cfg["val_dataloader_cfg"]
        val_dataset = val_loader["dataset_cfg"]
        scene_cfg = val_dataset["metaset_cfgs"][0]
        sample_cfg = val_dataset["metaset_cfgs"][1]

        self.assertEqual(val_dataset["meta_roots"], [BUSINESS_STD160K_VAL_ROOT])
        self.assertEqual(scene_cfg["target_img_list_file"], TARGET_LIST_STD160K74_CFG)
        self.assertEqual(sample_cfg["target_img_list_file"], TARGET_LIST_STD160K74_CFG)
        self.assertEqual(sample_cfg["source_type"], "MULTIVIEWSEQ")
        self.assertTrue(sample_cfg["use_3ddr"])
        self.assertEqual(val_loader["batch_sampler_cfg"]["n_srcs_list"], [35])
        self.assertEqual(val_loader["batch_sampler_cfg"]["n_srcs_prob"], [1.0])
        self.assertEqual(val_loader["sampler_cfg"]["type"], "DistributedSequentialSampler")
        self.assertEqual(cfg["runner_cfg"]["evaluator_cfg"]["metrics_file"], "metrics_business_std160k74.json")


if __name__ == "__main__":
    unittest.main()
