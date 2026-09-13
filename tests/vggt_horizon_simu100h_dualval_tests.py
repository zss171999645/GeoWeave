from pathlib import Path
import unittest

from aidi.scripts.baselines.eval_config_utils import load_resolved_config


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG_PATH = REPO_ROOT / "configs/exps/vggt/saturnv/base/vggt_mvseq_horizon_simu100h_20260424_dualval.yaml"
SIM_DATASET_CONFIG_PATH = REPO_ROOT / "configs/datasets/saturnv_horizon_simu100h_20260424.yaml"
SIM_VAL_CONFIG_PATH = REPO_ROOT / "configs/exps/vggt/saturnv/evaluation/vggt_horizon_simu100h_20260424.yaml"
BUSINESS_VAL_CONFIG_PATH = REPO_ROOT / "configs/exps/vggt/saturnv/evaluation/vggt_business_2000hv2_standard75_lio.yaml"
SUBMIT_SCRIPT_PATH = REPO_ROOT / "aidi/scripts/vggt/submit_vggt_horizon_simu100h_dualval_5090.sh"

SIM_ROOT = "/horizon-bucket/perception-dataprocess/4dlabel/static_visual/002_vggt_data/evc_data/horizon_simulation_data/driving_data/20260424"
BUSINESS_VAL_ROOT = "/horizon-bucket/perception-dataprocess/4dlabel/static_visual/002_vggt_data/evc_data/business_data/driving_data/distort_2000h_v2/eval1000clip_202603171836"
BUSINESS_75_LIST = "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/eval_lists/business_v2_standard75_from_hardsample_00021999_imglist.txt"
BUSINESS_2000HV2_CKPT = "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/vggt/finetune_model/basemodel/datascaleup/8x8_2e-5_resumebalancedatano1_2000hv2_use_dptpose_asxyz_liogt/clean.pt"


class VggtHorizonSimu100hDualValTests(unittest.TestCase):
    def test_training_config_matches_simu100h_resume_line(self):
        cfg = load_resolved_config(TRAIN_CONFIG_PATH)

        train_dataset = cfg["dataloader_cfg"]["dataset_cfg"]
        runner_cfg = cfg["runner_cfg"]
        sampler_cfg = cfg["model_cfg"]["sampler_cfg"]

        self.assertEqual(train_dataset["meta_roots"], [SIM_ROOT])
        self.assertEqual(train_dataset["data_roots_file"], "data_roots.txt")
        self.assertEqual(train_dataset["metaset_cfgs"][0]["dataset_name"], "horizon_simulation_data_100h")
        self.assertEqual(train_dataset["metaset_cfgs"][0]["ratio"], 1.0)
        self.assertEqual(train_dataset["metaset_cfgs"][0]["view_sample"], [0, 6, 1])
        self.assertEqual(cfg["dataloader_cfg"]["batch_sampler_cfg"]["n_srcs_list"], list(range(1, 24)))
        self.assertEqual(cfg["dataloader_cfg"]["batch_sampler_cfg"]["n_srcs_prob"], [1.0] * 23)

        self.assertFalse(runner_cfg["resume"])
        self.assertEqual(runner_cfg["pretrained_model"], BUSINESS_2000HV2_CKPT)
        self.assertFalse(runner_cfg["pretrained_load_training_state"])
        self.assertEqual(runner_cfg["epochs"], 80)
        self.assertEqual(runner_cfg["ep_iter"], 1000)
        self.assertEqual(runner_cfg["scheduler_cfg"]["warmup_iters"], 5000)
        self.assertEqual(runner_cfg["eval_ep"], 1)
        self.assertEqual(runner_cfg["log_interval"], 10)
        self.assertTrue(runner_cfg["test_before_first_epoch"])

        self.assertEqual(
            [entry["name"] for entry in runner_cfg["main_val_cfgs"]],
            ["horizon_simu100h", "business_2000hv2_standard75"],
        )
        self.assertFalse(sampler_cfg["load_pretrained"])
        self.assertTrue(sampler_cfg["use_cam_emb"])
        self.assertTrue(sampler_cfg["use_3ddr"])
        self.assertEqual(sampler_cfg["use_3ddr_ratio"], 1.0)
        self.assertTrue(sampler_cfg["use_dptpose_as_xyz"])
        self.assertFalse(sampler_cfg["use_xyz_head"])

    def test_simu100h_validation_uses_historical_317_protocol_inputs(self):
        cfg = load_resolved_config(SIM_VAL_CONFIG_PATH)
        val_dataset = cfg["val_dataloader_cfg"]["dataset_cfg"]

        self.assertEqual(val_dataset["meta_roots"], [SIM_ROOT])
        self.assertEqual(val_dataset["data_roots_file"], "val_data_roots.txt")
        self.assertEqual(val_dataset["metaset_cfgs"][0]["img_list_file"], f"{SIM_ROOT}/eval_keyframes.txt")
        self.assertEqual(cfg["val_dataloader_cfg"]["sampler_cfg"]["frame_sample"], [0, None, 400])
        self.assertEqual(cfg["val_dataloader_cfg"]["batch_sampler_cfg"]["n_srcs_list"], [23])

    def test_business_validation_uses_standard75_target_filter(self):
        cfg = load_resolved_config(BUSINESS_VAL_CONFIG_PATH)
        val_dataset = cfg["val_dataloader_cfg"]["dataset_cfg"]

        self.assertEqual(val_dataset["meta_roots"], [BUSINESS_VAL_ROOT])
        self.assertEqual(val_dataset["data_roots_file"], "val_data_roots.txt")
        self.assertEqual(val_dataset["metaset_cfgs"][0]["target_img_list_file"], BUSINESS_75_LIST)
        self.assertEqual(val_dataset["metaset_cfgs"][1]["target_img_list_file"], BUSINESS_75_LIST)
        self.assertEqual(cfg["val_dataloader_cfg"]["sampler_cfg"]["type"], "DistributedSequentialSampler")
        self.assertEqual(cfg["val_dataloader_cfg"]["sampler_cfg"]["frame_sample"], [0, None, 1])
        self.assertEqual(cfg["val_dataloader_cfg"]["batch_sampler_cfg"]["n_srcs_list"], [23])

    def test_submit_defaults_to_4x8_langfang_queue(self):
        script = SUBMIT_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn('CONFIG=${CONFIG:-"vggt/saturnv/base/vggt_mvseq_horizon_simu100h_20260424_dualval"}', script)
        self.assertIn('CLUSTER=${CLUSTER:-"project-5090-4dlabel-perception-v2-acloud-langfang"}', script)
        self.assertIn("TOTAL_GPU=${TOTAL_GPU:-32}", script)
        self.assertIn("vggt_horizon_simu100h_resume2000hv2_dualval_4x8_", script)
        self.assertIn("vggt/horizon_simu100h_dualval/", script)
        self.assertIn("TB_MIRROR_INTERVAL=30", script)
        self.assertIn("DRY_SUBMIT=${DRY_SUBMIT:-0}", script)
        self.assertIn("--dry_run", script)


if __name__ == "__main__":
    unittest.main()
