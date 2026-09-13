from pathlib import Path
import unittest

from aidi.scripts.baselines.eval_config_utils import load_resolved_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/exps/vggt/saturnv/base/vggt_mvseq_distort_2000hv2_lio.yaml"
DATASET_CONFIG_PATH = REPO_ROOT / "configs/datasets/saturnv_distortv2_2000h_260330.yaml"
SUBMIT_SCRIPT_PATH = REPO_ROOT / "aidi/scripts/vggt/submit_vggt_business_2000hv2_5090_control.sh"
SAMPLER_PATH = REPO_ROOT / "easyvolcap/models/samplers/vggt_sampler.py"


class VggtBusiness2000hV2ControlTests(unittest.TestCase):
    def test_config_matches_historical_data_logic(self):
        cfg = load_resolved_config(CONFIG_PATH)

        train_dataset = cfg["dataloader_cfg"]["dataset_cfg"]
        val_dataset = cfg["val_dataloader_cfg"]["dataset_cfg"]
        sampler_cfg = cfg["model_cfg"]["sampler_cfg"]

        self.assertEqual(
            train_dataset["meta_roots"],
            [
                "/horizon-bucket/perception-dataprocess/4dlabel/static_visual/002_vggt_data/evc_data/business_data/driving_data/distort_2000h_v2/train_202603171836"
            ],
        )
        self.assertEqual(
            val_dataset["meta_roots"],
            [
                "/horizon-bucket/perception-dataprocess/4dlabel/static_visual/002_vggt_data/evc_data/business_data/driving_data/distort_2000h_v2/eval1000clip_202603171836"
            ],
        )

        train_scene_cfg = train_dataset["metaset_cfgs"][0]
        train_clip_cfg = train_dataset["metaset_cfgs"][-1]
        val_scene_cfg = val_dataset["metaset_cfgs"][0]
        val_clip_cfg = val_dataset["metaset_cfgs"][-1]

        self.assertEqual(train_scene_cfg["view_sample"], [0, 6, 1])
        self.assertEqual(val_scene_cfg["view_sample"], [0, 6, 1])
        self.assertEqual(train_clip_cfg["source_type"], "MULTIVIEWSEQ")
        self.assertEqual(train_clip_cfg["extra_src_pool"], 36)
        self.assertEqual(val_clip_cfg["extra_src_pool"], 5)
        self.assertEqual(val_dataset["data_roots_file"], "val_data_roots.txt")
        self.assertEqual(cfg["dataloader_cfg"]["batch_sampler_cfg"]["n_srcs_list"], list(range(1, 24)))
        self.assertEqual(cfg["val_dataloader_cfg"]["batch_sampler_cfg"]["n_srcs_list"], [23])

        self.assertIs(sampler_cfg["use_cam_emb"], True)
        self.assertIs(sampler_cfg["use_3ddr"], True)
        self.assertIs(sampler_cfg["use_chunkwise_bp_dpt_decoder"], True)
        self.assertIs(sampler_cfg["use_chunkwise_checkpoint"], True)
        self.assertIs(sampler_cfg["use_xyz_head"], False)
        self.assertIs(sampler_cfg["use_dptpose_as_xyz"], True)

    def test_submit_defaults_to_2x8_generation_queue(self):
        script = SUBMIT_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn('CONFIG=${CONFIG:-"vggt/saturnv/base/vggt_mvseq_distort_2000hv2_lio"}', script)
        self.assertIn('CLUSTER=${CLUSTER:-"project-5090-4dlabel-depthgt-generation-bcloud"}', script)
        self.assertIn("TOTAL_GPU=${TOTAL_GPU:-16}", script)
        self.assertIn("SAVE_TO_AIDI=${SAVE_TO_AIDI:-1}", script)
        self.assertIn("DRY_SUBMIT=${DRY_SUBMIT:-0}", script)
        self.assertIn("--dry_run", script)
        self.assertIn("runner_cfg.resume=True", script)
        self.assertIn("model_cfg.sampler_cfg.use_checkpoint=True", script)
        self.assertIn("vggt/business_2000hv2/", script)

    def test_sampler_supports_depth_pose_xyz_supervision_path(self):
        source = SAMPLER_PATH.read_text(encoding="utf-8")

        self.assertIn("use_dptpose_as_xyz: bool = False", source)
        self.assertIn("self.use_dptpose_as_xyz = use_dptpose_as_xyz", source)
        self.assertIn("if self.use_dptpose_as_xyz:", source)
        self.assertIn("w2c, ixt = decode_camera_params(", source)
        self.assertIn("xyz_map = ray_o + ray_d * dpt_map", source)
        self.assertIn("xyz_cnf = torch.full_like(dpt_cnf, 2)", source)
        self.assertIn("or self.use_dptpose_as_xyz", source)


if __name__ == "__main__":
    unittest.main()
