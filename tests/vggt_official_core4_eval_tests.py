from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import yaml


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "aidi"
    / "scripts"
    / "vggt"
    / "run_official_core4_eval.py"
)


def load_module():
    if not SCRIPT_PATH.is_file():
        raise AssertionError(f"Missing unified core4 eval runner: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("run_official_core4_eval", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VggtOfficialCore4EvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_build_jobs_for_official_core4_eval(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "official_core4",
                    "gpu_ids": "0",
                    "save_root": "/tmp/vggt-save",
                    "output_root": "tmp/official_core4",
                },
                "model": {
                    "kind": "official",
                },
                "datasets": {
                    "re10k": {"enabled": True},
                    "co3dv2": {"enabled": True},
                    "dtu": {"enabled": True},
                    "eth3d": {"enabled": True},
                },
            },
            repo_root=self.repo_root,
        )

        jobs = module.build_jobs(config, repo_root=self.repo_root)
        self.assertEqual([job["name"] for job in jobs], ["re10k", "co3dv2", "dtu", "eth3d"])

        re10k_job = jobs[0]
        self.assertTrue(
            re10k_job["command"][1].endswith("aidi/scripts/vggt/eval_vggt_re10k_pose_lightweight.py")
        )
        self.assertEqual(re10k_job["env"], {})
        self.assertIn("--limit-scenes", re10k_job["command"])
        self.assertIn("0", re10k_job["command"])
        self.assertEqual(
            re10k_job["summary_path"],
            str(self.repo_root / "tmp" / "official_core4" / "re10k" / "re10k_pose_lightweight_summary.json"),
        )

        co3dv2_job = jobs[1]
        self.assertTrue(co3dv2_job["command"][-1].endswith("aidi/scripts/vggt/run_co3dv2_official_upstream.sh"))
        self.assertNotIn("CHECKPOINT", co3dv2_job["env"])
        self.assertNotIn("CONFIG", co3dv2_job["env"])

        dtu_job = jobs[2]
        self.assertTrue(dtu_job["command"][-1].endswith("aidi/scripts/vggt/run_eval_dtu_mv_recon_pi3_style.sh"))
        self.assertEqual(dtu_job["env"]["MODEL_TAG"], "official")

        eth3d_job = jobs[3]
        self.assertTrue(eth3d_job["command"][-1].endswith("aidi/scripts/vggt/run_eval_eth3d_mv_recon_pi3_style.sh"))
        self.assertEqual(eth3d_job["env"]["MODEL_TAG"], "official")

    def test_build_jobs_for_custom_checkpoint_fans_out_to_each_dataset(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "custom_core4",
                    "gpu_ids": "2",
                    "save_root": "/tmp/vggt-save",
                    "output_root": "tmp/custom_core4",
                },
                "model": {
                    "kind": "custom",
                    "checkpoint": "/tmp/checkpoints/34.pt",
                    "config": "/tmp/records/run.yaml",
                },
                "datasets": {
                    "re10k": {"enabled": True},
                    "co3dv2": {"enabled": True},
                    "dtu": {"enabled": True},
                    "eth3d": {"enabled": True},
                },
            },
            repo_root=self.repo_root,
        )

        jobs = module.build_jobs(config, repo_root=self.repo_root)

        re10k_job = jobs[0]
        self.assertTrue(
            re10k_job["command"][1].endswith("aidi/scripts/vggt/eval_vggt_re10k_pose_lightweight.py")
        )
        self.assertIn("--model-path", re10k_job["command"])
        self.assertIn("/tmp/checkpoints/34.pt", re10k_job["command"])
        self.assertIn("--config", re10k_job["command"])
        self.assertIn("/tmp/records/run.yaml", re10k_job["command"])
        self.assertIn("--model-tag", re10k_job["command"])
        self.assertIn("34", re10k_job["command"])

        co3dv2_job = jobs[1]
        self.assertEqual(co3dv2_job["env"]["CHECKPOINT"], "/tmp/checkpoints/34.pt")
        self.assertEqual(co3dv2_job["env"]["CONFIG"], "/tmp/records/run.yaml")

        dtu_job = jobs[2]
        self.assertEqual(dtu_job["env"]["VGGT_CHECKPOINT"], "/tmp/checkpoints/34.pt")
        self.assertEqual(dtu_job["env"]["VGGT_CONFIG"], "/tmp/records/run.yaml")

        eth3d_job = jobs[3]
        self.assertEqual(eth3d_job["env"]["VGGT_CHECKPOINT"], "/tmp/checkpoints/34.pt")
        self.assertEqual(eth3d_job["env"]["VGGT_CONFIG"], "/tmp/records/run.yaml")

    def test_build_jobs_honors_enabled_dataset_flags(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "dtu_only",
                    "gpu_ids": "1",
                    "save_root": "/tmp/vggt-save",
                    "output_root": "tmp/dtu_only",
                },
                "model": {
                    "kind": "official",
                },
                "datasets": {
                    "re10k": {"enabled": False},
                    "co3dv2": {"enabled": False},
                    "dtu": {"enabled": True},
                    "eth3d": {"enabled": False},
                },
            },
            repo_root=self.repo_root,
        )

        jobs = module.build_jobs(config, repo_root=self.repo_root)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["name"], "dtu")
        self.assertTrue(jobs[0]["command"][-1].endswith("aidi/scripts/vggt/run_eval_dtu_mv_recon_pi3_style.sh"))

    def test_build_jobs_support_lightweight_re10k_impl(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "re10k_lightweight",
                    "gpu_ids": "0",
                    "save_root": "/tmp/vggt-save",
                    "output_root": "tmp/re10k_lightweight",
                },
                "model": {
                    "kind": "official",
                },
                "datasets": {
                    "re10k": {
                        "enabled": True,
                        "impl": "lightweight",
                        "load_img_size": 518,
                    },
                    "co3dv2": {"enabled": False},
                    "dtu": {"enabled": False},
                    "eth3d": {"enabled": False},
                },
            },
            repo_root=self.repo_root,
        )

        jobs = module.build_jobs(config, repo_root=self.repo_root)

        self.assertEqual(len(jobs), 1)
        re10k_job = jobs[0]
        self.assertTrue(
            re10k_job["command"][1].endswith("aidi/scripts/vggt/eval_vggt_re10k_pose_lightweight.py")
        )
        self.assertEqual(re10k_job["command"][0], "python3")
        self.assertEqual(
            re10k_job["summary_path"],
            str(self.repo_root / "tmp" / "re10k_lightweight" / "re10k" / "re10k_pose_lightweight_summary.json"),
        )

    def test_build_jobs_support_lightweight_re10k_multi_gpu(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "re10k_lightweight_multi",
                    "gpu_ids": "1,3,5",
                    "save_root": "/tmp/vggt-save",
                    "output_root": "tmp/re10k_lightweight_multi",
                },
                "model": {
                    "kind": "official",
                },
                "datasets": {
                    "re10k": {
                        "enabled": True,
                        "impl": "lightweight",
                        "load_img_size": 518,
                    },
                    "co3dv2": {"enabled": False},
                    "dtu": {"enabled": False},
                    "eth3d": {"enabled": False},
                },
            },
            repo_root=self.repo_root,
        )

        jobs = module.build_jobs(config, repo_root=self.repo_root)

        self.assertEqual(len(jobs), 1)
        re10k_job = jobs[0]
        self.assertIn("--devices", re10k_job["command"])
        self.assertIn("cuda:1,cuda:3,cuda:5", re10k_job["command"])
        self.assertNotIn("--device", re10k_job["command"])

    def test_build_jobs_propagate_re10k_lightweight_limit_scenes(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "re10k_lightweight_limit",
                    "gpu_ids": "0",
                    "save_root": "/tmp/vggt-save",
                    "output_root": "tmp/re10k_lightweight_limit",
                },
                "model": {
                    "kind": "official",
                },
                "datasets": {
                    "re10k": {
                        "enabled": True,
                        "impl": "lightweight",
                        "limit_scenes": 8,
                    },
                    "co3dv2": {"enabled": False},
                    "dtu": {"enabled": False},
                    "eth3d": {"enabled": False},
                },
            },
            repo_root=self.repo_root,
        )

        jobs = module.build_jobs(config, repo_root=self.repo_root)

        self.assertEqual(len(jobs), 1)
        re10k_job = jobs[0]
        self.assertIn("--limit-scenes", re10k_job["command"])
        self.assertIn("8", re10k_job["command"])

    def test_build_jobs_propagate_configured_eval_resolutions(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "custom_sizes",
                    "gpu_ids": "3",
                    "save_root": "/tmp/vggt-save",
                    "output_root": "tmp/custom_sizes",
                },
                "model": {
                    "kind": "official",
                },
                "datasets": {
                    "re10k": {"enabled": True, "load_img_size": 672},
                    "co3dv2": {"enabled": True, "load_img_size": 644},
                    "dtu": {"enabled": True, "load_img_size": 602},
                    "eth3d": {"enabled": True, "load_img_size": 588},
                },
            },
            repo_root=self.repo_root,
        )

        jobs = module.build_jobs(config, repo_root=self.repo_root)

        re10k_job = jobs[0]
        load_img_size_idx = re10k_job["command"].index("--load-img-size")
        self.assertEqual(re10k_job["command"][load_img_size_idx + 1], "672")

        co3dv2_job = jobs[1]
        self.assertEqual(co3dv2_job["env"]["LOAD_IMG_SIZE"], "644")

        dtu_job = jobs[2]
        self.assertEqual(dtu_job["env"]["LOAD_IMG_SIZE"], "602")

        eth3d_job = jobs[3]
        self.assertEqual(eth3d_job["env"]["LOAD_IMG_SIZE"], "588")

    def test_auto_gpu_ids_fall_back_to_zero_when_nvidia_smi_is_missing(self):
        module = load_module()
        with mock.patch.object(module.subprocess, "check_output", side_effect=FileNotFoundError("nvidia-smi")):
            self.assertEqual(module.resolve_gpu_ids({"gpu_ids": "auto"}), "0")

    def test_default_eval_resolutions_match_local_official_protocol(self):
        module = load_module()
        self.assertEqual(module.DEFAULT_CONFIG["datasets"]["re10k"]["impl"], "lightweight")
        self.assertEqual(module.DEFAULT_CONFIG["datasets"]["re10k"]["limit_scenes"], 0)
        self.assertEqual(module.DEFAULT_CONFIG["datasets"]["re10k"]["load_img_size"], 518)
        self.assertEqual(module.DEFAULT_CONFIG["datasets"]["co3dv2"]["load_img_size"], 518)
        self.assertEqual(module.DEFAULT_CONFIG["datasets"]["dtu"]["load_img_size"], 518)
        self.assertEqual(module.DEFAULT_CONFIG["datasets"]["eth3d"]["load_img_size"], 518)

        re10k_cfg_path = self.repo_root / "configs" / "exps" / "vggt" / "evaluation" / "re10k_pose.yaml"
        co3dv2_cfg_path = self.repo_root / "configs" / "exps" / "vggt" / "evaluation" / "co3dv2.yaml"

        with re10k_cfg_path.open("r", encoding="utf-8") as f:
            re10k_cfg = yaml.safe_load(f.read().replace("{{fileBasenameNoExtension}}", '"re10k_pose"'))
        with co3dv2_cfg_path.open("r", encoding="utf-8") as f:
            co3dv2_cfg = yaml.safe_load(f.read().replace("{{fileBasenameNoExtension}}", '"co3dv2"'))

        re10k_point_cfg = re10k_cfg["val_dataloader_cfg"]["dataset_cfg"]["metaset_cfgs"][1]
        co3dv2_point_cfg = co3dv2_cfg["val_dataloader_cfg"]["dataset_cfg"]["metaset_cfgs"][1]

        self.assertEqual(re10k_point_cfg["vggt_official_preprocess_mode"], "crop")
        self.assertEqual(re10k_point_cfg["vggt_official_target_size"], 518)
        self.assertEqual(co3dv2_point_cfg["max_size"], 518)


if __name__ == "__main__":
    unittest.main()
