from __future__ import annotations

import importlib.util
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import yaml


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aidi"
    / "scripts"
    / "vggt"
    / "run_unified_eval.py"
)


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"Missing unified eval runner: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("run_unified_eval", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VggtUnifiedEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_unified_eval_local_7scenes_root_uses_official_raw_layout(self):
        cfg_path = self.repo_root / "aidi" / "configs" / "vggt" / "unified_eval_local.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertEqual(
            cfg["datasets"]["7scenes"]["root"],
            "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/zinan.lv/dataset_point/7Scenes",
        )

    def test_unified_eval_local_exact_relpose_roots_use_pi3_benchmark_aliases(self):
        cfg_path = self.repo_root / "aidi" / "configs" / "vggt" / "unified_eval_local.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        base = "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/datasets/pi3_relpose_exact_benchmark"
        expected_roots = {
            "sintel": f"{base}/sintel",
            "tum": f"{base}/tum_dynamics",
            "scannet": f"{base}/scannet_exact100",
        }
        for dataset, expected_root in expected_roots.items():
            self.assertEqual(cfg["datasets"][dataset]["root"], expected_root)
            self.assertTrue(cfg["datasets"][dataset]["pose"]["require_official_layout"])

    def test_select_task_ids_uses_dataset_centric_switches(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "dataset_switches",
                    "gpu_ids": "0",
                    "output_root": "tmp/dataset_switches",
                },
                "datasets": {
                    "co3dv2": {
                        "pose": {"enabled": True},
                    },
                    "eth3d": {
                        "pointcloud": {"enabled": True},
                    },
                },
            },
            repo_root=self.repo_root,
        )

        task_ids = module.select_task_ids(config)

        self.assertEqual(task_ids, ["co3dv2_pose", "eth3d_pointcloud"])

    def test_build_re10k_pose_job_for_official_model(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "re10k_only",
                    "gpu_ids": "0",
                    "output_root": "tmp/re10k_only",
                },
                "datasets": {
                    "re10k": {
                        "pose": {
                            "enabled": True,
                            "limit_scenes": 3,
                        },
                    },
                },
            },
            repo_root=self.repo_root,
        )

        jobs = module.build_jobs(config, repo_root=self.repo_root)

        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job["task"], "re10k_pose")
        self.assertEqual(job["family"], "pose")
        self.assertEqual(job["dataset"], "re10k")
        self.assertTrue(job["command"][1].endswith("aidi/scripts/vggt/eval_vggt_re10k_pose_lightweight.py"))
        self.assertIn("--limit-scenes", job["command"])
        self.assertIn("3", job["command"])
        self.assertEqual(
            job["summary_path"],
            str(self.repo_root / "tmp" / "re10k_only" / "re10k_pose" / "re10k_pose_lightweight_summary.json"),
        )

    def test_build_sintel_pose_job_uses_relpose_distance_protocol(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "sintel_pose_only",
                    "gpu_ids": "2",
                    "device": "cuda:2",
                    "output_root": "tmp/sintel_pose_only",
                },
                "datasets": {
                    "sintel": {
                        "root": "/tmp/sintel_pose_root",
                        "pose": {
                            "enabled": True,
                            "limit_seqs": 5,
                            "require_official_layout": True,
                        },
                    },
                },
            },
            repo_root=self.repo_root,
        )

        jobs = module.build_jobs(config, repo_root=self.repo_root)

        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job["task"], "sintel_pose")
        self.assertEqual(job["family"], "pose")
        self.assertEqual(job["dataset"], "sintel")
        self.assertTrue(job["command"][1].endswith("aidi/scripts/baselines/eval_pi3_relpose_distance_protocol.py"))
        self.assertIn("--datasets", job["command"])
        self.assertIn("sintel", job["command"])
        self.assertIn("--sintel-root", job["command"])
        self.assertIn("/tmp/sintel_pose_root", job["command"])
        self.assertIn("--limit-seqs", job["command"])
        self.assertIn("5", job["command"])
        self.assertIn("--require-official-layout", job["command"])
        self.assertEqual(
            job["summary_path"],
            str(self.repo_root / "tmp" / "sintel_pose_only" / "sintel_pose" / "summary.json"),
        )

    def test_extract_summary_for_relpose_distance_payload(self):
        module = load_module()
        payload = {
            "datasets": ["sintel"],
            "results": [
                {
                    "dataset": "sintel",
                    "summary": {
                        "ATE": 0.074,
                        "RPE trans": 0.040,
                        "RPE rot": 0.282,
                        "num_sequences": 14,
                    },
                }
            ],
        }

        summary = module.extract_summary("sintel_pose", payload)

        self.assertEqual(
            summary,
            {
                "ATE": 0.074,
                "RPE trans": 0.040,
                "RPE rot": 0.282,
                "num_sequences": 14,
            },
        )

    def test_build_depth_job_for_custom_checkpoint(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "depth_custom",
                    "gpu_ids": "1",
                    "device": "cuda:1",
                    "output_root": "tmp/depth_custom",
                },
                "model": {
                    "kind": "custom",
                    "checkpoint": "/tmp/checkpoints/34.pt",
                    "config": "/tmp/configs/run.yaml",
                },
                "datasets": {
                    "sintel": {
                        "data_root": "/tmp/sintel",
                        "depth": {
                            "monodepth": {
                                "enabled": True,
                                "max_seqs": 2,
                                "max_frames_per_seq": 4,
                            },
                        },
                    },
                },
            },
            repo_root=self.repo_root,
        )

        jobs = module.build_jobs(config, repo_root=self.repo_root)

        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job["task"], "sintel_monodepth")
        self.assertTrue(job["command"][1].endswith("aidi/scripts/baselines/eval_pi3_depth_protocol.py"))
        self.assertIn("--mode", job["command"])
        self.assertIn("monodepth", job["command"])
        self.assertIn("--dataset", job["command"])
        self.assertIn("sintel", job["command"])
        self.assertIn("--data-root", job["command"])
        self.assertIn("/tmp/sintel", job["command"])
        self.assertIn("--vggt-model-tag", job["command"])
        self.assertIn("pt34", job["command"])
        self.assertIn("--vggt-pt34-ckpt", job["command"])
        self.assertIn("/tmp/checkpoints/34.pt", job["command"])
        self.assertIn("--vggt-config", job["command"])
        self.assertIn("/tmp/configs/run.yaml", job["command"])
        self.assertEqual(
            job["summary_path"],
            str(
                self.repo_root
                / "tmp"
                / "depth_custom"
                / "sintel_monodepth"
                / "sintel_monodepth_protocol_summary.json"
            ),
        )

    def test_build_co3dv2_depth_job_falls_back_when_primary_root_is_incomplete(self):
        module = load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            incomplete = root / "co3dv2_incomplete"
            complete = root / "co3dv2_complete"
            for scene in ("scene_a",):
                (incomplete / "apple" / scene / "images" / "000000").mkdir(parents=True)
                (incomplete / "apple" / scene / "depths" / "000000").mkdir(parents=True)
            for category, scenes in {"apple": ("scene_a",), "banana": ("scene_b",)}.items():
                for scene in scenes:
                    (complete / category / scene / "images" / "000000").mkdir(parents=True)
                    (complete / category / scene / "depths" / "000000").mkdir(parents=True)

            config = module.normalize_config(
                {
                    "run": {
                        "name": "co3dv2_depth_fallback",
                        "gpu_ids": "0",
                        "device": "cuda:0",
                        "output_root": "tmp/co3dv2_depth_fallback",
                    },
                    "datasets": {
                        "co3dv2": {
                            "data_root": str(incomplete),
                            "data_root_candidates": [str(incomplete), str(complete)],
                            "data_root_min_categories": 2,
                            "data_root_min_scenes": 2,
                            "depth": {
                                "monodepth": {
                                    "enabled": True,
                                },
                            },
                        },
                    },
                },
                repo_root=self.repo_root,
            )

            jobs = module.build_jobs(config, repo_root=self.repo_root)

        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        data_root_idx = job["command"].index("--data-root") + 1
        self.assertEqual(job["command"][data_root_idx], str(complete))

    def test_co3dv2_pose_job_passes_shared_image_root(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "co3dv2_pose_shared_root",
                    "gpu_ids": "0",
                    "device": "cuda:0",
                    "output_root": "tmp/co3dv2_pose_shared_root",
                },
                "datasets": {
                    "co3dv2": {
                        "shared_image_root": "/tmp/co3d_mirror",
                        "pose": {
                            "enabled": True,
                        },
                    },
                },
            },
            repo_root=self.repo_root,
        )

        jobs = module.build_jobs(config, repo_root=self.repo_root)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["task"], "co3dv2_pose")
        self.assertEqual(jobs[0]["env"]["CO3D_SHARED_IMAGE_ROOT"], "/tmp/co3d_mirror")

    def test_build_da3_pose_job_with_dataset_root(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "pose_da3",
                    "gpu_ids": "2",
                    "device": "cuda:2",
                    "output_root": "tmp/pose_da3",
                },
                "datasets": {
                    "7scenes": {
                        "root": "/tmp/7scenes",
                        "pose": {
                            "enabled": True,
                            "max_frames": 8,
                        },
                    },
                },
            },
            repo_root=self.repo_root,
        )

        jobs = module.build_jobs(config, repo_root=self.repo_root)

        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertTrue(job["command"][1].endswith("aidi/scripts/vggt/eval_da3_pose_benchmark.py"))
        self.assertIn("--datasets", job["command"])
        self.assertIn("7scenes", job["command"])
        self.assertIn("--7scenes-root", job["command"])
        self.assertIn("/tmp/7scenes", job["command"])
        self.assertIn("--max-frames", job["command"])
        self.assertIn("8", job["command"])
        self.assertIn("--image-preprocess-style", job["command"])
        style_idx = job["command"].index("--image-preprocess-style")
        self.assertEqual(job["command"][style_idx + 1], "official_vggt")

    def test_build_pointcloud_job_for_custom_checkpoint(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "dtu_custom",
                    "gpu_ids": "3",
                    "device": "cuda:3",
                    "output_root": "tmp/dtu_custom",
                },
                "model": {
                    "kind": "custom",
                    "checkpoint": "/tmp/checkpoints/60.pt",
                    "config": "/tmp/configs/custom.yaml",
                },
                "datasets": {
                    "dtu": {
                        "dataset_root": "/tmp/dtu",
                        "pointcloud": {
                            "enabled": True,
                        },
                    },
                },
            },
            repo_root=self.repo_root,
        )

        jobs = module.build_jobs(config, repo_root=self.repo_root)

        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertTrue(job["command"][-1].endswith("aidi/scripts/vggt/run_eval_dtu_mv_recon_pi3_style.sh"))
        self.assertEqual(job["env"]["VGGT_CHECKPOINT"], "/tmp/checkpoints/60.pt")
        self.assertEqual(job["env"]["VGGT_CONFIG"], "/tmp/configs/custom.yaml")
        self.assertEqual(job["env"]["DATASET_ROOT"], "/tmp/dtu")

    def test_build_pointcloud_jobs_for_7scenes_and_nrgbd_protocols(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "pointcloud_protocols",
                    "gpu_ids": "4",
                    "device": "cuda:4",
                    "output_root": "tmp/pointcloud_protocols",
                },
                "datasets": {
                    "7scenes": {
                        "dataset_root": "/tmp/7scenes",
                        "pointcloud": {
                            "sparse": {
                                "enabled": True,
                            },
                        },
                    },
                    "nrgbd": {
                        "dataset_root": "/tmp/nrgbd",
                        "pointcloud": {
                            "dense": {
                                "enabled": True,
                            },
                        },
                    },
                },
            },
            repo_root=self.repo_root,
        )

        jobs = module.build_jobs(config, repo_root=self.repo_root)

        self.assertEqual([job["task"] for job in jobs], ["7scenes_sparse_pointcloud", "nrgbd_dense_pointcloud"])
        self.assertTrue(jobs[0]["command"][-1].endswith("aidi/scripts/vggt/run_eval_7scenes_mv_recon_pi3_style.sh"))
        self.assertEqual(jobs[0]["env"]["PROTOCOL"], "sparse")
        self.assertEqual(jobs[0]["env"]["DATASET_ROOT"], "/tmp/7scenes")
        self.assertTrue(jobs[1]["command"][-1].endswith("aidi/scripts/vggt/run_eval_nrgbd_mv_recon_pi3_style.sh"))
        self.assertEqual(jobs[1]["env"]["PROTOCOL"], "dense")
        self.assertEqual(jobs[1]["env"]["DATASET_ROOT"], "/tmp/nrgbd")

    def test_build_jobs_round_robins_devices_when_multiple_gpus_are_requested(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "multi_gpu_round_robin",
                    "gpu_ids": "0,1,2",
                    "output_root": "tmp/multi_gpu_round_robin",
                },
                "datasets": {
                    "sintel": {
                        "data_root": "/tmp/sintel",
                        "depth": {
                            "videodepth": {"enabled": True},
                        },
                    },
                    "bonn": {
                        "data_root": "/tmp/bonn",
                        "depth": {
                            "videodepth": {"enabled": True},
                        },
                    },
                    "7scenes": {
                        "dataset_root": "/tmp/7scenes",
                        "pointcloud": {
                            "dense": {"enabled": True},
                        },
                    },
                },
            },
            repo_root=self.repo_root,
        )

        jobs = module.build_jobs(config, repo_root=self.repo_root)

        self.assertEqual([job["task"] for job in jobs], ["sintel_videodepth", "bonn_videodepth", "7scenes_dense_pointcloud"])
        self.assertEqual(jobs[0]["command"][jobs[0]["command"].index("--device") + 1], "cuda:0")
        self.assertEqual(jobs[1]["command"][jobs[1]["command"].index("--device") + 1], "cuda:1")
        self.assertEqual(jobs[2]["env"]["DEVICE"], "cuda:2")

    def test_legacy_top_level_keys_are_rejected(self):
        module = load_module()

        with self.assertRaisesRegex(ValueError, "Legacy unified-eval config keys are no longer supported"):
            module.normalize_config(
                {
                    "run": {
                        "name": "legacy_rejected",
                        "gpu_ids": "0",
                        "output_root": "tmp/legacy_rejected",
                    },
                    "tasks": {
                        "re10k_pose": {"enabled": True},
                    },
                },
                repo_root=self.repo_root,
            )

    def test_execute_jobs_dry_run_lists_planned_tasks(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "dry_run_records",
                    "gpu_ids": "0",
                    "output_root": "tmp/dry_run_records",
                },
                "datasets": {
                    "7scenes": {
                        "root": "/tmp/7scenes_pose_root",
                        "pose": {
                            "enabled": True,
                        },
                    },
                },
            },
            repo_root=self.repo_root,
        )
        config["run"]["dry_run"] = True
        jobs = module.build_jobs(config, repo_root=self.repo_root)

        aggregate = module.execute_jobs(config, jobs, repo_root=self.repo_root)

        self.assertEqual(aggregate["run_name"], "dry_run_records")
        self.assertEqual(len(aggregate["tasks"]), 1)
        self.assertEqual(aggregate["tasks"][0]["task"], "7scenes_pose")
        self.assertEqual(aggregate["tasks"][0]["status"], "PLANNED")

    def test_dataset_centric_config_supports_multiple_task_types_per_dataset(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "dataset_centric_7scenes",
                    "gpu_ids": "0,1,2,3",
                    "output_root": "tmp/dataset_centric_7scenes",
                },
                "datasets": {
                    "7scenes": {
                        "root": "/tmp/7scenes_pose_root",
                        "data_root": "/tmp/7scenes_depth_root",
                        "dataset_root": "/tmp/7scenes_pointcloud_root",
                        "pose": {
                            "enabled": True,
                            "max_frames": 8,
                        },
                        "depth": {
                            "videodepth": {
                                "enabled": True,
                            },
                        },
                        "pointcloud": {
                            "sparse": {
                                "enabled": True,
                            },
                            "dense": {
                                "enabled": True,
                            },
                        },
                    },
                },
            },
            repo_root=self.repo_root,
        )

        task_ids = module.select_task_ids(config)
        self.assertEqual(
            task_ids,
            [
                "7scenes_pose",
                "7scenes_videodepth",
                "7scenes_sparse_pointcloud",
                "7scenes_dense_pointcloud",
            ],
        )

        jobs = module.build_jobs(config, repo_root=self.repo_root)
        by_task = {job["task"]: job for job in jobs}
        self.assertIn("/tmp/7scenes_pose_root", by_task["7scenes_pose"]["command"])
        self.assertIn("/tmp/7scenes_depth_root", by_task["7scenes_videodepth"]["command"])
        self.assertEqual(by_task["7scenes_sparse_pointcloud"]["env"]["DATASET_ROOT"], "/tmp/7scenes_pointcloud_root")
        self.assertEqual(by_task["7scenes_dense_pointcloud"]["env"]["DATASET_ROOT"], "/tmp/7scenes_pointcloud_root")

    def test_dataset_centric_depth_shared_options_apply_to_each_enabled_mode(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "dataset_centric_kitti_depth",
                    "gpu_ids": "0,1",
                    "output_root": "tmp/dataset_centric_kitti_depth",
                },
                "datasets": {
                    "kitti": {
                        "data_root": "/tmp/kitti_depth_root",
                        "depth": {
                            "max_size": 1440,
                            "monodepth": {
                                "enabled": True,
                            },
                            "videodepth": {
                                "enabled": True,
                            },
                        },
                    },
                },
            },
            repo_root=self.repo_root,
        )

        jobs = module.build_jobs(config, repo_root=self.repo_root)
        self.assertEqual([job["task"] for job in jobs], ["kitti_monodepth", "kitti_videodepth"])
        for job in jobs:
            self.assertIn("/tmp/kitti_depth_root", job["command"])
            self.assertIn("--max-size", job["command"])
            self.assertIn("1440", job["command"])

    def test_official_vggt_videodepth_defaults_to_official_preprocess(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "official_videodepth_preprocess",
                    "gpu_ids": "0",
                    "output_root": "tmp/official_videodepth_preprocess",
                },
                "model": {"kind": "official"},
                "datasets": {
                    "blendedmvs": {
                        "data_root": "/tmp/blendedmvs",
                        "depth": {
                            "videodepth": {"enabled": True},
                        },
                    },
                },
            },
            repo_root=self.repo_root,
        )

        jobs = module.build_jobs(config, repo_root=self.repo_root)
        command = jobs[0]["command"]

        self.assertIn("--vggt-input-preprocess", command)
        idx = command.index("--vggt-input-preprocess")
        self.assertEqual(command[idx + 1], "official")

    def test_videodepth_eval_frame_indices_are_forwarded(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "videodepth_target_only",
                    "gpu_ids": "0",
                    "output_root": "tmp/videodepth_target_only",
                    "force_run_known_oom_tasks": ["vkitti2_videodepth"],
                },
                "datasets": {
                    "vkitti2": {
                        "data_root": "/tmp/vkitti2_candidate_pool",
                        "depth": {
                            "videodepth": {
                                "enabled": True,
                                "eval_frame_indices": "0,1,2,3,4,5",
                            },
                        },
                    },
                },
            },
            repo_root=self.repo_root,
        )

        jobs = module.build_jobs(config, repo_root=self.repo_root)
        command = jobs[0]["command"]

        self.assertIn("--eval-frame-indices", command)
        idx = command.index("--eval-frame-indices")
        self.assertEqual(command[idx + 1], "0,1,2,3,4,5")

    def test_plan_jobs_auto_skips_known_oom_official_videodepth_on_5090(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "skip_oom_on_5090",
                    "gpu_ids": "0",
                    "output_root": "tmp/skip_oom_on_5090",
                },
                "datasets": {
                    "kitti": {
                        "data_root": "/tmp/kitti",
                        "depth": {
                            "videodepth": {"enabled": True},
                        },
                    },
                    "7scenes": {
                        "data_root": "/tmp/7scenes",
                        "depth": {"videodepth": {"enabled": True}},
                    },
                    "blendedmvs": {
                        "data_root": "/tmp/blendedmvs",
                        "depth": {"videodepth": {"enabled": True}},
                    },
                    "vkitti2": {
                        "data_root": "/tmp/vkitti2",
                        "depth": {"videodepth": {"enabled": True}},
                    },
                },
            },
            repo_root=self.repo_root,
        )

        with mock.patch.object(module, "query_gpu_name_map", return_value={"0": "NVIDIA GeForce RTX 5090"}):
            jobs, skipped = module.plan_jobs(config, repo_root=self.repo_root)

        self.assertEqual([job["task"] for job in jobs], ["blendedmvs_videodepth"])
        self.assertEqual([record["task"] for record in skipped], ["kitti_videodepth", "7scenes_videodepth", "vkitti2_videodepth"])
        for record in skipped:
            self.assertEqual(record["status"], "SKIPPED")
            self.assertIn("known official videodepth OOM on 5090", record["skip_reason"])

    def test_plan_jobs_allows_force_running_known_oom_task(self):
        module = load_module()
        config = module.normalize_config(
            {
                "run": {
                    "name": "force_run_oom_task",
                    "gpu_ids": "0",
                    "output_root": "tmp/force_run_oom_task",
                    "force_run_known_oom_tasks": ["kitti_videodepth"],
                },
                "datasets": {
                    "kitti": {
                        "data_root": "/tmp/kitti",
                        "depth": {
                            "videodepth": {"enabled": True},
                        },
                    },
                },
            },
            repo_root=self.repo_root,
        )

        with mock.patch.object(module, "query_gpu_name_map", return_value={"0": "NVIDIA GeForce RTX 5090"}):
            jobs, skipped = module.plan_jobs(config, repo_root=self.repo_root)

        self.assertEqual([job["task"] for job in jobs], ["kitti_videodepth"])
        self.assertEqual(skipped, [])

    def test_canonical_metric_brief_surfaces_skipped_reason(self):
        module = load_module()
        brief = module.canonical_metric_brief(
            {
                "task": "kitti_videodepth",
                "status": "SKIPPED",
                "skip_reason": "known official videodepth OOM on 5090",
                "summary": {},
            }
        )

        self.assertIn("SKIPPED", brief)
        self.assertIn("known official videodepth OOM on 5090", brief)

    def test_task_registry_excludes_internal_legacy_tasks(self):
        module = load_module()
        for task_id in (
            "tum_rgbd_pose",
            "scannet_full_pose",
            "waymo_pose",
            "kitti_pose",
            "vkitti2_pose",
            "scannet_ecc2_pose",
        ):
            self.assertNotIn(task_id, module.TASK_REGISTRY)

    def test_print_task_list_uses_external_facing_columns(self):
        module = load_module()
        buf = StringIO()
        with redirect_stdout(buf):
            module.print_task_list()
        text = buf.getvalue()

        self.assertIn("task,family,dataset,variant", text.splitlines()[0])
        self.assertNotIn("protocol", text.splitlines()[0])
        self.assertNotIn("adapter", text.splitlines()[0])
        self.assertNotIn("waymo_pose", text)
        self.assertIn("re10k_pose,pose,re10k,pose", text)
        self.assertIn("sintel_pose,pose,sintel,pose", text)
        self.assertIn("tum_pose,pose,tum,pose", text)
        self.assertIn("scannet_pose,pose,scannet,pose", text)
        self.assertIn("co3dv2_monodepth,depth,co3dv2,monodepth", text)
        self.assertIn("7scenes_sparse_pointcloud,pointcloud,7scenes,sparse_pointcloud", text)
        self.assertIn("nrgbd_dense_pointcloud,pointcloud,nrgbd,dense_pointcloud", text)

    def test_write_aggregate_artifacts(self):
        module = load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payloads = {
                "re10k_pose": {"summary": {"Auc30": 0.9}},
                "sintel_monodepth": {"summary": {"AbsRel": 0.1}},
                "dtu_pointcloud": {"summary": {"overall": 1.2}},
            }
            records = []
            for task_id, payload in payloads.items():
                summary_path = root / f"{task_id}.json"
                summary_path.write_text(json.dumps(payload), encoding="utf-8")
                spec = module.TASK_REGISTRY[task_id]
                records.append(
                    {
                        "task": task_id,
                        "family": spec.family,
                        "dataset": spec.dataset,
                        "variant": spec.variant,
                        "summary_path": str(summary_path),
                        "payload": payload,
                        "summary": payload["summary"],
                    }
                )

            module.write_aggregate_artifacts(
                output_root=root,
                run_name="aggregate_test",
                records=records,
            )
            aggregate = json.loads((root / "aggregate_summary.json").read_text(encoding="utf-8"))
            markdown = (root / "aggregate_summary.md").read_text(encoding="utf-8")

        self.assertEqual(aggregate["run_name"], "aggregate_test")
        self.assertEqual([item["task"] for item in aggregate["tasks"]], list(payloads.keys()))
        self.assertIn("## pose", markdown.lower())
        self.assertIn("re10k_pose", markdown)
        self.assertIn("sintel_monodepth", markdown)
        self.assertIn("dtu_pointcloud", markdown)
        self.assertIn("| Task | Dataset | Variant | Summary |", markdown)
        self.assertNotIn("Protocol", markdown)

    def test_tracked_unified_eval_configs_do_not_use_container_local_da3_paths(self):
        local_prefix = "/home/feng01.zhou/da3_benchmark_dataset_20260420"
        tracked_configs = [
            self.repo_root / "tmp" / "unified_eval_official_dev5090_gpu0_20260422.yaml",
            self.repo_root / "tmp" / "unified_eval_official_dev5090_gpu0_remaining_20260422.yaml",
            self.repo_root / "tmp" / "unified_pose_fix_dev5090_gpu3_da3_20260422.yaml",
            self.repo_root / "tmp" / "unified_pose_fix_dev5090_gpu6_da3_misc_20260422.yaml",
        ]

        for config_path in tracked_configs:
            text = config_path.read_text(encoding="utf-8")
            self.assertNotIn(local_prefix, text, msg=f"{config_path} still uses container-local DA3 paths")


if __name__ == "__main__":
    unittest.main()
