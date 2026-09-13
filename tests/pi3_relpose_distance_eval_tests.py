from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from tempfile import TemporaryDirectory
from unittest import mock
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aidi"
    / "scripts"
    / "baselines"
    / "eval_pi3_relpose_distance_protocol.py"
)


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"Missing evaluator script: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("eval_pi3_relpose_distance_protocol", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Pi3RelposeDistanceEvalTests(unittest.TestCase):
    def test_module_import_bootstraps_repo_root(self):
        module_root = str(MODULE_PATH.parents[3])
        original_path = list(sys.path)
        try:
            sys.path[:] = [entry for entry in sys.path if entry != module_root]
            module = load_module()
            self.assertIn(module_root, sys.path)
            self.assertEqual(module.repo_root(), MODULE_PATH.parents[3])
        finally:
            sys.path[:] = original_path

    def test_default_dataset_roots_cover_pose_benchmarks(self):
        module = load_module()

        self.assertEqual(
            module.DEFAULT_DATA_ROOTS["sintel"],
            "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/datasets/pi3_relpose_exact_benchmark/sintel",
        )
        self.assertEqual(
            module.DEFAULT_DATA_ROOTS["tum"],
            "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/datasets/pi3_relpose_exact_benchmark/tum_dynamics",
        )
        self.assertEqual(
            module.DEFAULT_DATA_ROOTS["scannetv2"],
            "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/datasets/pi3_relpose_exact_benchmark/scannet_exact100",
        )
        self.assertEqual(
            module.DEFAULT_DATA_ROOTS["vkitti2"],
            "/horizon-bucket/saturn_v_dev/users/tao02.xie/datasets/vkitti2",
        )

    def test_paper_targets_match_known_values(self):
        module = load_module()

        self.assertEqual(module.PAPER_TARGETS["sintel"], {"ATE": 0.074, "RPE trans": 0.040, "RPE rot": 0.282})
        self.assertEqual(module.PAPER_TARGETS["tum"], {"ATE": 0.014, "RPE trans": 0.009, "RPE rot": 0.312})
        self.assertEqual(module.PAPER_TARGETS["scannetv2"], {"ATE": 0.031, "RPE trans": 0.013, "RPE rot": 0.347})
        self.assertEqual(module.PAPER_TARGETS["vkitti2"], {"ATE": 0.0, "RPE trans": 0.0, "RPE rot": 0.0})

    def test_parse_dataset_names_supports_all_and_csv(self):
        module = load_module()

        self.assertEqual(module.parse_dataset_names("all"), ["sintel", "tum", "scannetv2", "vkitti2"])
        self.assertEqual(module.parse_dataset_names("sintel,tum,vkitti2"), ["sintel", "tum", "vkitti2"])

    def test_parse_eval_frame_indices_supports_empty_and_csv(self):
        module = load_module()

        self.assertEqual(module.parse_eval_frame_indices(""), [])
        self.assertEqual(module.parse_eval_frame_indices("0,1,2,5"), [0, 1, 2, 5])

    def test_setup_args_accepts_eval_frame_indices_flag(self):
        module = load_module()
        argv = [
            "eval_pi3_relpose_distance_protocol.py",
            "--datasets",
            "sintel",
            "--eval-frame-indices",
            "0,1",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = module.setup_args()

        self.assertEqual(args.datasets, "sintel")
        self.assertEqual(args.eval_frame_indices, "0,1")

    def test_setup_args_accepts_native_sparse_pi3_flags(self):
        module = load_module()
        argv = [
            "eval_pi3_relpose_distance_protocol.py",
            "--model-family",
            "pi3",
            "--model-path",
            "/tmp/checkpoint.bin",
            "--pi3-config",
            "/tmp/config.yaml",
            "--pi3-model-impl",
            "native_sparse",
            "--pi3-native-root",
            "/tmp/pi3_training",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = module.setup_args()

        self.assertEqual(args.model_family, "pi3")
        self.assertEqual(args.model_path, "/tmp/checkpoint.bin")
        self.assertEqual(args.pi3_config, "/tmp/config.yaml")
        self.assertEqual(args.pi3_model_impl, "native_sparse")
        self.assertEqual(args.pi3_native_root, "/tmp/pi3_training")

    def test_load_model_runtime_dispatches_pi3_args(self):
        module = load_module()
        args = types.SimpleNamespace(model_family="pi3", model_path="/tmp/checkpoint.bin")

        with mock.patch.object(module, "load_pi3_model_runtime", return_value=("model", "ckpt")) as mocked:
            result = module.load_model_runtime(args, device="cuda:0")

        self.assertEqual(result, ("model", "ckpt"))
        mocked.assert_called_once_with(args, "cuda:0")

    def test_resolve_evo_utils_path_falls_back_to_existing_candidate(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            fallback = tmp_root / "bucket" / "pi3-official2_clean" / "relpose" / "evo_utils.py"
            fallback.parent.mkdir(parents=True)
            fallback.write_text("# stub\n", encoding="utf-8")

            with mock.patch.object(
                module,
                "DEFAULT_PI3_RELPOSE_EVO_UTILS_CANDIDATES",
                ("tmp/external_refs/pi3-official2/relpose/evo_utils.py", str(fallback)),
            ), mock.patch.object(module, "repo_root", return_value=tmp_root):
                resolved = module.resolve_evo_utils_path()

        self.assertEqual(resolved, fallback.resolve())

    def test_resolve_vggt_config_path_defaults_match_model_tag(self):
        module = load_module()

        self.assertEqual(
            module.resolve_vggt_config_path(model_tag="official", provided_path=""),
            "configs/exps/vggt/vggt_official_eval_paper.yaml",
        )
        self.assertEqual(
            module.resolve_vggt_config_path(model_tag="finetuned", provided_path=""),
            "aidi/configs/vggt/finetune_5090_sparse_topk2048_l9_19_p34_record.yaml",
        )
        self.assertEqual(
            module.resolve_vggt_config_path(model_tag="finetuned", provided_path="custom.yaml"),
            "custom.yaml",
        )

    def test_load_vggt_eval_config_expands_legacy_base_yaml(self):
        module = load_module()

        cfg = module.load_vggt_eval_config(
            module.repo_root() / "aidi/configs/vggt/finetune_5090_sparse_topk1024_l9_19_p34_record.yaml"
        )

        self.assertEqual(cfg.model_cfg.type, "OfficialVGGTModel")
        self.assertTrue(cfg.model_cfg.vggt_cfg.enable_point)
        self.assertFalse(cfg.model_cfg.vggt_cfg.enable_track)
        self.assertEqual(cfg.model_cfg.vggt_cfg.depth_head_cfg.type, "DPTHeadWithChunkwiseBP")
        self.assertEqual(cfg.model_cfg.vggt_cfg.point_head_cfg.type, "DPTHeadWithChunkwiseBP")
        self.assertTrue(cfg.model_cfg.vggt_cfg.indexer_cfg.enabled)
        self.assertTrue(cfg.model_cfg.vggt_cfg.indexer_cfg.inference_sparse)
        self.assertEqual(cfg.model_cfg.vggt_cfg.indexer_cfg.indexer_layers, "9-19")
        self.assertEqual(cfg.model_cfg.vggt_cfg.indexer_cfg.topk, 1024)

    def test_build_vggt_model_cfg_for_custom_ckpt_clears_component_ckpts_and_applies_topk(self):
        module = load_module()

        raw_model_cfg = {
            "agg_ckpt": "/tmp/agg.pt",
            "cam_ckpt": "/tmp/cam.pt",
            "xyz_ckpt": "/tmp/xyz.pt",
            "dpt_ckpt": "/tmp/dpt.pt",
            "tra_ckpt": "/tmp/tra.pt",
            "vggt_cfg": {"indexer_cfg": {"topk": 2048}},
        }

        model_cfg = module.build_vggt_model_cfg(
            raw_model_cfg=raw_model_cfg,
            model_tag="finetuned",
            official_ckpt_root="/tmp/official",
            topk_override=1024,
            clear_component_ckpts=True,
        )

        self.assertEqual(model_cfg.pretrained_path, "")
        self.assertEqual(model_cfg.agg_ckpt, "")
        self.assertEqual(model_cfg.cam_ckpt, "")
        self.assertEqual(model_cfg.xyz_ckpt, "")
        self.assertEqual(model_cfg.dpt_ckpt, "")
        self.assertEqual(model_cfg.tra_ckpt, "")
        self.assertEqual(model_cfg.vggt_cfg.indexer_cfg.topk, 1024)

    def test_build_vggt_model_cfg_applies_indexer_runtime_overrides(self):
        module = load_module()

        raw_model_cfg = {
            "vggt_cfg": {
                "indexer_cfg": {
                    "topk": 2048,
                    "topk_block": 256,
                    "topk_merge_blocks": 8,
                    "source_downsample_enabled": False,
                    "source_downsample_factor": 1,
                    "source_downsample_strategy": "none",
                    "source_downsample_query_chunk": 0,
                    "source_downsample_coarse_topk": 16,
                    "source_downsample_coarse_ratio": 1.0,
                }
            }
        }

        model_cfg = module.build_vggt_model_cfg(
            raw_model_cfg=raw_model_cfg,
            model_tag="finetuned",
            official_ckpt_root="/tmp/official",
            topk_override=1024,
            clear_component_ckpts=True,
            indexer_overrides={
                "topk_block": 1536,
                "topk_merge_blocks": 18,
                "source_downsample_enabled": True,
                "source_downsample_factor": 2,
                "source_downsample_strategy": "legacy",
                "source_downsample_query_chunk": 4096,
                "source_downsample_coarse_topk": 0,
                "source_downsample_coarse_ratio": 1.1,
            },
        )

        indexer_cfg = model_cfg.vggt_cfg.indexer_cfg
        self.assertEqual(indexer_cfg.topk, 1024)
        self.assertEqual(indexer_cfg.topk_block, 1536)
        self.assertEqual(indexer_cfg.topk_merge_blocks, 18)
        self.assertTrue(indexer_cfg.source_downsample_enabled)
        self.assertEqual(indexer_cfg.source_downsample_factor, 2)
        self.assertEqual(indexer_cfg.source_downsample_strategy, "legacy")
        self.assertEqual(indexer_cfg.source_downsample_query_chunk, 4096)
        self.assertEqual(indexer_cfg.source_downsample_coarse_topk, 0)
        self.assertAlmostEqual(indexer_cfg.source_downsample_coarse_ratio, 1.1)

    def test_collect_vggt_indexer_overrides_ignores_unset_values(self):
        module = load_module()

        args = types.SimpleNamespace(
            vggt_topk_block_override=None,
            vggt_topk_merge_blocks_override=None,
            vggt_source_downsample_enabled=False,
            vggt_source_downsample_factor_override=None,
            vggt_source_downsample_strategy_override="",
            vggt_source_downsample_query_chunk_override=None,
            vggt_source_downsample_coarse_topk_override=None,
            vggt_source_downsample_coarse_ratio_override=None,
        )
        self.assertEqual(module.collect_vggt_indexer_overrides(args), {})

        args = types.SimpleNamespace(
            vggt_topk_block_override=1536,
            vggt_topk_merge_blocks_override=18,
            vggt_source_downsample_enabled=True,
            vggt_source_downsample_factor_override=2,
            vggt_source_downsample_strategy_override="legacy",
            vggt_source_downsample_query_chunk_override=4096,
            vggt_source_downsample_coarse_topk_override=0,
            vggt_source_downsample_coarse_ratio_override=1.1,
        )
        self.assertEqual(
            module.collect_vggt_indexer_overrides(args),
            {
                "topk_block": 1536,
                "topk_merge_blocks": 18,
                "source_downsample_enabled": True,
                "source_downsample_factor": 2,
                "source_downsample_strategy": "legacy",
                "source_downsample_query_chunk": 4096,
                "source_downsample_coarse_topk": 0,
                "source_downsample_coarse_ratio": 1.1,
            },
        )

    def test_build_vggt_model_cfg_applies_head_chunk_runtime_overrides(self):
        module = load_module()

        raw_model_cfg = {
            "vggt_cfg": {
                "depth_head_cfg": {"chunk_cfg": {"frames_chunk_size": 4}},
                "point_head_cfg": {"chunk_cfg": {"frames_chunk_size": 4}},
            }
        }

        model_cfg = module.build_vggt_model_cfg(
            raw_model_cfg=raw_model_cfg,
            model_tag="finetuned",
            official_ckpt_root="/tmp/official",
            topk_override=1024,
            clear_component_ckpts=True,
            indexer_overrides={},
            head_chunk_overrides={
                "depth_frames_chunk_size": 2,
                "point_frames_chunk_size": 2,
            },
        )

        self.assertEqual(model_cfg.vggt_cfg.depth_head_cfg.chunk_cfg.frames_chunk_size, 2)
        self.assertEqual(model_cfg.vggt_cfg.point_head_cfg.chunk_cfg.frames_chunk_size, 2)

    def test_collect_vggt_head_chunk_overrides_ignores_unset_values(self):
        module = load_module()

        args = types.SimpleNamespace(
            vggt_depth_frames_chunk_size_override=None,
            vggt_point_frames_chunk_size_override=None,
        )
        self.assertEqual(module.collect_vggt_head_chunk_overrides(args), {})

        args = types.SimpleNamespace(
            vggt_depth_frames_chunk_size_override=2,
            vggt_point_frames_chunk_size_override=2,
        )
        self.assertEqual(
            module.collect_vggt_head_chunk_overrides(args),
            {
                "depth_frames_chunk_size": 2,
                "point_frames_chunk_size": 2,
            },
        )

    def test_build_vggt_model_cfg_for_official_model_points_to_component_root(self):
        module = load_module()

        model_cfg = module.build_vggt_model_cfg(
            raw_model_cfg={"vggt_cfg": {"indexer_cfg": {"topk": 2048}}},
            model_tag="official",
            official_ckpt_root="/tmp/official_root",
            topk_override=0,
            clear_component_ckpts=False,
        )

        self.assertEqual(model_cfg.pretrained_path, "")
        self.assertEqual(model_cfg.agg_ckpt, "/tmp/official_root/aggregator.pt")
        self.assertEqual(model_cfg.cam_ckpt, "/tmp/official_root/camera.pt")
        self.assertEqual(model_cfg.xyz_ckpt, "/tmp/official_root/point.pt")
        self.assertEqual(model_cfg.dpt_ckpt, "/tmp/official_root/depth.pt")
        self.assertEqual(model_cfg.tra_ckpt, "/tmp/official_root/track.pt")

    def test_vggt_extrinsics_to_c2w_inverts_predicted_w2c_batch(self):
        module = load_module()

        extrinsics = np.array(
            [
                [[1.0, 0.0, 0.0, 1.5], [0.0, 1.0, 0.0, -2.0], [0.0, 0.0, 1.0, 0.5]],
                [[1.0, 0.0, 0.0, -3.0], [0.0, 1.0, 0.0, 4.0], [0.0, 0.0, 1.0, 2.0]],
            ],
            dtype=np.float64,
        )

        c2w = module.vggt_extrinsics_to_c2w(extrinsics)

        self.assertEqual(c2w.shape, (2, 4, 4))
        self.assertTrue(np.allclose(c2w[0, :3, 3], [-1.5, 2.0, -0.5]))
        self.assertTrue(np.allclose(c2w[1, :3, 3], [3.0, -4.0, -2.0]))

    def test_build_summary_averages_sequence_metrics(self):
        module = load_module()

        seq_metrics = [
            {"ATE": 0.10, "RPE trans": 0.20, "RPE rot": 0.30},
            {"ATE": 0.30, "RPE trans": 0.40, "RPE rot": 0.50},
        ]

        summary = module.build_dataset_summary(seq_metrics)

        self.assertTrue(np.isclose(summary["ATE"], 0.20))
        self.assertTrue(np.isclose(summary["RPE trans"], 0.30))
        self.assertTrue(np.isclose(summary["RPE rot"], 0.40))
        self.assertEqual(summary["num_sequences"], 2)

    def test_evaluate_dataset_skips_sequence_when_official_gt_traj_is_invalid(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "scannetv2"
            dataset_root.mkdir(parents=True)
            output_root = Path(tmpdir) / "outputs"
            args = types.SimpleNamespace(
                sintel_root=str(dataset_root),
                tum_root=str(dataset_root),
                scannet_root=str(dataset_root),
                require_official_layout=True,
                limit_seqs=0,
                pose_eval_stride=1,
                device="cpu",
                load_img_size=64,
                image_load_retries=1,
                image_load_retry_sleep=0.0,
                skip_plot=True,
                verbose=False,
            )

            good_gt = dataset_root / "scene_good_pose_90.txt"
            good_gt.write_text("1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n", encoding="utf-8")

            evo_utils = types.SimpleNamespace(
                get_tum_poses=mock.Mock(return_value=("pred_traj", "pred_ts")),
                save_tum_poses=mock.Mock(),
                load_traj=mock.Mock(side_effect=[np.linalg.LinAlgError("bad gt"), ("gt_traj", "gt_ts")]),
                eval_metrics=mock.Mock(return_value=(0.11, 0.22, 0.33)),
                plot_trajectory=mock.Mock(),
            )

            with mock.patch.object(module, "load_evo_utils_runtime", return_value=evo_utils), mock.patch.object(
                module, "detect_layout", return_value="official"
            ), mock.patch.object(
                module, "list_sequence_names", return_value=["scene_bad", "scene_good"]
            ), mock.patch.object(
                module,
                "load_sequence_inputs",
                side_effect=[
                    (["bad_frame.jpg"], [dataset_root / "scene_bad_pose_90.txt"]),
                    (["good_frame.jpg"], [good_gt]),
                ],
            ), mock.patch.object(
                module, "infer_cameras_c2w", return_value=np.zeros((1, 4, 4), dtype=np.float32)
            ), mock.patch.object(
                module, "resolve_device", return_value=types.SimpleNamespace(type="cpu")
            ):
                result = module.evaluate_dataset(
                    dataset_name="scannetv2",
                    args=args,
                    model=object(),
                    loaded_ckpt="dummy.safetensors",
                    output_root=output_root,
                )

        self.assertEqual(result["num_sequences"], 1)
        self.assertTrue(np.isclose(result["summary"]["ATE"], 0.11))
        self.assertTrue(np.isclose(result["summary"]["RPE trans"], 0.22))
        self.assertTrue(np.isclose(result["summary"]["RPE rot"], 0.33))
        self.assertEqual(evo_utils.load_traj.call_count, 2)
        self.assertEqual(evo_utils.eval_metrics.call_count, 1)
        self.assertEqual(evo_utils.eval_metrics.call_args.kwargs["seq"], "scene_good")

    def test_evaluate_dataset_can_score_only_requested_scannet_frames(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "scannetv2"
            dataset_root.mkdir(parents=True)
            output_root = Path(tmpdir) / "outputs"
            gt_path = dataset_root / "scene_good_pose_90.txt"
            poses = []
            for idx in range(10):
                pose = np.eye(4, dtype=np.float64)
                pose[0, 3] = float(idx)
                poses.append(" ".join(str(float(x)) for x in pose.reshape(-1)))
            gt_path.write_text("\n".join(poses) + "\n", encoding="utf-8")

            args = types.SimpleNamespace(
                sintel_root=str(dataset_root),
                tum_root=str(dataset_root),
                scannet_root=str(dataset_root),
                require_official_layout=True,
                limit_seqs=0,
                pose_eval_stride=1,
                eval_frame_indices="0,1,2,3,4,5",
                device="cpu",
                load_img_size=64,
                image_load_retries=1,
                image_load_retry_sleep=0.0,
                skip_plot=True,
                verbose=False,
            )

            pred_poses = np.stack([np.eye(4, dtype=np.float32) for _ in range(10)], axis=0)
            evo_utils = types.SimpleNamespace(
                get_tum_poses=mock.Mock(side_effect=[("pred_traj", "pred_ts"), ("gt_traj", "gt_ts")]),
                save_tum_poses=mock.Mock(),
                load_traj=mock.Mock(),
                eval_metrics=mock.Mock(return_value=(0.11, 0.22, 0.33)),
                plot_trajectory=mock.Mock(),
            )

            with mock.patch.object(module, "load_evo_utils_runtime", return_value=evo_utils), mock.patch.object(
                module, "detect_layout", return_value="official"
            ), mock.patch.object(
                module, "list_sequence_names", return_value=["scene_good"]
            ), mock.patch.object(
                module, "load_sequence_inputs", return_value=([f"frame_{idx}.jpg" for idx in range(10)], [gt_path])
            ), mock.patch.object(
                module, "infer_cameras_c2w", return_value=pred_poses
            ), mock.patch.object(
                module, "resolve_device", return_value=types.SimpleNamespace(type="cpu")
            ):
                result = module.evaluate_dataset(
                    dataset_name="scannetv2",
                    args=args,
                    model=object(),
                    loaded_ckpt="dummy.safetensors",
                    output_root=output_root,
                )

        self.assertEqual(result["num_sequences"], 1)
        self.assertEqual(evo_utils.load_traj.call_count, 0)
        self.assertEqual(evo_utils.get_tum_poses.call_count, 2)
        pred_subset = evo_utils.get_tum_poses.call_args_list[0].args[0]
        gt_subset = evo_utils.get_tum_poses.call_args_list[1].args[0]
        self.assertEqual(pred_subset.shape, (6, 4, 4))
        self.assertEqual(gt_subset.shape, (6, 4, 4))
        self.assertTrue(np.allclose(gt_subset[:, 0, 3], np.arange(6, dtype=np.float64)))

    def test_evaluate_dataset_can_score_only_requested_vkitti_frames(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "vkitti2"
            dataset_root.mkdir(parents=True)
            output_root = Path(tmpdir) / "outputs"
            gt_path = dataset_root / "scene_clone_pose_90.txt"
            poses = []
            for idx in range(10):
                pose = np.eye(4, dtype=np.float64)
                pose[1, 3] = float(idx)
                poses.append(" ".join(str(float(x)) for x in pose.reshape(-1)))
            gt_path.write_text("\n".join(poses) + "\n", encoding="utf-8")

            args = types.SimpleNamespace(
                sintel_root=str(dataset_root),
                tum_root=str(dataset_root),
                scannet_root=str(dataset_root),
                vkitti_root=str(dataset_root),
                require_official_layout=True,
                limit_seqs=0,
                pose_eval_stride=1,
                eval_frame_indices="0,1,2,3,4,5",
                device="cpu",
                load_img_size=64,
                image_load_retries=1,
                image_load_retry_sleep=0.0,
                skip_plot=True,
                verbose=False,
            )

            pred_poses = np.stack([np.eye(4, dtype=np.float32) for _ in range(10)], axis=0)
            evo_utils = types.SimpleNamespace(
                get_tum_poses=mock.Mock(side_effect=[("pred_traj", "pred_ts"), ("gt_traj", "gt_ts")]),
                save_tum_poses=mock.Mock(),
                load_traj=mock.Mock(),
                eval_metrics=mock.Mock(return_value=(0.41, 0.52, 0.63)),
                plot_trajectory=mock.Mock(),
            )

            with mock.patch.object(module, "load_evo_utils_runtime", return_value=evo_utils), mock.patch.object(
                module, "detect_layout", return_value="official"
            ), mock.patch.object(
                module, "list_sequence_names", return_value=["scene_clone"]
            ), mock.patch.object(
                module, "load_sequence_inputs", return_value=([f"frame_{idx}.jpg" for idx in range(10)], [gt_path])
            ), mock.patch.object(
                module, "infer_cameras_c2w", return_value=pred_poses
            ), mock.patch.object(
                module, "resolve_device", return_value=types.SimpleNamespace(type="cpu")
            ):
                result = module.evaluate_dataset(
                    dataset_name="vkitti2",
                    args=args,
                    model=object(),
                    loaded_ckpt="dummy.safetensors",
                    output_root=output_root,
                )

        self.assertEqual(result["num_sequences"], 1)
        self.assertEqual(evo_utils.load_traj.call_count, 0)
        self.assertEqual(evo_utils.get_tum_poses.call_count, 2)
        pred_subset = evo_utils.get_tum_poses.call_args_list[0].args[0]
        gt_subset = evo_utils.get_tum_poses.call_args_list[1].args[0]
        self.assertEqual(pred_subset.shape, (6, 4, 4))
        self.assertEqual(gt_subset.shape, (6, 4, 4))
        self.assertTrue(np.allclose(gt_subset[:, 1, 3], np.arange(6, dtype=np.float64)))

    def test_detect_layout_supports_tum_scannet_evc_and_vkitti_official(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            tum_root = root / "tum"
            (tum_root / "rgbd_dataset_freiburg1_360" / "images" / "00").mkdir(parents=True)
            (tum_root / "rgbd_dataset_freiburg1_360" / "cameras" / "00").mkdir(parents=True)
            tum_layout = module.detect_layout(module.DATASET_SPECS["tum"], tum_root)

            scan_root = root / "scannet"
            (scan_root / "scene0000_00" / "images").mkdir(parents=True)
            (scan_root / "scene0000_00" / "intri.yml").write_text("stub")
            (scan_root / "scene0000_00" / "extri.yml").write_text("stub")
            scan_layout = module.detect_layout(module.DATASET_SPECS["scannetv2"], scan_root)

            vkitti_root = root / "vkitti"
            (vkitti_root / "Scene01-clone-cam00__anchor0060__clean" / "color_90").mkdir(parents=True)
            (vkitti_root / "Scene01-clone-cam00__anchor0060__clean" / "pose_90.txt").write_text("stub")
            vkitti_layout = module.detect_layout(module.DATASET_SPECS["vkitti2"], vkitti_root)

        self.assertEqual(tum_layout, "evc")
        self.assertEqual(scan_layout, "evc")
        self.assertEqual(vkitti_layout, "official")

    def test_detect_layout_rejects_evc_when_official_layout_required(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tum_root = root / "tum"
            (tum_root / "rgbd_dataset_freiburg1_360" / "images" / "00").mkdir(parents=True)
            (tum_root / "rgbd_dataset_freiburg1_360" / "cameras" / "00").mkdir(parents=True)

            with self.assertRaisesRegex(RuntimeError, "official layout"):
                module.detect_layout(module.DATASET_SPECS["tum"], tum_root, require_official_layout=True)

    def test_build_tum_evc_image_map_indexes_files_by_stem(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            seq_root = Path(tmpdir) / "rgbd_dataset_freiburg1_360"
            image_dir = seq_root / "images" / "00"
            image_dir.mkdir(parents=True)
            (image_dir / "000010.jpg").write_bytes(b"jpg")
            (image_dir / "000020.png").write_bytes(b"png")

            image_map = module.build_tum_evc_image_map(seq_root)

        self.assertEqual(sorted(image_map.keys()), ["000010", "000020"])
        self.assertTrue(str(image_map["000010"]).endswith("000010.jpg"))
        self.assertTrue(str(image_map["000020"]).endswith("000020.png"))

    def test_build_scannet_evc_image_map_indexes_nested_frame_dirs(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            seq_root = Path(tmpdir) / "scene0000_00"
            frame_dir = seq_root / "images" / "000040"
            frame_dir.mkdir(parents=True)
            (frame_dir / "000000.jpg").write_bytes(b"jpg")

            image_map = module.build_scannet_evc_image_map(seq_root)

        self.assertEqual(list(image_map.keys()), ["000040"])
        self.assertTrue(str(image_map["000040"]).endswith("000000.jpg"))

    def test_load_sequence_inputs_for_tum_evc_matches_monst3r_stride3_prefix_sampling(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seq = "rgbd_dataset_freiburg1_360"
            image_dir = root / seq / "images" / "00"
            image_dir.mkdir(parents=True)

            cameras = {}
            for idx in range(300):
                name = f"{idx:06d}"
                (image_dir / f"{name}.jpg").write_bytes(b"jpg")
                rt = np.array([[1, 0, 0, float(idx)], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
                cameras[name] = types.SimpleNamespace(RT=rt)

            with mock.patch.object(module, "resolve_evc_cameras", return_value=cameras):
                image_names, gt_payload = module.load_sequence_inputs(
                    spec=module.DATASET_SPECS["tum"],
                    root=root,
                    seq=seq,
                    layout="evc",
                )

        self.assertEqual(len(image_names), 90)
        self.assertEqual(gt_payload[0].shape, (90, 4, 4))
        self.assertTrue(image_names[0].endswith("000000.jpg"))
        self.assertTrue(image_names[1].endswith("000003.jpg"))
        self.assertTrue(image_names[-1].endswith("000267.jpg"))

    def test_align_image_map_and_camera_names_uses_sorted_numeric_intersection(self):
        module = load_module()

        image_map = {
            "000020": Path("/tmp/20.jpg"),
            "000010": Path("/tmp/10.jpg"),
            "000030": Path("/tmp/30.jpg"),
        }
        ordered = module.align_image_map_and_camera_names(image_map=image_map, camera_names=["000020", "000010"])

        self.assertEqual(ordered, ["000010", "000020"])

    def test_camera_rt_to_c2w_inverts_w2c_pose(self):
        module = load_module()

        rt = np.array(
            [
                [1.0, 0.0, 0.0, 1.5],
                [0.0, 1.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, 0.5],
            ],
            dtype=np.float64,
        )

        c2w = module.camera_rt_to_c2w(rt)

        self.assertEqual(c2w.shape, (4, 4))
        self.assertTrue(np.allclose(c2w[:3, 3], [-1.5, 2.0, -0.5]))

    def test_camera_rt_to_c2w_keeps_scannet_evc_pose_as_c2w(self):
        module = load_module()

        rt = np.array(
            [
                [1.0, 0.0, 0.0, 1.5],
                [0.0, 1.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, 0.5],
            ],
            dtype=np.float64,
        )

        c2w = module.camera_rt_to_c2w(rt, dataset_name="scannetv2")

        self.assertEqual(c2w.shape, (4, 4))
        self.assertTrue(np.allclose(c2w[:3, 3], [1.5, -2.0, 0.5]))

    def test_build_c2w_stack_from_cameras_preserves_requested_order(self):
        module = load_module()

        cameras = {
            "000000": types.SimpleNamespace(RT=np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)),
            "000010": types.SimpleNamespace(RT=np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)),
        }

        stack = module.build_c2w_stack_from_cameras(cameras, ["000010", "000000"])

        self.assertEqual(stack.shape, (2, 4, 4))
        self.assertTrue(np.allclose(stack[0, :3, 3], [-2.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(stack[1, :3, 3], [-1.0, 0.0, 0.0]))

    def test_ensure_headless_matplotlib_backend_defaults_to_agg(self):
        module = load_module()

        original = os.environ.pop("MPLBACKEND", None)
        try:
            module.ensure_headless_matplotlib_backend()
            self.assertEqual(os.environ["MPLBACKEND"], "Agg")
        finally:
            if original is None:
                os.environ.pop("MPLBACKEND", None)
            else:
                os.environ["MPLBACKEND"] = original

    def test_ensure_headless_matplotlib_backend_overrides_evo_plot_backend(self):
        module = load_module()

        settings_mod = types.ModuleType("evo.tools.settings")
        settings_mod.SETTINGS = types.SimpleNamespace(plot_backend="TkAgg")
        tools_mod = types.ModuleType("evo.tools")
        tools_mod.settings = settings_mod
        evo_mod = types.ModuleType("evo")
        evo_mod.tools = tools_mod

        with mock.patch.dict(
            sys.modules,
            {
                "evo": evo_mod,
                "evo.tools": tools_mod,
                "evo.tools.settings": settings_mod,
            },
            clear=False,
        ):
            module.ensure_headless_matplotlib_backend()

        self.assertEqual(settings_mod.SETTINGS.plot_backend, "Agg")


if __name__ == "__main__":
    unittest.main()
