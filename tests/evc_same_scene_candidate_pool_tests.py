from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "aidi" / "scripts" / "baselines" / "build_evc_same_scene_candidate_pool_benchmark.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("build_evc_same_scene_candidate_pool_benchmark", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EVCSameSceneCandidatePoolTests(unittest.TestCase):
    def test_select_pool_keeps_clean_core_and_expands_by_overlap_bands(self):
        module = load_script_module()
        overlap_by_ref = {
            1: 0.95,
            2: 0.90,
            3: 0.85,
            4: 0.80,
            5: 0.75,
            6: 0.18,
            7: 0.15,
            8: 0.08,
            9: 0.06,
            10: 0.04,
            11: 0.02,
            12: 0.015,
            13: 0.009,
            14: 0.006,
            15: 0.004,
            16: 0.003,
            17: 0.002,
            18: 0.001,
            19: 0.0,
        }

        selection = module.select_same_scene_candidate_pool(
            ref_id=0,
            overlap_by_ref=overlap_by_ref,
            core_size=6,
            pool_sizes=[6, 10, 14, 20],
            clean_overlap_threshold=0.20,
            clean_temporal_dedup=1,
        )

        self.assertEqual(selection.core_ids, [0, 1, 2, 3, 4, 5])
        self.assertEqual(selection.pool_ids_by_size[6], [0, 1, 2, 3, 4, 5])
        self.assertEqual(selection.pool_ids_by_size[10], [0, 1, 2, 3, 4, 5, 6, 8, 11, 15])
        self.assertEqual(selection.pool_ids_by_size[14], [0, 1, 2, 3, 4, 5, 6, 8, 11, 15, 7, 9, 12, 16])
        self.assertEqual(
            selection.pool_ids_by_size[20],
            [0, 1, 2, 3, 4, 5, 6, 8, 11, 15, 7, 9, 12, 16, 10, 13, 17, 14, 18, 19],
        )
        self.assertEqual([row.band for row in selection.candidate_rows[:4]], ["near", "mid", "far", "tail"])
        self.assertEqual([row.band for row in selection.candidate_rows[4:8]], ["near", "mid", "far", "tail"])

    def test_select_pool_requires_enough_same_scene_candidates_for_largest_pool(self):
        module = load_script_module()
        overlap_by_ref = {
            1: 0.95,
            2: 0.90,
            3: 0.85,
            4: 0.80,
            5: 0.75,
            6: 0.18,
        }

        with self.assertRaises(ValueError):
            module.select_same_scene_candidate_pool(
                ref_id=0,
                overlap_by_ref=overlap_by_ref,
                core_size=6,
                pool_sizes=[6, 10],
                clean_overlap_threshold=0.20,
                clean_temporal_dedup=1,
            )

    def test_build_benchmark_materializes_pool_variants_and_eval_prefix_summary(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "vkitti2"
            scene_root = dataset_root / "Scene01" / "clone"
            (scene_root / "images" / "00").mkdir(parents=True)
            (scene_root / "depths" / "00").mkdir(parents=True)
            (scene_root / "cameras" / "00").mkdir(parents=True)
            output_root = Path(tmpdir) / "out"

            fake_scene_record = {
                "scene_name": "Scene01-clone-cam00",
                "frame_ids": list(range(20)),
                "color_paths": [],
                "depth_paths": [],
                "poses": np.zeros((20, 4, 4), dtype=np.float32),
                "intrinsics": [],
            }
            overlap_by_ref = {
                1: 0.95,
                2: 0.90,
                3: 0.85,
                4: 0.80,
                5: 0.75,
                6: 0.18,
                7: 0.15,
                8: 0.08,
                9: 0.06,
                10: 0.04,
                11: 0.02,
                12: 0.015,
                13: 0.009,
                14: 0.006,
                15: 0.004,
                16: 0.003,
                17: 0.002,
                18: 0.001,
                19: 0.0,
            }

            with mock.patch.object(module, "build_evc_camera_scene_record", return_value=fake_scene_record):
                with mock.patch.object(module, "build_scene_overlap_table", return_value={0: overlap_by_ref}):
                    with mock.patch.object(module, "materialize_tuple_sequence") as materialize_mock:
                        summary_path = module.build_same_scene_candidate_pool_benchmark(
                            dataset_root=dataset_root,
                            output_root=output_root,
                            scene_specs=["Scene01/clone"],
                            pool_sizes=[6, 10, 14, 20],
                            sample_stride=16,
                            depth_rel_tol=0.01,
                            clean_overlap_threshold=0.20,
                            clean_temporal_dedup=1,
                            anchor_quantiles=[0.1],
                            max_anchors_per_scene=1,
                            image_subdir="images/00",
                            depth_subdir="depths/00",
                            camera_subdir="cameras/00",
                            target_num_frames=90,
                            source_prestride=3,
                            write_preview_grids=False,
                        )

            self.assertEqual(materialize_mock.call_count, 4)
            ordered_id_lengths = [len(call.kwargs["ordered_ids"]) for call in materialize_mock.call_args_list]
            self.assertEqual(ordered_id_lengths, [6, 10, 14, 20])

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["num_scenes_kept"], 1)
            self.assertEqual(payload["num_anchors"], 1)
            self.assertEqual(payload["num_pool_tuples"], 4)
            self.assertEqual(payload["tuples"][0]["eval_frame_indices"], [0, 1, 2, 3, 4, 5])
            self.assertEqual(payload["tuples"][0]["pool_size"], 6)
            self.assertEqual(payload["tuples"][-1]["pool_size"], 20)

    def test_build_benchmark_candidate_first_records_eval_suffix_indices(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "vkitti2"
            scene_root = dataset_root / "Scene01" / "clone"
            (scene_root / "images" / "00").mkdir(parents=True)
            (scene_root / "depths" / "00").mkdir(parents=True)
            (scene_root / "cameras" / "00").mkdir(parents=True)
            output_root = Path(tmpdir) / "out"

            fake_scene_record = {
                "scene_name": "Scene01-clone-cam00",
                "frame_ids": list(range(20)),
                "color_paths": [],
                "depth_paths": [],
                "poses": np.zeros((20, 4, 4), dtype=np.float32),
                "intrinsics": [],
            }
            overlap_by_ref = {
                1: 0.95,
                2: 0.90,
                3: 0.85,
                4: 0.80,
                5: 0.75,
                6: 0.18,
                7: 0.15,
                8: 0.08,
                9: 0.06,
            }

            with mock.patch.object(module, "build_evc_camera_scene_record", return_value=fake_scene_record):
                with mock.patch.object(module, "build_scene_overlap_table", return_value={0: overlap_by_ref}):
                    with mock.patch.object(module, "materialize_tuple_sequence") as materialize_mock:
                        summary_path = module.build_same_scene_candidate_pool_benchmark(
                            dataset_root=dataset_root,
                            output_root=output_root,
                            scene_specs=["Scene01/clone"],
                            pool_sizes=[10],
                            sample_stride=16,
                            depth_rel_tol=0.01,
                            clean_overlap_threshold=0.20,
                            clean_temporal_dedup=1,
                            anchor_quantiles=[0.1],
                            max_anchors_per_scene=1,
                            image_subdir="images/00",
                            depth_subdir="depths/00",
                            camera_subdir="cameras/00",
                            target_num_frames=90,
                            source_prestride=3,
                            write_preview_grids=False,
                            order_mode="candidate_first",
                        )

            self.assertEqual(materialize_mock.call_args.kwargs["ordered_ids"], [6, 8, 7, 9, 0, 1, 2, 3, 4, 5])

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["order_mode"], "candidate_first")
            self.assertEqual(payload["tuples"][0]["eval_frame_indices"], [4, 5, 6, 7, 8, 9])
            self.assertEqual(payload["tuples"][0]["base_ordered_frame_ids"], [0, 1, 2, 3, 4, 5, 6, 8, 7, 9])
            self.assertEqual(payload["tuples"][0]["ordered_frame_ids"], [6, 8, 7, 9, 0, 1, 2, 3, 4, 5])

    def test_discover_generic_evc_scene_roots_accepts_direct_one_and_two_level_layouts(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            direct = root / "direct"
            one = root / "one_root" / "scene_a"
            two = root / "two_root" / "group" / "scene_b"
            for scene_root in (direct, one, two):
                (scene_root / "images" / "00").mkdir(parents=True)
                (scene_root / "depths" / "00").mkdir(parents=True)
                (scene_root / "cameras" / "00").mkdir(parents=True)

            self.assertEqual(module.discover_generic_evc_scene_roots(direct), [direct])
            self.assertEqual(module.discover_generic_evc_scene_roots(root / "one_root"), [one])
            self.assertEqual(module.discover_generic_evc_scene_roots(root / "two_root"), [two])

    def test_discover_nested_evc_scene_roots_accepts_direct_one_and_two_level_layouts(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            direct = root / "direct"
            one = root / "one_root" / "scene_a"
            two = root / "two_root" / "group" / "scene_b"
            for scene_root in (direct, one, two):
                (scene_root / "images" / "000000").mkdir(parents=True)
                (scene_root / "depths" / "000000").mkdir(parents=True)
                (scene_root / "intri.yml").write_text("intri\n", encoding="utf-8")
                (scene_root / "extri.yml").write_text("extri\n", encoding="utf-8")

            self.assertEqual(module.discover_nested_evc_scene_roots(direct), [direct])
            self.assertEqual(module.discover_nested_evc_scene_roots(root / "one_root"), [one])
            self.assertEqual(module.discover_nested_evc_scene_roots(root / "two_root"), [two])

    def test_build_benchmark_nested_evc_uses_nested_scene_record(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "co3dv2"
            scene_root = dataset_root / "cup" / "seq001"
            (scene_root / "images" / "000000").mkdir(parents=True)
            (scene_root / "depths" / "000000").mkdir(parents=True)
            (scene_root / "intri.yml").write_text("intri\n", encoding="utf-8")
            (scene_root / "extri.yml").write_text("extri\n", encoding="utf-8")
            output_root = Path(tmpdir) / "out"

            fake_scene_record = {
                "scene_name": "cup-seq001",
                "frame_ids": list(range(10)),
                "color_paths": [],
                "depth_paths": [],
                "poses": np.zeros((10, 4, 4), dtype=np.float32),
                "intrinsics": [],
            }
            overlap_by_ref = {
                1: 0.95,
                2: 0.90,
                3: 0.85,
                4: 0.80,
                5: 0.75,
                6: 0.18,
                7: 0.15,
                8: 0.08,
                9: 0.06,
            }

            with mock.patch.object(module, "build_nested_evc_camera_scene_record", return_value=fake_scene_record) as build_mock:
                with mock.patch.object(module, "build_scene_overlap_table", return_value={0: overlap_by_ref}):
                    with mock.patch.object(module, "materialize_tuple_sequence") as materialize_mock:
                        summary_path = module.build_same_scene_candidate_pool_benchmark(
                            dataset_root=dataset_root,
                            output_root=output_root,
                            scene_specs=[],
                            pool_sizes=[10],
                            sample_stride=16,
                            depth_rel_tol=0.01,
                            clean_overlap_threshold=0.20,
                            clean_temporal_dedup=1,
                            anchor_quantiles=[0.1],
                            max_anchors_per_scene=1,
                            image_subdir="images",
                            depth_subdir="depths",
                            camera_subdir="",
                            target_num_frames=90,
                            source_prestride=3,
                            write_preview_grids=False,
                            dataset_name="co3dv2",
                            source_layout="nested_evc",
                        )

            build_kwargs = build_mock.call_args.kwargs
            self.assertEqual(build_kwargs["seq_root"], scene_root.resolve())
            self.assertEqual(build_kwargs["dataset_name"], "co3dv2")
            self.assertEqual(build_kwargs["image_rel_path"], "images")
            self.assertEqual(build_kwargs["depth_rel_path"], "depths")
            self.assertEqual(materialize_mock.call_count, 1)

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["source_layout"], "nested_evc")
            self.assertEqual(payload["config"]["image_subdir"], "images")
            self.assertEqual(payload["config"]["depth_subdir"], "depths")

    def test_build_benchmark_generic_dataset_uses_dataset_name_and_scene_slug(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "tum"
            scene_root = dataset_root / "rgbd_dataset_freiburg1_room"
            (scene_root / "images" / "00").mkdir(parents=True)
            (scene_root / "depths" / "00").mkdir(parents=True)
            (scene_root / "cameras" / "00").mkdir(parents=True)
            output_root = Path(tmpdir) / "out"

            fake_scene_record = {
                "scene_name": "rgbd_dataset_freiburg1_room",
                "frame_ids": list(range(10)),
                "color_paths": [],
                "depth_paths": [],
                "poses": np.zeros((10, 4, 4), dtype=np.float32),
                "intrinsics": [],
            }
            overlap_by_ref = {
                1: 0.95,
                2: 0.90,
                3: 0.85,
                4: 0.80,
                5: 0.75,
                6: 0.18,
                7: 0.15,
                8: 0.08,
                9: 0.06,
            }

            with mock.patch.object(module, "build_evc_camera_scene_record", return_value=fake_scene_record) as build_mock:
                with mock.patch.object(module, "build_scene_overlap_table", return_value={0: overlap_by_ref}):
                    with mock.patch.object(module, "materialize_tuple_sequence") as materialize_mock:
                        summary_path = module.build_same_scene_candidate_pool_benchmark(
                            dataset_root=dataset_root,
                            output_root=output_root,
                            scene_specs=[],
                            pool_sizes=[10],
                            sample_stride=16,
                            depth_rel_tol=0.01,
                            clean_overlap_threshold=0.20,
                            clean_temporal_dedup=1,
                            anchor_quantiles=[0.1],
                            max_anchors_per_scene=1,
                            image_subdir="images/00",
                            depth_subdir="depths/00",
                            camera_subdir="cameras/00",
                            target_num_frames=90,
                            source_prestride=3,
                            write_preview_grids=False,
                            dataset_name="tum",
                        )

            build_kwargs = build_mock.call_args.kwargs
            self.assertEqual(build_kwargs["dataset_name"], "tum")
            self.assertEqual(build_kwargs["scene_name"], "rgbd_dataset_freiburg1_room")
            self.assertEqual(materialize_mock.call_count, 1)

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["dataset"], "tum")
            self.assertEqual(payload["scenes"][0]["scene_name"], "rgbd_dataset_freiburg1_room")

    def test_build_benchmark_exact_official_uses_exact_scene_record(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "scannet_exact"
            scene_root = dataset_root / "scene0707_00"
            (scene_root / "color_90").mkdir(parents=True)
            (scene_root / "depth_90").mkdir(parents=True)
            (scene_root / "pose_90.txt").write_text("", encoding="utf-8")
            output_root = Path(tmpdir) / "out"

            fake_scene_record = {
                "scene_name": "scene0707_00",
                "frame_ids": list(range(10)),
                "color_paths": [],
                "depth_paths": [],
                "poses": np.zeros((10, 4, 4), dtype=np.float32),
                "intrinsics": [],
            }
            overlap_by_ref = {
                1: 0.95,
                2: 0.90,
                3: 0.85,
                4: 0.80,
                5: 0.75,
                6: 0.18,
                7: 0.15,
                8: 0.08,
                9: 0.06,
            }

            with mock.patch.object(module, "build_exact_official_scene_record", return_value=fake_scene_record) as build_mock:
                with mock.patch.object(module, "build_scene_overlap_table", return_value={0: overlap_by_ref}):
                    with mock.patch.object(module, "materialize_tuple_sequence") as materialize_mock:
                        summary_path = module.build_same_scene_candidate_pool_benchmark(
                            dataset_root=dataset_root,
                            output_root=output_root,
                            scene_specs=[],
                            pool_sizes=[10],
                            sample_stride=16,
                            depth_rel_tol=0.01,
                            clean_overlap_threshold=0.20,
                            clean_temporal_dedup=1,
                            anchor_quantiles=[0.1],
                            max_anchors_per_scene=1,
                            image_subdir="images/00",
                            depth_subdir="depths/00",
                            camera_subdir="cameras/00",
                            target_num_frames=90,
                            source_prestride=3,
                            write_preview_grids=False,
                            dataset_name="scannetv2",
                            source_layout="exact_official",
                        )

            self.assertEqual(build_mock.call_args.args[0], scene_root.resolve())
            self.assertEqual(materialize_mock.call_count, 1)

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["source_layout"], "exact_official")
            self.assertEqual(payload["scenes"][0]["scene_name"], "scene0707_00")


if __name__ == "__main__":
    unittest.main()
