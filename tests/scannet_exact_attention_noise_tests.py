from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = (
    REPO_ROOT
    / "aidi"
    / "scripts"
    / "baselines"
    / "overlap_noise_seq_map_utils.py"
)
SCRIPT_PATH = (
    REPO_ROOT
    / "aidi"
    / "scripts"
    / "baselines"
    / "build_scannet_exact_attention_noise_benchmark.py"
)


def load_module():
    if not HELPER_PATH.is_file():
        raise AssertionError(f"Missing helper module: {HELPER_PATH}")
    spec = importlib.util.spec_from_file_location("overlap_noise_seq_map_utils", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_script_module():
    if not SCRIPT_PATH.is_file():
        raise AssertionError(f"Missing generator script: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("build_scannet_exact_attention_noise_benchmark", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScanNetExactAttentionNoiseTests(unittest.TestCase):
    def test_parse_pose_txt_returns_stacked_4x4_poses(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            pose_path = Path(tmpdir) / "pose_90.txt"
            pose_path.write_text(
                "\n".join(
                    [
                        "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
                        "1 0 0 3 0 1 0 0 0 0 1 0 0 0 0 1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            poses = module.parse_pose_txt(pose_path)

        self.assertEqual(poses.shape, (2, 4, 4))
        self.assertEqual(float(poses[1, 0, 3]), 3.0)

    def test_build_scene_record_reads_color_depth_pose_and_frame_ids(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_root = Path(tmpdir) / "scene0000_00"
            color_dir = seq_root / "color_90"
            depth_dir = seq_root / "depth_90"
            color_dir.mkdir(parents=True)
            depth_dir.mkdir(parents=True)

            for idx in range(3):
                Image.fromarray(np.full((4, 5, 3), idx, dtype=np.uint8)).save(color_dir / f"frame_{idx:04d}.jpg")
                Image.fromarray(np.full((4, 5), 1000 + idx, dtype=np.uint16)).save(depth_dir / f"frame_{idx:04d}.png")

            (seq_root / "pose_90.txt").write_text(
                "\n".join(["1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"] * 3) + "\n",
                encoding="utf-8",
            )

            scene = module.build_exact_official_scene_record(seq_root)

        self.assertEqual(scene["scene_name"], "scene0000_00")
        self.assertEqual(scene["frame_ids"], [0, 1, 2])
        self.assertEqual(len(scene["color_paths"]), 3)
        self.assertEqual(len(scene["depth_paths"]), 3)
        self.assertEqual(scene["poses"].shape, (3, 4, 4))

    def test_build_scene_record_prefers_symlink_target_frame_ids_when_available(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = Path(tmpdir) / "raw_scene"
            images_dir = raw_root / "images"
            depths_dir = raw_root / "depths"
            images_dir.mkdir(parents=True)
            depths_dir.mkdir(parents=True)
            Image.fromarray(np.full((4, 5, 3), 7, dtype=np.uint8)).save(images_dir / "00117.jpg")
            Image.fromarray(np.full((4, 5), 1000, dtype=np.uint16)).save(depths_dir / "00117.png")

            seq_root = Path(tmpdir) / "scene0000_00"
            color_dir = seq_root / "color_90"
            depth_dir = seq_root / "depth_90"
            color_dir.mkdir(parents=True)
            depth_dir.mkdir(parents=True)
            (color_dir / "frame_0000.jpg").symlink_to(images_dir / "00117.jpg")
            (depth_dir / "frame_0000.png").symlink_to(depths_dir / "00117.png")
            (seq_root / "pose_90.txt").write_text(
                "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n",
                encoding="utf-8",
            )

            scene = module.build_exact_official_scene_record(seq_root)

        self.assertEqual(scene["frame_ids"], [117])

    def test_select_attention_noise_pair_reuses_front_clean_views_and_replaces_tail(self):
        module = load_module()
        overlap_by_ref = {
            11: 0.92,
            31: 0.88,
            51: 0.84,
            71: 0.80,
            91: 0.76,
            111: 0.72,
            141: 0.68,
            171: 0.63,
            201: 0.59,
            231: 0.004,
            251: 0.003,
            271: 0.002,
            291: 0.001,
        }

        pair = module.select_attention_noise_pair(
            ref_id=10,
            overlap_by_ref=overlap_by_ref,
            clean_overlap_threshold=0.20,
            noise_overlap_threshold=0.01,
            clean_temporal_dedup=20,
        )

        self.assertEqual(pair.clean_ids, [10, 11, 31, 51, 71, 91, 111, 141, 171, 201])
        self.assertEqual(pair.noise_ids, [10, 11, 31, 51, 71, 91, 231, 251, 271, 291])

    def test_materialize_tuple_sequence_writes_10_frame_official_layout(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "scene0000_00"
            color_dir = source_root / "color_90"
            depth_dir = source_root / "depth_90"
            color_dir.mkdir(parents=True)
            depth_dir.mkdir(parents=True)
            poses = []
            for idx in range(12):
                Image.fromarray(np.full((4, 5, 3), idx, dtype=np.uint8)).save(color_dir / f"frame_{idx:04d}.jpg")
                Image.fromarray(np.full((4, 5), 1000 + idx, dtype=np.uint16)).save(depth_dir / f"frame_{idx:04d}.png")
                pose = np.eye(4, dtype=np.float64)
                pose[0, 3] = idx
                poses.append(" ".join(str(float(x)) for x in pose.reshape(-1)))
            (source_root / "pose_90.txt").write_text("\n".join(poses) + "\n", encoding="utf-8")

            scene_record = module.build_exact_official_scene_record(source_root)
            output_root = Path(tmpdir) / "diag"
            tuple_root = module.materialize_tuple_sequence(
                scene_record=scene_record,
                tuple_name="scene0000_00__anchor0003__clean",
                ordered_ids=[3, 4, 5, 6, 7, 8, 9, 10, 11, 1],
                output_root=output_root,
            )

            self.assertEqual(tuple_root.name, "scene0000_00__anchor0003__clean")
            self.assertEqual(len(list((tuple_root / "color_90").glob("*.jpg"))), 10)
            self.assertEqual(len(list((tuple_root / "depth_90").glob("*.png"))), 10)
            pose_lines = (tuple_root / "pose_90.txt").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(pose_lines), 10)
            self.assertEqual(float(pose_lines[0].split()[3]), 3.0)
            self.assertEqual(float(pose_lines[-1].split()[3]), 1.0)

    def test_build_scene_overlap_table_assigns_high_overlap_to_identical_pose_and_low_overlap_to_far_pose(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_root = Path(tmpdir) / "scene0000_00"
            color_dir = seq_root / "color_90"
            depth_dir = seq_root / "depth_90"
            intrinsic_dir = seq_root / "depth_intrinsics"
            color_dir.mkdir(parents=True)
            depth_dir.mkdir(parents=True)
            intrinsic_dir.mkdir(parents=True)

            for idx in range(3):
                Image.fromarray(np.full((32, 32, 3), idx, dtype=np.uint8)).save(color_dir / f"frame_{idx:04d}.jpg")
                Image.fromarray(np.full((32, 32), 1000, dtype=np.uint16)).save(depth_dir / f"frame_{idx:04d}.png")

            K = np.array(
                [
                    [50.0, 0.0, 15.5, 0.0],
                    [0.0, 50.0, 15.5, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            np.savetxt(intrinsic_dir / "depth_intrinsic.txt", K, fmt="%.6f")

            pose0 = np.eye(4, dtype=np.float64)
            pose1 = np.eye(4, dtype=np.float64)
            pose2 = np.eye(4, dtype=np.float64)
            pose2[0, 3] = 100.0
            (seq_root / "pose_90.txt").write_text(
                "\n".join(" ".join(str(float(x)) for x in pose.reshape(-1)) for pose in (pose0, pose1, pose2)) + "\n",
                encoding="utf-8",
            )

            scene_record = module.build_exact_official_scene_record(seq_root)
            overlap_table = module.build_scene_overlap_table(
                scene_record=scene_record,
                sample_stride=8,
                depth_rel_tol=0.01,
            )

        self.assertGreater(overlap_table[0][1], 0.95)
        self.assertLess(overlap_table[0][2], 0.01)

    def test_select_anchor_ref_ids_uses_fixed_quantiles(self):
        module = load_module()

        selected = module.select_anchor_ref_ids(
            eligible_ref_ids=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            anchor_quantiles=[0.1, 0.3, 0.5, 0.7, 0.9],
            max_anchors=5,
        )

        self.assertEqual(selected, [20, 40, 60, 80, 100])

    def test_write_attention_noise_summary_reports_counts(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = module.write_attention_noise_summary(
                output_root=Path(tmpdir),
                config={
                    "source_exact_root": "/tmp/source",
                    "clean_overlap_threshold": 0.2,
                    "noise_overlap_threshold": 0.01,
                },
                per_scene_rows=[
                    {
                        "scene_name": "scene0000_00",
                        "num_frames": 90,
                        "num_eligible_refs": 7,
                        "selected_anchor_ids": [10, 20, 30, 40, 50],
                        "num_clean_tuples": 5,
                        "num_noise_tuples": 5,
                    }
                ],
            )
            payload = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["num_scenes_scanned"], 1)
        self.assertEqual(payload["num_scenes_kept"], 1)
        self.assertEqual(payload["num_clean_tuples"], 5)
        self.assertEqual(payload["num_noise_tuples"], 5)

    def test_build_attention_noise_benchmark_materializes_clean_and_noise_sequences(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "exact_root"
            scene_root = dataset_root / "scene0000_00"
            color_dir = scene_root / "color_90"
            depth_dir = scene_root / "depth_90"
            intrinsic_dir = scene_root / "depth_intrinsics"
            color_dir.mkdir(parents=True)
            depth_dir.mkdir(parents=True)
            intrinsic_dir.mkdir(parents=True)

            clean_frame_ids = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
            noise_frame_ids = [400, 420, 440, 460]
            poses = []
            for frame_id in clean_frame_ids:
                Image.fromarray(np.full((32, 32, 3), frame_id % 255, dtype=np.uint8)).save(
                    color_dir / f"frame_{frame_id:04d}.jpg"
                )
                Image.fromarray(np.full((32, 32), 1000, dtype=np.uint16)).save(
                    depth_dir / f"frame_{frame_id:04d}.png"
                )
                poses.append(np.eye(4, dtype=np.float64))
            for offset, frame_id in enumerate(noise_frame_ids, start=1):
                Image.fromarray(np.full((32, 32, 3), frame_id % 255, dtype=np.uint8)).save(
                    color_dir / f"frame_{frame_id:04d}.jpg"
                )
                Image.fromarray(np.full((32, 32), 1000, dtype=np.uint16)).save(
                    depth_dir / f"frame_{frame_id:04d}.png"
                )
                pose = np.eye(4, dtype=np.float64)
                pose[0, 3] = 100.0 + offset
                poses.append(pose)

            K = np.array(
                [
                    [50.0, 0.0, 15.5, 0.0],
                    [0.0, 50.0, 15.5, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            np.savetxt(intrinsic_dir / "depth_intrinsic.txt", K, fmt="%.6f")
            (scene_root / "pose_90.txt").write_text(
                "\n".join(" ".join(str(float(x)) for x in pose.reshape(-1)) for pose in poses) + "\n",
                encoding="utf-8",
            )

            output_root = Path(tmpdir) / "diag"
            summary_path = module.build_attention_noise_benchmark(
                dataset_root=dataset_root,
                output_root=output_root,
                sample_stride=8,
                depth_rel_tol=0.01,
                clean_overlap_threshold=0.20,
                noise_overlap_threshold=0.01,
                clean_temporal_dedup=20,
                anchor_quantiles=[0.1, 0.9],
                max_anchors_per_scene=2,
            )
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            sequence_names = sorted(path.name for path in output_root.iterdir() if path.is_dir())

        self.assertEqual(payload["num_clean_tuples"], 2)
        self.assertEqual(payload["num_noise_tuples"], 2)
        self.assertEqual(payload["scenes"][0]["selected_anchor_ids"], [20, 180])
        self.assertEqual(
            sequence_names,
            [
                "scene0000_00__anchor0020__clean",
                "scene0000_00__anchor0020__noise",
                "scene0000_00__anchor0180__clean",
                "scene0000_00__anchor0180__noise",
            ],
        )


if __name__ == "__main__":
    unittest.main()
