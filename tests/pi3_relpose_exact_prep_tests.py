from __future__ import annotations

import importlib.util
import json
import builtins
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "easyvolcap"
    / "utils"
    / "pi3"
    / "relpose_exact_prep.py"
)

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "aidi"
    / "scripts"
    / "baselines"
    / "prepare_pi3_relpose_exact_dataset.py"
)


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"Missing exact preparation helper: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("relpose_exact_prep", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_script_module():
    if not SCRIPT_PATH.is_file():
        raise AssertionError(f"Missing exact preparation script: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("prepare_pi3_relpose_exact_dataset", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_module_without_numpy():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"Missing exact preparation helper: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("relpose_exact_prep_no_numpy", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numpy":
            raise ModuleNotFoundError("numpy blocked by test")
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=guarded_import):
        spec.loader.exec_module(module)
    return module


class Pi3RelposeExactPrepTests(unittest.TestCase):
    def test_prepare_script_import_bootstraps_repo_root(self):
        module_root = str(SCRIPT_PATH.parents[3])
        original_path = list(sys.path)
        try:
            sys.path[:] = [entry for entry in sys.path if entry != module_root]
            module = load_script_module()
            self.assertIn(module_root, sys.path)
            self.assertEqual(module.Path(__file__).resolve().parents[1], SCRIPT_PATH.parents[3])
        finally:
            sys.path[:] = original_path

    def test_prepare_tum_requires_rgb_txt(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            source_seq = Path(tmpdir) / "rgbd_dataset_freiburg1_360"
            output_seq = Path(tmpdir) / "prepared"
            source_seq.mkdir(parents=True)
            (source_seq / "groundtruth.txt").write_text("0.0 0 0 0 0 0 0 1\n", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "rgb.txt"):
                module.prepare_tum_sequence(source_seq, output_seq)

    def test_prepare_tum_matches_monst3r_association_and_stride3_prefix_sampling(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            source_seq = Path(tmpdir) / "rgbd_dataset_freiburg1_360"
            output_seq = Path(tmpdir) / "prepared"
            rgb_dir = source_seq / "rgb"
            rgb_dir.mkdir(parents=True)

            rgb_lines = []
            gt_lines = []
            for idx in range(6):
                rgb_stamp = 1.0 + idx * 0.03
                gt_stamp = rgb_stamp + 0.005
                frame_name = f"{idx:06d}.png"
                (rgb_dir / frame_name).write_bytes(b"png")
                rgb_lines.append(f"{rgb_stamp:.6f} rgb/{frame_name}")
                gt_lines.append(f"{gt_stamp:.6f} {idx} 0 0 0 0 0 1")

            (source_seq / "rgb.txt").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")
            (source_seq / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")

            summary = module.prepare_tum_sequence(source_seq, output_seq)

            rgb_90_dir = output_seq / "rgb_90"
            self.assertEqual(summary["num_source_frames"], 6)
            self.assertEqual(summary["num_selected_frames"], 2)
            self.assertEqual(sorted(path.name for path in rgb_90_dir.iterdir()), ["000000.png", "000003.png"])

            gt_90_lines = (output_seq / "groundtruth_90.txt").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(gt_90_lines), 2)
            self.assertTrue(gt_90_lines[0].startswith("1.005 "))
            self.assertTrue(gt_90_lines[1].startswith("1.095 "))

    def test_prepare_scannet_requires_pose_dir(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            source_seq = Path(tmpdir) / "scene0000_00"
            output_seq = Path(tmpdir) / "prepared"
            (source_seq / "color").mkdir(parents=True)
            (source_seq / "depth").mkdir(parents=True)

            with self.assertRaisesRegex(FileNotFoundError, "pose"):
                module.prepare_scannet_sequence(source_seq, output_seq)

    def test_prepare_scannet_matches_monst3r_prefix_sampling_and_renaming(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            source_seq = Path(tmpdir) / "scene0000_00"
            output_seq = Path(tmpdir) / "prepared"
            color_dir = source_seq / "color"
            depth_dir = source_seq / "depth"
            pose_dir = source_seq / "pose"
            color_dir.mkdir(parents=True)
            depth_dir.mkdir(parents=True)
            pose_dir.mkdir(parents=True)

            for idx in range(6):
                (color_dir / f"{idx}.jpg").write_bytes(b"jpg")
                (depth_dir / f"{idx}.png").write_bytes(b"png")
                pose = np.eye(4, dtype=np.float64)
                pose[0, 3] = idx
                np.savetxt(pose_dir / f"{idx}.txt", pose)

            summary = module.prepare_scannet_sequence(source_seq, output_seq)

            color_90_dir = output_seq / "color_90"
            depth_90_dir = output_seq / "depth_90"
            pose_90_lines = (output_seq / "pose_90.txt").read_text(encoding="utf-8").strip().splitlines()

            self.assertEqual(summary["num_source_frames"], 6)
            self.assertEqual(summary["num_selected_frames"], 2)
            self.assertEqual(sorted(path.name for path in color_90_dir.iterdir()), ["frame_0000.jpg", "frame_0001.jpg"])
            self.assertEqual(sorted(path.name for path in depth_90_dir.iterdir()), ["frame_0000.png", "frame_0001.png"])
            self.assertEqual(len(pose_90_lines), 2)
            self.assertEqual(len(pose_90_lines[0].split()), 16)
            self.assertEqual(float(pose_90_lines[0].split()[3]), 0.0)
            self.assertEqual(float(pose_90_lines[1].split()[3]), 3.0)

    def test_prepare_scannet_does_not_require_numpy_import(self):
        module = load_module_without_numpy()

        with TemporaryDirectory() as tmpdir:
            source_seq = Path(tmpdir) / "scene0000_00"
            output_seq = Path(tmpdir) / "prepared"
            color_dir = source_seq / "color"
            depth_dir = source_seq / "depth"
            pose_dir = source_seq / "pose"
            color_dir.mkdir(parents=True)
            depth_dir.mkdir(parents=True)
            pose_dir.mkdir(parents=True)

            (color_dir / "0.jpg").write_bytes(b"jpg")
            (depth_dir / "0.png").write_bytes(b"png")
            (pose_dir / "0.txt").write_text(
                "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n",
                encoding="utf-8",
            )

            summary = module.prepare_scannet_sequence(source_seq, output_seq)

            self.assertEqual(summary["num_source_frames"], 1)
            self.assertEqual(summary["num_selected_frames"], 1)
            pose_90_lines = (output_seq / "pose_90.txt").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(pose_90_lines, ["1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0"])

    def test_prepare_dataset_writes_summary_json_and_honors_limit(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "tum"
            output_root = Path(tmpdir) / "prepared"
            for seq_name in ("rgbd_dataset_freiburg1_360", "rgbd_dataset_freiburg1_rpy"):
                source_seq = source_root / seq_name
                rgb_dir = source_seq / "rgb"
                rgb_dir.mkdir(parents=True)
                (rgb_dir / "000000.png").write_bytes(b"png")
                (source_seq / "rgb.txt").write_text("1.0 rgb/000000.png\n", encoding="utf-8")
                (source_seq / "groundtruth.txt").write_text("1.005 0 0 0 0 0 0 1\n", encoding="utf-8")

            summary = module.prepare_dataset("tum", source_root, output_root, limit_seqs=1)
            summary_json = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["dataset"], "tum")
            self.assertEqual(summary["num_sequences"], 1)
            self.assertEqual(summary_json["dataset"], "tum")
            self.assertEqual(summary_json["num_sequences"], 1)
            self.assertEqual(len(summary_json["sequences"]), 1)


if __name__ == "__main__":
    unittest.main()
