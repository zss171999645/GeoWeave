from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aidi"
    / "scripts"
    / "baselines"
    / "build_hiroom_distractor_benchmark.py"
)


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"Missing builder script: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("build_hiroom_distractor_benchmark", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_scene(root: Path, scene: str, num_frames: int) -> None:
    scene_root = root / scene
    image_root = scene_root / "image"
    pose_root = scene_root / "pose"
    image_root.mkdir(parents=True)
    pose_root.mkdir(parents=True)
    np.save(scene_root / "cam_K.npy", np.eye(3, dtype=np.float32))
    for index in range(num_frames):
        Image.new("RGB", (16, 12), color=(index, 10, 20)).save(image_root / f"{index:06d}.jpg")
        w2c = np.eye(4, dtype=np.float32)
        w2c[0, 3] = float(index)
        np.save(pose_root / f"{index:06d}.npy", w2c)


class HiRoomDistractorBenchmarkTests(unittest.TestCase):
    def test_list_hiroom_frame_names_filters_hidden_and_requires_pose(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "data"
            scene_root = root / "building" / "room" / "cam_sampled_01"
            (scene_root / "image").mkdir(parents=True)
            (scene_root / "pose").mkdir()
            for stem in ("000010", "000002", "000001"):
                (scene_root / "image" / f"{stem}.jpg").write_bytes(b"jpg")
            (scene_root / "image" / ".000003.jpg.tmp").write_bytes(b"tmp")
            np.save(scene_root / "pose" / "000010.npy", np.eye(4, dtype=np.float32))
            np.save(scene_root / "pose" / "000001.npy", np.eye(4, dtype=np.float32))

            frames = module.list_hiroom_frame_names(scene_root)

        self.assertEqual(frames, ["000001", "000010"])

    def test_build_scene_tuples_keeps_first_six_and_replaces_tail(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "data"
            write_scene(data_root, "day/scene_a/cam_sampled_01", 12)
            write_scene(data_root, "day/scene_b/cam_sampled_02", 12)
            records = [
                module.load_hiroom_scene_record(data_root, "day/scene_a/cam_sampled_01"),
                module.load_hiroom_scene_record(data_root, "day/scene_b/cam_sampled_02"),
            ]
            output_root = Path(tmpdir) / "out"
            args = module.BuildArgs(
                total_views=10,
                eval_views=6,
                noise_views=4,
                frame_stride=1,
                max_anchors_per_scene=1,
                distractor_scene_offset=1,
                output_image_ext=".jpg",
                copy_images=True,
            )

            summary = module.build_scene_tuples(
                scene_index=0,
                scene_records=records,
                output_root=output_root,
                args=args,
                anchor_quantiles=[0.0],
            )

            self.assertEqual(summary["num_clean_tuples"], 1)
            self.assertEqual(summary["num_noise_tuples"], 1)
            clean_meta = module.read_json(next(output_root.glob("*__clean/tuple_meta.json")))
            noise_meta = module.read_json(next(output_root.glob("*__noise/tuple_meta.json")))

        self.assertEqual(noise_meta["ordered_frame_names"][:6], clean_meta["ordered_frame_names"][:6])
        self.assertNotEqual(noise_meta["target_scene_name"], noise_meta["distractor_scene_name"])
        self.assertEqual(noise_meta["eval_frame_indices"], [0, 1, 2, 3, 4, 5])

    def test_materialize_tuple_writes_pi3_official_layout_and_c2w_poses(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "data"
            write_scene(data_root, "day/scene_a/cam_sampled_01", 10)
            record = module.load_hiroom_scene_record(data_root, "day/scene_a/cam_sampled_01")
            tuple_root = Path(tmpdir) / "tuple"

            frame_names = record.frame_names[:10]
            module.materialize_hiroom_tuple(
                tuple_root=tuple_root,
                record=record,
                frame_names=frame_names,
                copy_images=True,
                output_image_ext=".jpg",
                metadata={"tuple_kind": "clean"},
            )

            pose_rows = [
                [float(item) for item in line.split()]
                for line in (tuple_root / "pose_90.txt").read_text(encoding="utf-8").splitlines()
            ]
            output_image_count = len(list((tuple_root / "color_90").glob("frame_*.jpg")))
            tuple_meta_exists = (tuple_root / "tuple_meta.json").is_file()

        self.assertEqual(output_image_count, 10)
        self.assertEqual(len(pose_rows), 10)
        first_pose = np.asarray(pose_rows[1], dtype=np.float32).reshape(4, 4)
        # Source HiRoom pose is w2c with tx=1, so PI3 official pose must be c2w with tx=-1.
        self.assertAlmostEqual(float(first_pose[0, 3]), -1.0)
        self.assertTrue(tuple_meta_exists)


if __name__ == "__main__":
    unittest.main()
