from __future__ import annotations

import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "aidi" / "scripts" / "baselines" / "overlap_noise_seq_map_utils.py"
SCRIPT_PATH = REPO_ROOT / "aidi" / "scripts" / "baselines" / "build_evc_attention_noise_benchmark.py"


def load_helper_module():
    spec = importlib.util.spec_from_file_location("evc_overlap_noise_seq_map_utils", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_script_module():
    spec = importlib.util.spec_from_file_location("build_evc_attention_noise_benchmark", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EVCAttentionNoiseTests(unittest.TestCase):
    def test_subsample_scene_record_matches_eval_prefix_window_rule(self):
        module = load_helper_module()
        scene_record = {
            "scene_name": "toy-seq",
            "frame_names": [f"{idx:06d}" for idx in range(12)],
            "frame_ids": list(range(12)),
            "color_paths": [Path(f"/tmp/{idx:06d}.jpg") for idx in range(12)],
            "depth_paths": [Path(f"/tmp/{idx:06d}.exr") for idx in range(12)],
            "poses": np.stack([np.eye(4, dtype=np.float32) * (idx + 1) for idx in range(12)], axis=0),
            "intrinsics": [np.eye(3, dtype=np.float32) * (idx + 1) for idx in range(12)],
        }

        subsampled = module.subsample_scene_record(scene_record=scene_record, target_num_frames=4, source_prestride=3)

        self.assertEqual(subsampled["frame_ids"], [0, 3, 6, 9])
        self.assertEqual(subsampled["frame_names"], ["000000", "000003", "000006", "000009"])
        self.assertEqual(len(subsampled["color_paths"]), 4)
        self.assertEqual(len(subsampled["depth_paths"]), 4)
        self.assertEqual(subsampled["poses"].shape, (4, 4, 4))
        self.assertEqual(len(subsampled["intrinsics"]), 4)

    def test_build_evc_camera_scene_record_aligns_camera_image_and_depth_names(self):
        module = load_helper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_root = Path(tmpdir) / "rgbd_dataset_freiburg1_room"
            image_dir = seq_root / "images" / "00"
            depth_dir = seq_root / "depths" / "00"
            camera_dir = seq_root / "cameras" / "00"
            image_dir.mkdir(parents=True)
            depth_dir.mkdir(parents=True)
            camera_dir.mkdir(parents=True)

            for frame_name in ("000000", "000005", "000012"):
                Image.fromarray(np.full((4, 5, 3), 7, dtype=np.uint8)).save(image_dir / f"{frame_name}.jpg")
                (depth_dir / f"{frame_name}.exr").write_bytes(b"exr")
            (camera_dir / "intri.yml").write_text("%YAML:1.0\n", encoding="utf-8")
            (camera_dir / "extri.yml").write_text("%YAML:1.0\n", encoding="utf-8")

            fake_cameras = {
                "000000": types.SimpleNamespace(K=np.eye(3, dtype=np.float32), RT=np.hstack([np.eye(3), np.zeros((3, 1))])),
                "000005": types.SimpleNamespace(K=np.eye(3, dtype=np.float32) * 2.0, RT=np.hstack([np.eye(3), np.ones((3, 1))])),
            }

            with mock.patch.object(module, "load_read_camera", return_value=lambda *args, **kwargs: fake_cameras):
                scene = module.build_evc_camera_scene_record(seq_root=seq_root, dataset_name="tum")

        self.assertEqual(scene["scene_name"], "rgbd_dataset_freiburg1_room")
        self.assertEqual(scene["frame_names"], ["000000", "000005"])
        self.assertEqual(scene["frame_ids"], [0, 5])
        self.assertEqual(len(scene["color_paths"]), 2)
        self.assertEqual(len(scene["depth_paths"]), 2)
        self.assertEqual(scene["poses"].shape, (2, 4, 4))
        self.assertEqual(len(scene["intrinsics"]), 2)
        self.assertEqual(float(scene["intrinsics"][1][0, 0]), 2.0)

    def test_materialize_tuple_sequence_preserves_depth_extension(self):
        module = load_helper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "scene"
            source_root.mkdir(parents=True)
            color_a = source_root / "000000.jpg"
            color_b = source_root / "000005.jpg"
            depth_a = source_root / "000000.exr"
            depth_b = source_root / "000005.exr"
            Image.fromarray(np.full((4, 5, 3), 1, dtype=np.uint8)).save(color_a)
            Image.fromarray(np.full((4, 5, 3), 2, dtype=np.uint8)).save(color_b)
            depth_a.write_bytes(b"exr-a")
            depth_b.write_bytes(b"exr-b")

            scene_record = {
                "scene_name": "tum-room",
                "frame_ids": [0, 5],
                "color_paths": [color_a, color_b],
                "depth_paths": [depth_a, depth_b],
                "poses": np.stack([np.eye(4, dtype=np.float64), np.eye(4, dtype=np.float64)], axis=0),
            }
            tuple_root = module.materialize_tuple_sequence(
                scene_record=scene_record,
                tuple_name="tum-room__anchor0000__clean",
                ordered_ids=[0, 5],
                output_root=Path(tmpdir) / "diag",
            )
            self.assertTrue((tuple_root / "color_90" / "frame_0000.jpg").is_file())
            self.assertTrue((tuple_root / "depth_90" / "frame_0000.exr").is_file())
            self.assertTrue((tuple_root / "depth_90" / "frame_0001.exr").is_file())

    def test_eth3d_pi3_scene_record_materializes_raw_depth_as_npy(self):
        module = load_helper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_root = Path(tmpdir) / "kicker"
            image_dir = seq_root / "images" / "custom_undistorted"
            depth_dir = seq_root / "ground_truth_depth" / "custom_undistorted"
            camera_dir = seq_root / "custom_undistorted_cam"
            image_dir.mkdir(parents=True)
            depth_dir.mkdir(parents=True)
            camera_dir.mkdir(parents=True)

            image = np.full((4, 5, 3), 11, dtype=np.uint8)
            depth = np.arange(20, dtype=np.float32).reshape(4, 5) + 1.0
            for stem, tx in (("DSC_0001", 1.0), ("DSC_0002", 2.0)):
                Image.fromarray(image).save(image_dir / f"{stem}.JPG")
                depth.astype(np.float32).tofile(depth_dir / f"{stem}.JPG")
                w2c = np.eye(4, dtype=np.float32)
                w2c[0, 3] = tx
                np.savez(camera_dir / f"{stem}.npz", intrinsics=np.eye(3, dtype=np.float32), extrinsics=w2c)

            scene = module.build_eth3d_pi3_scene_record(seq_root=seq_root)
            self.assertEqual(scene["frame_ids"], [1, 2])
            self.assertEqual(scene["poses"].shape, (2, 4, 4))
            self.assertAlmostEqual(float(scene["poses"][0, 0, 3]), -1.0)
            self.assertEqual(len(scene["depth_shapes"]), 2)

            tuple_root = module.materialize_tuple_sequence(
                scene_record=scene,
                tuple_name="kicker__anchor0001__pool02",
                ordered_ids=[1, 2],
                output_root=Path(tmpdir) / "out",
            )

            saved = np.load(tuple_root / "depth_90" / "frame_0000.npy")
            np.testing.assert_allclose(saved, depth)

    def test_evc_builder_discovers_tum_and_vkitti_scene_roots(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tum_a = root / "tum" / "rgbd_dataset_freiburg1_room" / "images" / "00"
            tum_b = root / "tum" / "rgbd_dataset_freiburg1_xyz" / "images" / "00"
            (root / "tum" / "rgbd_dataset_freiburg1_room" / "depths" / "00").mkdir(parents=True)
            (root / "tum" / "rgbd_dataset_freiburg1_xyz" / "depths" / "00").mkdir(parents=True)
            tum_a.mkdir(parents=True)
            tum_b.mkdir(parents=True)

            vkitti_seq = root / "vkitti2" / "Scene01" / "clone" / "images" / "00"
            (root / "vkitti2" / "Scene01" / "clone" / "depths" / "00").mkdir(parents=True)
            vkitti_seq.mkdir(parents=True)

            tum_roots = module.discover_tum_scene_roots(root / "tum")
            vkitti_roots = module.discover_vkitti_scene_roots(root / "vkitti2")

        self.assertEqual([path.name for path in tum_roots], ["rgbd_dataset_freiburg1_room", "rgbd_dataset_freiburg1_xyz"])
        self.assertEqual([path.as_posix().split("/")[-2:] for path in vkitti_roots], [["Scene01", "clone"]])
        self.assertEqual(module.vkitti_scene_slug(vkitti_roots[0]), "Scene01-clone-cam00")

    def test_evc_builder_accepts_direct_sequence_root(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tum_root = Path(tmpdir) / "rgbd_dataset_freiburg1_room"
            (tum_root / "images" / "00").mkdir(parents=True)
            (tum_root / "depths" / "00").mkdir(parents=True)

            vkitti_root = Path(tmpdir) / "Scene01" / "clone"
            (vkitti_root / "images" / "00").mkdir(parents=True)
            (vkitti_root / "depths" / "00").mkdir(parents=True)

            tum_roots = module.discover_tum_scene_roots(tum_root)
            vkitti_roots = module.discover_vkitti_scene_roots(vkitti_root)

        self.assertEqual(tum_roots, [tum_root])
        self.assertEqual(vkitti_roots, [vkitti_root])

    def test_dataset_protocol_defaults_use_smaller_vkitti_dedup(self):
        module = load_script_module()

        tum_defaults = module.resolve_protocol_defaults(dataset="tum")
        vkitti_defaults = module.resolve_protocol_defaults(dataset="vkitti")

        self.assertEqual(tum_defaults["clean_temporal_dedup"], 20)
        self.assertEqual(vkitti_defaults["clean_temporal_dedup"], 5)
        self.assertEqual(vkitti_defaults["noise_overlap_threshold"], 0.01)

    def test_vkitti_discovery_can_filter_to_clone_variant(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "vkitti2"
            for variant in ("clone", "fog", "15-deg-left"):
                (root / "Scene01" / variant / "images" / "00").mkdir(parents=True)
                (root / "Scene01" / variant / "depths" / "00").mkdir(parents=True)

            clone_only = module.discover_vkitti_scene_roots(root, variant_filter="clone")

        self.assertEqual([path.as_posix().split("/")[-2:] for path in clone_only], [["Scene01", "clone"]])

    def test_evc_builder_materializes_clean_and_noise_tuples(self):
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
                "frame_ids": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 200, 220, 240, 260],
                "color_paths": [],
                "depth_paths": [],
                "poses": np.zeros((16, 4, 4), dtype=np.float32),
                "intrinsics": [],
            }

            with mock.patch.object(module, "build_evc_camera_scene_record", return_value=fake_scene_record):
                with mock.patch.object(
                    module,
                    "build_scene_overlap_table",
                    return_value={
                        0: {
                            10: 0.95,
                            20: 0.92,
                            30: 0.88,
                            40: 0.84,
                            50: 0.80,
                            60: 0.76,
                            70: 0.72,
                            80: 0.68,
                            90: 0.64,
                            200: 0.01,
                            220: 0.009,
                            240: 0.008,
                            260: 0.007,
                        }
                    },
                ):
                    with mock.patch.object(module, "materialize_tuple_sequence") as materialize_mock:
                        summary_path = module.build_attention_noise_benchmark(
                            dataset="tum",
                            dataset_root=dataset_root,
                            output_root=output_root,
                            sample_stride=16,
                            depth_rel_tol=0.01,
                            clean_overlap_threshold=0.20,
                            noise_overlap_threshold=0.01,
                            clean_temporal_dedup=1,
                            anchor_quantiles=[0.1],
                            max_anchors_per_scene=1,
                            image_subdir="images/00",
                            depth_subdir="depths/00",
                            camera_subdir="cameras/00",
                            limit_scenes=1,
                        )

            self.assertEqual(materialize_mock.call_count, 2)
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["num_clean_tuples"], 1)
            self.assertEqual(payload["num_noise_tuples"], 1)
            self.assertEqual(payload["scenes"][0]["scene_name"], "rgbd_dataset_freiburg1_room")


if __name__ == "__main__":
    unittest.main()
