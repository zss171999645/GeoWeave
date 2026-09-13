from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
PI3_ROOT = REPO_ROOT / "aidi" / "third_party" / "pi3_training"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PI3_ROOT))

from datasets import __HIGH_QUALITY_DATASETS__, __MIDDLE_QUALITY_DATASETS__  # noqa: E402
from datasets.meshx_evc_dataset import MeshXBusinessMultiviewDataset, MeshXEvcDataset  # noqa: E402
from utils.vggt_validation import VggtStylePi3MetricAccumulator  # noqa: E402


def write_camera_files(
    camera_root: Path,
    names: list[str],
    width: int = 16,
    height: int = 16,
    extri_name: str = "extri.yml",
    translation_offset: float = 0.0,
) -> None:
    camera_root.mkdir(parents=True, exist_ok=True)
    intri_lines = ["%YAML:1.0", "---", "names:"]
    extri_lines = ["%YAML:1.0", "---", "names:"]
    intri_lines.extend([f'  - "{name}"' for name in names])
    extri_lines.extend([f'  - "{name}"' for name in names])

    for idx, name in enumerate(names):
        intri_lines.extend(
            [
                f"K_{name}: !!opencv-matrix",
                "  rows: 3",
                "  cols: 3",
                "  dt: d",
                f"  data: [10., 0., {width / 2.0:.1f}, 0., 10., {height / 2.0:.1f}, 0., 0., 1.]",
                f"H_{name}: {float(height):.1f}",
                f"W_{name}: {float(width):.1f}",
                f"D_{name}: !!opencv-matrix",
                "  rows: 5",
                "  cols: 1",
                "  dt: d",
                "  data: [0., 0., 0., 0., 0.]",
            ]
        )
        extri_lines.extend(
            [
                f"R_{name}: !!opencv-matrix",
                "  rows: 3",
                "  cols: 1",
                "  dt: d",
                "  data: [0., 0., 0.]",
                f"Rot_{name}: !!opencv-matrix",
                "  rows: 3",
                "  cols: 3",
                "  dt: d",
                "  data: [1., 0., 0., 0., 1., 0., 0., 0., 1.]",
                f"T_{name}: !!opencv-matrix",
                "  rows: 3",
                "  cols: 1",
                "  dt: d",
                f"  data: [0., 0., {float(idx) + translation_offset:.1f}]",
                f"t_{name}: {float(idx) + translation_offset:.1f}",
                f"n_{name}: 0.1",
                f"f_{name}: 1000.0",
            ]
        )

    (camera_root / "intri.yml").write_text("\n".join(intri_lines) + "\n", encoding="utf-8")
    (camera_root / extri_name).write_text("\n".join(extri_lines) + "\n", encoding="utf-8")


def write_image(path: Path, value: int = 127, size: tuple[int, int] = (16, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def write_depth(path: Path, depth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, depth.astype(np.float32))


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


class MeshXEvcDatasetTests(unittest.TestCase):
    def test_use_masks_zeroes_depth_outside_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seq = Path(tmpdir) / "scene"
            write_camera_files(seq, ["000000"])
            write_image(seq / "images" / "000000" / "000000.jpg")
            write_depth(seq / "depths" / "000000" / "000000.npy", np.ones((16, 16), dtype=np.float32) * 5.0)
            mask = np.zeros((16, 16), dtype=bool)
            mask[:, :8] = True
            write_mask(seq / "masks" / "000000" / "000000.png", mask)

            dataset = MeshXEvcDataset(
                data_root=str(tmpdir),
                dataset_label="CO3Dv2",
                resolution=[[16, 16]],
                frame_num=1,
                use_masks=True,
                masks_dir="masks",
            )
            view = dataset._get_views(0, [16, 16], np.random.default_rng(0))[0]

            self.assertTrue(np.all(view["depthmap"][:, :8] > 0))
            self.assertTrue(np.all(view["depthmap"][:, 8:] == 0))

    def test_camera_subdirectory_layout_is_scanned_as_camera_sequence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene = Path(tmpdir) / "scene"
            write_camera_files(scene / "cameras" / "00", ["000000", "000001"])
            for name in ("000000", "000001"):
                write_image(scene / "images" / "00" / f"{name}.jpg")
                write_depth(scene / "depths" / "00" / f"{name}.npy", np.ones((16, 16), dtype=np.float32))

            dataset = MeshXEvcDataset(
                data_root=str(tmpdir),
                dataset_label="VirtualKitti",
                resolution=[[16, 16]],
                frame_num=2,
            )

            self.assertEqual(len(dataset), 1)
            self.assertEqual(list(dataset.frame_ids[dataset.sequences[0]]), ["000000", "000001"])
            views = dataset._get_views(0, [16, 16], np.random.default_rng(0))
            self.assertEqual(len(views), 2)
            self.assertTrue(all(view["depthmap"].shape == (16, 16) for view in views))

    def test_camera_subdirectory_layout_can_filter_camera_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene = Path(tmpdir) / "scene"
            for camera_id in ("00", "01", "06"):
                write_camera_files(scene / "cameras" / camera_id, ["000000"])
                write_image(scene / "images" / camera_id / "000000.jpg")
                write_depth(scene / "depths" / camera_id / "000000.npy", np.ones((16, 16), dtype=np.float32))

            dataset = MeshXEvcDataset(
                data_root=str(tmpdir),
                dataset_label="BusinessParkingMechanicalFisheye",
                resolution=[[16, 16]],
                frame_num=1,
                camera_ids=["00", "06"],
            )

            self.assertEqual(len(dataset), 2)
            camera_dirs = {Path(seq).name for seq in dataset.sequences}
            self.assertEqual(camera_dirs, {"00", "06"})

    def test_business_multiview_dataset_samples_across_cameras_without_target_role(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene = Path(tmpdir) / "scene"
            frame_ids = [f"{idx:06d}" for idx in range(4)]
            for camera_id in ("00", "01", "02"):
                write_camera_files(scene / "cameras" / camera_id, frame_ids)
                for frame_id in frame_ids:
                    write_image(scene / "images" / camera_id / f"{frame_id}.jpg")
                    write_depth(
                        scene / "depths" / camera_id / f"{frame_id}.npy",
                        np.ones((16, 16), dtype=np.float32) * (1 + int(camera_id)),
                    )

            dataset = MeshXBusinessMultiviewDataset(
                data_root=str(tmpdir),
                dataset_label="BusinessDriving",
                resolution=[[16, 16]],
                frame_num=6,
                camera_ids=["00", "01", "02"],
                min_cameras=2,
                max_distance=2,
                shuffle=False,
            )

            self.assertEqual(len(dataset), 1)
            dataset.convert_attributes()
            views = dataset._get_views(0, [16, 16], np.random.default_rng(0))
            camera_ids = {view["camera_id"] for view in views}
            self.assertEqual(len(views), 6)
            self.assertGreaterEqual(len(camera_ids), 2)
            self.assertTrue(all("/" in view["instance"] for view in views))
            self.assertTrue(callable(dataset._read_depth))

    def test_business_multiview_dataset_supports_v2_data_roots_and_pandar_extri(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene = Path(tmpdir) / "scene"
            frame_ids = ["00000", "00001"]
            for camera_id in ("00", "01"):
                camera_root = scene / "cameras" / camera_id
                write_camera_files(camera_root, frame_ids)
                write_camera_files(camera_root, frame_ids, extri_name="extri_pandar.yml", translation_offset=10.0)
                for frame_id in frame_ids:
                    write_image(scene / "images" / camera_id / f"{frame_id}.jpg")
                    write_depth(
                        scene / "depths" / camera_id / f"{frame_id}.npy",
                        np.ones((16, 16), dtype=np.float32),
                    )

            (Path(tmpdir) / "val_data_roots.txt").write_text("scene\n", encoding="utf-8")

            dataset = MeshXBusinessMultiviewDataset(
                data_root=str(tmpdir),
                data_roots_file="val_data_roots.txt",
                dataset_label="BusinessDriving",
                resolution=[[16, 16]],
                frame_num=2,
                camera_ids=["00", "01"],
                min_cameras=2,
                extri_file="extri_pandar.yml",
                shuffle=False,
            )

            self.assertEqual(len(dataset), 1)
            views = dataset._get_views(0, [16, 16], np.random.default_rng(0))
            self.assertEqual(len(views), 2)
            self.assertTrue(all(view["camera_pose"][2, 3] <= -10.0 for view in views))

    def test_business_multiview_dataset_can_emit_3ddr_pose_prior(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene = Path(tmpdir) / "scene"
            frame_ids = ["00000", "00001"]
            for camera_id in ("00", "01"):
                camera_root = scene / "cameras" / camera_id
                write_camera_files(camera_root, frame_ids, extri_name="extri_pandar.yml", translation_offset=10.0)
                write_camera_files(camera_root, frame_ids, extri_name="extri_3ddr.yml", translation_offset=20.0)
                for frame_id in frame_ids:
                    write_image(scene / "images" / camera_id / f"{frame_id}.jpg")
                    write_depth(
                        scene / "depths" / camera_id / f"{frame_id}.npy",
                        np.ones((16, 16), dtype=np.float32),
                    )

            dataset = MeshXBusinessMultiviewDataset(
                data_root=str(tmpdir),
                dataset_label="BusinessDriving",
                resolution=[[16, 16]],
                frame_num=2,
                camera_ids=["00", "01"],
                min_cameras=2,
                extri_file="extri_pandar.yml",
                pose_prior_extri_file="extri_3ddr.yml",
                pose_prior_required=True,
                shuffle=False,
            )

            views = dataset._get_views(0, [16, 16], np.random.default_rng(0))
            self.assertEqual(len(views), 2)
            self.assertTrue(all("pose_prior" in view for view in views))
            self.assertTrue(all(view["camera_pose"][2, 3] <= -10.0 for view in views))
            self.assertTrue(all(view["pose_prior"][2, 3] <= -20.0 for view in views))

    def test_business_multiview_vggt_val_protocol_expands_target_window_by_camera(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene = Path(tmpdir) / "scene"
            frame_ids = [f"{idx:05d}" for idx in range(5)]
            for camera_id in ("00", "01", "02"):
                camera_root = scene / "cameras" / camera_id
                write_camera_files(camera_root, frame_ids, extri_name="extri_pandar.yml")
                for frame_id in frame_ids:
                    write_image(scene / "images" / camera_id / f"{frame_id}.jpg")
                    write_depth(
                        scene / "depths" / camera_id / f"{frame_id}.npy",
                        np.ones((16, 16), dtype=np.float32),
                    )

            (Path(tmpdir) / "val_data_roots.txt").write_text("scene\n", encoding="utf-8")

            dataset = MeshXBusinessMultiviewDataset(
                data_root=str(tmpdir),
                data_roots_file="val_data_roots.txt",
                dataset_label="BusinessDriving",
                resolution=[[16, 16]],
                frame_num=9,
                camera_ids=["00", "01", "02"],
                min_cameras=2,
                extri_file="extri_pandar.yml",
                vggt_val_protocol=True,
                vggt_val_extra_src_pool=2,
                vggt_val_frame_sample=[0, None, 2],
                shuffle=False,
            )

            self.assertEqual(len(dataset), 3)
            views = dataset._get_views(0, [16, 16], np.random.default_rng(0))
            self.assertEqual(
                [(view["camera_id"], view["frame_id"]) for view in views],
                [
                    ("00", "00000"),
                    ("00", "00001"),
                    ("00", "00002"),
                    ("01", "00000"),
                    ("01", "00001"),
                    ("01", "00002"),
                    ("02", "00000"),
                    ("02", "00001"),
                    ("02", "00002"),
                ],
            )

    def test_business_multiview_vggt_val_protocol_can_follow_vggt_metrics_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene = Path(tmpdir) / "scene"
            frame_ids = [f"{idx:05d}" for idx in range(5)]
            for camera_id in ("00", "01", "02"):
                camera_root = scene / "cameras" / camera_id
                write_camera_files(camera_root, frame_ids, extri_name="extri_pandar.yml")
                for frame_id in frame_ids:
                    write_image(scene / "images" / camera_id / f"{frame_id}.jpg")
                    write_depth(
                        scene / "depths" / camera_id / f"{frame_id}.npy",
                        np.ones((16, 16), dtype=np.float32),
                    )

            (Path(tmpdir) / "val_data_roots.txt").write_text("scene\n", encoding="utf-8")
            metrics_path = Path(tmpdir) / "vggt_metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "metrics": [
                            {"path": str(scene), "camera": 0, "frame": 2},
                            {"path": str(scene), "camera": 0, "frame": 4},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            dataset = MeshXBusinessMultiviewDataset(
                data_root=str(tmpdir),
                data_roots_file="val_data_roots.txt",
                dataset_label="BusinessDriving",
                resolution=[[16, 16]],
                frame_num=9,
                camera_ids=["00", "01", "02"],
                min_cameras=2,
                extri_file="extri_pandar.yml",
                vggt_val_protocol=True,
                vggt_val_extra_src_pool=2,
                vggt_val_target_metrics_file=str(metrics_path),
                shuffle=False,
            )

            self.assertEqual(len(dataset), 2)
            views = dataset._get_views(0, [16, 16], np.random.default_rng(0))
            self.assertEqual(
                [(view["camera_id"], view["frame_id"]) for view in views],
                [
                    ("00", "00002"),
                    ("00", "00001"),
                    ("00", "00003"),
                    ("01", "00002"),
                    ("01", "00001"),
                    ("01", "00003"),
                    ("02", "00002"),
                    ("02", "00001"),
                    ("02", "00003"),
                ],
            )
            self.assertTrue(all(view["sample_id"].endswith("|00|00002") for view in views))

    def test_vggt_style_metric_accumulator_deduplicates_sample_ids(self):
        acc = VggtStylePi3MetricAccumulator({"enabled": True})

        acc.update({"_sample_id": "scene_a|00|00000", "cam:pose_auc_30": 0.1})
        acc.update({"_sample_id": "scene_a|00|00000", "cam:pose_auc_30": 0.9})
        acc.update({"_sample_id": "scene_b|00|00000", "cam:pose_auc_30": 0.5})

        summary = acc.summarize()
        self.assertAlmostEqual(summary["cam:pose_auc_30_mean"], 0.7)
        self.assertAlmostEqual(summary["cam:pose_auc_30_std"], 0.2)

    def test_business_multiview_dataset_does_not_retain_scene_metadata_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for scene_name in ("scene_a", "scene_b"):
                scene = Path(tmpdir) / scene_name
                frame_ids = ["000000", "000001"]
                for camera_id in ("00", "01"):
                    write_camera_files(scene / "cameras" / camera_id, frame_ids)
                    for frame_id in frame_ids:
                        write_image(scene / "images" / camera_id / f"{frame_id}.jpg")
                        write_depth(
                            scene / "depths" / camera_id / f"{frame_id}.npy",
                            np.ones((16, 16), dtype=np.float32),
                        )

            dataset = MeshXBusinessMultiviewDataset(
                data_root=str(tmpdir),
                dataset_label="BusinessDriving",
                resolution=[[16, 16]],
                frame_num=2,
                camera_ids=["00", "01"],
                min_cameras=2,
                shuffle=False,
            )

            dataset.convert_attributes()
            for idx in range(len(dataset)):
                dataset._get_views(idx, [16, 16], np.random.default_rng(idx))

            self.assertEqual(dataset.cameras, {})
            scene_specs_values = dataset.scene_specs.values
            if callable(scene_specs_values):
                scene_specs_values = scene_specs_values()
            for scene_spec in scene_specs_values:
                for camera_spec in scene_spec["camera_specs"].values():
                    self.assertEqual(camera_spec["frames"], [])

    def test_business_multiview_lazy_frames_uses_camera_names_without_directory_listing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene = Path(tmpdir) / "scene"
            frame_ids = ["000000", "000001", "000002"]
            for camera_id in ("00", "01"):
                write_camera_files(scene / "cameras" / camera_id, frame_ids)
                for frame_id in frame_ids:
                    write_image(scene / "images" / camera_id / f"{frame_id}.jpg")
                    write_depth(
                        scene / "depths" / camera_id / f"{frame_id}.npy",
                        np.ones((16, 16), dtype=np.float32),
                    )

            dataset = MeshXBusinessMultiviewDataset(
                data_root=str(tmpdir),
                dataset_label="BusinessDriving",
                resolution=[[16, 16]],
                frame_num=2,
                camera_ids=["00", "01"],
                min_cameras=2,
                shuffle=False,
            )

            import os as _os

            original_listdir = _os.listdir

            def guarded_listdir(path):
                text = str(path)
                if "/images/" in text or "/depths/" in text:
                    raise AssertionError(f"unexpected frame directory listing: {text}")
                return original_listdir(path)

            _os.listdir = guarded_listdir
            try:
                views = dataset._get_views(0, [16, 16], np.random.default_rng(0))
            finally:
                _os.listdir = original_listdir

            self.assertEqual(len(views), 2)
            self.assertTrue(all(view["depthmap"].shape == (16, 16) for view in views))

    def test_data_roots_file_fast_path_supports_direct_and_camera_layouts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            direct = Path(tmpdir) / "direct_scene"
            write_camera_files(direct, ["000000"])
            write_image(direct / "images" / "000000" / "000000.jpg")
            write_depth(direct / "depths" / "000000" / "000000.npy", np.ones((16, 16), dtype=np.float32))

            camera_scene = Path(tmpdir) / "camera_scene"
            write_camera_files(camera_scene / "cameras" / "00", ["000001"])
            write_image(camera_scene / "images" / "00" / "000001.jpg")
            write_depth(camera_scene / "depths" / "00" / "000001.npy", np.ones((16, 16), dtype=np.float32))

            (Path(tmpdir) / "data_roots.txt").write_text(
                "direct_scene\ncamera_scene\n",
                encoding="utf-8",
            )

            dataset = MeshXEvcDataset(
                data_root=str(tmpdir),
                dataset_label="MeshXEVC",
                resolution=[[16, 16]],
                frame_num=1,
            )

            self.assertEqual(len(dataset), 2)
            self.assertTrue(all(dataset.frame_ids[seq] == [] for seq in dataset.sequences))
            labels = {Path(dataset.sequence_labels[seq]).name for seq in dataset.sequences}
            self.assertEqual(labels, {"direct_scene", "00"})
            views = [dataset._get_views(idx, [16, 16], np.random.default_rng(0))[0] for idx in range(len(dataset))]
            self.assertTrue(all(view["depthmap"].shape == (16, 16) for view in views))

    def test_large_data_roots_file_infers_layout_without_eager_frame_listing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            direct = Path(tmpdir) / "direct_scene"
            write_camera_files(direct, ["000000"])
            write_image(direct / "images" / "000000" / "000000.jpg")
            write_depth(direct / "depths" / "000000" / "000000.npy", np.ones((16, 16), dtype=np.float32))
            rel_roots = ["direct_scene"] + [f"missing_{idx:04d}" for idx in range(1000)]
            (Path(tmpdir) / "data_roots.txt").write_text("\n".join(rel_roots) + "\n", encoding="utf-8")

            dataset = MeshXEvcDataset(
                data_root=str(tmpdir),
                dataset_label="MeshXEVC",
                resolution=[[16, 16]],
                frame_num=1,
            )

            self.assertEqual(len(dataset), 1001)
            self.assertEqual(dataset.frame_ids[dataset.sequences[0]], [])
            view = dataset._get_views(0, [16, 16], np.random.default_rng(0))[0]
            self.assertEqual(view["depthmap"].shape, (16, 16))

    def test_depth_filtering_applies_max_depth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seq = Path(tmpdir) / "scene"
            write_camera_files(seq, ["000000"])
            write_image(seq / "images" / "000000" / "000000.jpg")
            depth = np.ones((16, 16), dtype=np.float32)
            depth[:, 8:] = 100.0
            write_depth(seq / "depths" / "000000" / "000000.npy", depth)

            dataset = MeshXEvcDataset(
                data_root=str(tmpdir),
                dataset_label="GTAV",
                resolution=[[16, 16]],
                frame_num=1,
                max_depth=50.0,
            )
            view = dataset._get_views(0, [16, 16], np.random.default_rng(0))[0]

            self.assertTrue(np.all(view["depthmap"][:, :8] > 0))
            self.assertTrue(np.all(view["depthmap"][:, 8:] == 0))

    def test_depth_filtering_applies_quantiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seq = Path(tmpdir) / "scene"
            write_camera_files(seq, ["000000"])
            write_image(seq / "images" / "000000" / "000000.jpg")
            depth = np.tile(np.arange(1, 17, dtype=np.float32), (16, 1))
            write_depth(seq / "depths" / "000000" / "000000.npy", depth)

            dataset = MeshXEvcDataset(
                data_root=str(tmpdir),
                dataset_label="GTAV",
                resolution=[[16, 16]],
                frame_num=1,
                min_depth_quantile=0.0,
                max_depth_quantile=0.5,
            )
            view = dataset._get_views(0, [16, 16], np.random.default_rng(0))[0]

            self.assertTrue(np.all(view["depthmap"][:, :8] > 0))
            self.assertTrue(np.all(view["depthmap"][:, 8:] == 0))

    def test_meshx_dataset_labels_are_visible_to_pi3_normal_loss(self):
        normal_loss_labels = set(__HIGH_QUALITY_DATASETS__ + __MIDDLE_QUALITY_DATASETS__)
        expected_labels = {
            "Replica",
            "ASE",
            "ADT",
            "MegaDepth",
            "WildRGBD",
            "MapFree",
            "CO3Dv2",
            "Mapillary",
            "DL3DV",
            "BusinessDriving",
            "BusinessParking",
            "BusinessParkingMechanical",
            "BusinessParkingMechanicalFisheye",
        }

        self.assertTrue(expected_labels.issubset(normal_loss_labels))


if __name__ == "__main__":
    unittest.main()
