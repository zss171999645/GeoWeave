from __future__ import annotations

import importlib.util
import json
import builtins
import sys
import io
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


HELPER_PATH = (
    Path(__file__).resolve().parents[1]
    / "easyvolcap"
    / "utils"
    / "pi3"
    / "scannet_sens_export.py"
)

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "aidi"
    / "scripts"
    / "baselines"
    / "prepare_pi3_scannet_sens_exact.py"
)


def load_helper_module():
    if not HELPER_PATH.is_file():
        raise AssertionError(f"Missing ScanNet sens export helper: {HELPER_PATH}")
    spec = importlib.util.spec_from_file_location("scannet_sens_export", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading helper from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_script_module():
    if not SCRIPT_PATH.is_file():
        raise AssertionError(f"Missing ScanNet sens exact script: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("prepare_pi3_scannet_sens_exact", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading script from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_helper_module_without_numpy():
    if not HELPER_PATH.is_file():
        raise AssertionError(f"Missing ScanNet sens export helper: {HELPER_PATH}")
    spec = importlib.util.spec_from_file_location("scannet_sens_export_no_numpy", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading helper from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numpy":
            raise ModuleNotFoundError("numpy blocked by test")
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=guarded_import):
        spec.loader.exec_module(module)
    return module


class FakeSensorData:
    def __init__(self, filename: str):
        self.filename = filename
        self.frames = [object(), object(), object(), object()]

    def export_depth_images(self, output_path: str, image_size=None, frame_skip: int = 1, max_frames: int | None = None):
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "0.png").write_bytes(b"png")
        (output_dir / "3.png").write_bytes(b"png")

    def export_color_images(self, output_path: str, image_size=None, frame_skip: int = 1, max_frames: int | None = None):
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "0.jpg").write_bytes(b"jpg")
        (output_dir / "3.jpg").write_bytes(b"jpg")

    def export_poses(self, output_path: str, frame_skip: int = 1, max_frames: int | None = None):
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "0.txt").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="utf-8")
        (output_dir / "3.txt").write_text("1 0 0 3\n0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="utf-8")

    def export_intrinsics(self, output_path: str):
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "intrinsic_color.txt").write_text("1 0 0 0\n", encoding="utf-8")


class TrackingSensorData(FakeSensorData):
    last_instance = None

    def __init__(self, filename: str):
        super().__init__(filename)
        self.color_max_frames = None
        self.depth_max_frames = None
        self.pose_max_frames = None
        type(self).last_instance = self

    def export_depth_images(self, output_path: str, image_size=None, frame_skip: int = 1, max_frames: int | None = None):
        self.depth_max_frames = max_frames
        super().export_depth_images(output_path, image_size=image_size, frame_skip=frame_skip, max_frames=max_frames)

    def export_color_images(self, output_path: str, image_size=None, frame_skip: int = 1, max_frames: int | None = None):
        self.color_max_frames = max_frames
        super().export_color_images(output_path, image_size=image_size, frame_skip=frame_skip, max_frames=max_frames)

    def export_poses(self, output_path: str, frame_skip: int = 1, max_frames: int | None = None):
        self.pose_max_frames = max_frames
        super().export_poses(output_path, frame_skip=frame_skip, max_frames=max_frames)


class Pi3ScanNetSensExportTests(unittest.TestCase):
    def test_list_scannet_sens_paths_requires_existing_sens_inputs(self):
        module = load_helper_module()

        with TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "sens"):
                module.list_scannet_sens_paths(Path(tmpdir))

    def test_list_scannet_sens_paths_honors_limit_and_sorting(self):
        module = load_helper_module()

        with TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir)
            for name in ("scene0002_00.sens", "scene0000_00.sens", "scene0001_00.sens"):
                (source_root / name).write_bytes(b"sens")

            sens_paths = module.list_scannet_sens_paths(source_root, limit_seqs=2)

            self.assertEqual([path.name for path in sens_paths], ["scene0000_00.sens", "scene0001_00.sens"])

    def test_export_scannet_sens_sequence_writes_raw_layout_summary(self):
        module = load_helper_module()

        with TemporaryDirectory() as tmpdir:
            sens_path = Path(tmpdir) / "scene0000_00.sens"
            output_seq = Path(tmpdir) / "raw" / "scene0000_00"
            sens_path.write_bytes(b"sens")

            summary = module.export_scannet_sens_sequence(
                sens_path=sens_path,
                output_seq=output_seq,
                sensor_data_cls=FakeSensorData,
            )

            self.assertEqual(summary["sequence"], "scene0000_00")
            self.assertEqual(summary["num_exported_color_frames"], 2)
            self.assertEqual(summary["num_exported_depth_frames"], 2)
            self.assertEqual(summary["num_exported_pose_frames"], 2)
            self.assertTrue((output_seq / "color" / "0.jpg").is_file())
            self.assertTrue((output_seq / "depth" / "3.png").is_file())
            self.assertTrue((output_seq / "pose" / "3.txt").is_file())
            self.assertTrue((output_seq / "intrinsic" / "intrinsic_color.txt").is_file())

    def test_export_scannet_sens_sequence_passes_max_frames_to_sensor_exports(self):
        module = load_helper_module()

        with TemporaryDirectory() as tmpdir:
            sens_path = Path(tmpdir) / "scene0000_00.sens"
            output_seq = Path(tmpdir) / "raw" / "scene0000_00"
            sens_path.write_bytes(b"sens")

            module.export_scannet_sens_sequence(
                sens_path=sens_path,
                output_seq=output_seq,
                sensor_data_cls=TrackingSensorData,
                max_frames=270,
            )

            self.assertIsNotNone(TrackingSensorData.last_instance)
            self.assertEqual(TrackingSensorData.last_instance.color_max_frames, 270)
            self.assertEqual(TrackingSensorData.last_instance.depth_max_frames, 270)
            self.assertEqual(TrackingSensorData.last_instance.pose_max_frames, 270)

    def test_rgbd_frame_load_resolves_numpy_at_runtime(self):
        module = load_helper_module()

        frame_stream = io.BytesIO(
            struct.pack("f" * 16, *[float(i) for i in range(16)])
            + struct.pack("Q", 1)
            + struct.pack("Q", 2)
            + struct.pack("Q", 0)
            + struct.pack("Q", 0)
        )

        frame = module.RGBDFrame()
        frame.load(frame_stream)

        self.assertEqual(frame.camera_to_world.shape, (4, 4))
        self.assertEqual(frame.timestamp_color, 1)
        self.assertEqual(frame.timestamp_depth, 2)

    def test_helper_import_does_not_require_numpy(self):
        module = load_helper_module_without_numpy()

        with TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir)
            (source_root / "scene0000_00.sens").write_bytes(b"sens")
            sens_paths = module.list_scannet_sens_paths(source_root, limit_seqs=1)
            self.assertEqual([path.name for path in sens_paths], ["scene0000_00.sens"])

    def test_prepare_scannet_sens_exact_script_runs_full_pipeline_and_writes_summary(self):
        module = load_script_module()

        with TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "sens"
            raw_output_root = Path(tmpdir) / "raw"
            prepared_output_root = Path(tmpdir) / "prepared"
            source_root.mkdir(parents=True)
            (source_root / "scene0000_00.sens").write_bytes(b"sens")

            export_summary = {
                "source_root": str(source_root),
                "raw_output_root": str(raw_output_root),
                "num_sens_files": 1,
                "sequences": [{"sequence": "scene0000_00", "num_exported_color_frames": 2}],
            }
            prepared_summary = {
                "dataset": "scannetv2",
                "source_root": str(raw_output_root),
                "output_root": str(prepared_output_root),
                "num_sequences": 1,
                "sequences": [{"sequence": "scene0000_00", "num_selected_frames": 1}],
            }

            with (
                mock.patch.object(module, "export_scannet_sens_dataset", return_value=export_summary) as export_mock,
                mock.patch.object(module, "prepare_dataset", return_value=prepared_summary) as prepare_mock,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "prepare_pi3_scannet_sens_exact.py",
                        "--source-root",
                        str(source_root),
                        "--raw-output-root",
                        str(raw_output_root),
                        "--prepared-output-root",
                        str(prepared_output_root),
                    ],
                ),
            ):
                module.main()

            export_mock.assert_called_once()
            prepare_mock.assert_called_once_with(
                dataset_name="scannetv2",
                source_root=raw_output_root.resolve(),
                output_root=prepared_output_root.resolve(),
                limit_seqs=0,
            )
            self.assertEqual(export_mock.call_args.kwargs["max_frames"], 270)
            summary = json.loads((prepared_output_root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["num_sens_files"], 1)
            self.assertEqual(summary["raw_export"]["sequences"][0]["sequence"], "scene0000_00")
            self.assertEqual(summary["prepared_export"]["num_sequences"], 1)

    def test_prepare_scannet_sens_exact_script_honors_custom_max_raw_frames(self):
        module = load_script_module()

        with TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "sens"
            raw_output_root = Path(tmpdir) / "raw"
            prepared_output_root = Path(tmpdir) / "prepared"
            source_root.mkdir(parents=True)
            (source_root / "scene0000_00.sens").write_bytes(b"sens")

            with (
                mock.patch.object(module, "export_scannet_sens_dataset", return_value={"num_sens_files": 0, "sequences": []}) as export_mock,
                mock.patch.object(module, "prepare_dataset", return_value={"num_sequences": 0, "sequences": []}),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "prepare_pi3_scannet_sens_exact.py",
                        "--source-root",
                        str(source_root),
                        "--raw-output-root",
                        str(raw_output_root),
                        "--prepared-output-root",
                        str(prepared_output_root),
                        "--max-raw-frames",
                        "12",
                    ],
                ),
            ):
                module.main()

            self.assertEqual(export_mock.call_args.kwargs["max_frames"], 12)

    def test_prepare_scannet_sens_exact_script_honors_skip_flags(self):
        module = load_script_module()

        with TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "sens"
            raw_output_root = Path(tmpdir) / "raw"
            prepared_output_root = Path(tmpdir) / "prepared"
            source_root.mkdir(parents=True)
            raw_output_root.mkdir(parents=True)
            (source_root / "scene0000_00.sens").write_bytes(b"sens")

            with (
                mock.patch.object(module, "export_scannet_sens_dataset") as export_mock,
                mock.patch.object(module, "prepare_dataset") as prepare_mock,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "prepare_pi3_scannet_sens_exact.py",
                        "--source-root",
                        str(source_root),
                        "--raw-output-root",
                        str(raw_output_root),
                        "--prepared-output-root",
                        str(prepared_output_root),
                        "--skip-raw-export",
                        "--skip-prepared-export",
                    ],
                ),
            ):
                module.main()

            export_mock.assert_not_called()
            prepare_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
