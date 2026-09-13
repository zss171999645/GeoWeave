import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


def _install_import_stubs() -> None:
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        torch_mod.Tensor = object
        torch_mod.device = object
        torch_mod.dtype = object
        torch_mod.bfloat16 = object()
        torch_mod.float16 = object()
        sys.modules["torch"] = torch_mod

    if "torch.nn" not in sys.modules:
        sys.modules["torch.nn"] = types.ModuleType("torch.nn")
    if "torch.nn.functional" not in sys.modules:
        sys.modules["torch.nn.functional"] = types.ModuleType("torch.nn.functional")

    if "torchvision" not in sys.modules:
        sys.modules["torchvision"] = types.ModuleType("torchvision")
    if "torchvision.transforms" not in sys.modules:
        sys.modules["torchvision.transforms"] = types.ModuleType("torchvision.transforms")

    if "cv2" not in sys.modules:
        cv2_mod = types.ModuleType("cv2")
        cv2_mod.IMREAD_GRAYSCALE = 0
        cv2_mod.IMREAD_ANYCOLOR = 0
        cv2_mod.IMREAD_ANYDEPTH = 0
        cv2_mod.INTER_NEAREST = 0
        cv2_mod.imread = lambda *args, **kwargs: None
        cv2_mod.resize = lambda image, size, interpolation=None: image
        sys.modules["cv2"] = cv2_mod

    if "easyvolcap.engine" not in sys.modules:
        engine_mod = types.ModuleType("easyvolcap.engine")

        class _DummyConfig:
            @staticmethod
            def fromfile(path):  # pragma: no cover - helper for import only
                raise RuntimeError("Config.fromfile should not be called in this unit test")

        class _DummyModels:
            @staticmethod
            def build(cfg):  # pragma: no cover - helper for import only
                raise RuntimeError("MODELS.build should not be called in this unit test")

        engine_mod.Config = _DummyConfig
        engine_mod.MODELS = _DummyModels
        sys.modules["easyvolcap.engine"] = engine_mod

    if "easyvolcap.models.official_vggt_model" not in sys.modules:
        vggt_mod = types.ModuleType("easyvolcap.models.official_vggt_model")

        class _DummyOfficialVGGTModel:
            @staticmethod
            def _remap_special_token_keys(state_dict):
                return state_dict, {}

        vggt_mod.OfficialVGGTModel = _DummyOfficialVGGTModel
        sys.modules["easyvolcap.models.official_vggt_model"] = vggt_mod

    if "easyvolcap.utils.pi3.models.pi3" not in sys.modules:
        pi3_mod = types.ModuleType("easyvolcap.utils.pi3.models.pi3")

        class _DummyPi3:
            pass

        pi3_mod.Pi3 = _DummyPi3
        sys.modules["easyvolcap.utils.pi3.models.pi3"] = pi3_mod

    if "easyvolcap.utils.base_utils" not in sys.modules:
        base_utils_mod = types.ModuleType("easyvolcap.utils.base_utils")

        class _DotDict(dict):
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        base_utils_mod.dotdict = _DotDict
        sys.modules["easyvolcap.utils.base_utils"] = base_utils_mod

    if "easyvolcap.utils.pi3.kitti_videodepth" not in sys.modules:
        kitti_mod = types.ModuleType("easyvolcap.utils.pi3.kitti_videodepth")
        kitti_mod.explain_kitti_videodepth_layout_error = lambda root: f"bad root: {root}"
        kitti_mod.has_kitti_monst3r_gathered_layout = lambda root: False
        kitti_mod.has_kitti_official_flat_layout = lambda root: False
        sys.modules["easyvolcap.utils.pi3.kitti_videodepth"] = kitti_mod


_install_import_stubs()

from aidi.scripts.baselines import eval_pi3_monodepth_protocol as monodepth
from aidi.scripts.baselines import eval_pi3_videodepth_protocol as videodepth


EXTENDED_DATASETS = {"eth3d", "7scenes", "blendedmvs", "mvs_synth", "scannetpp", "vkitti2"}


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


class Pi3DepthProtocolDatasetExtensionTests(unittest.TestCase):
    def test_videodepth_parse_eval_frame_indices_supports_empty_and_csv(self) -> None:
        self.assertEqual(videodepth.parse_eval_frame_indices(""), [])
        self.assertEqual(videodepth.parse_eval_frame_indices("0, 2,5"), [0, 2, 5])

    def test_videodepth_resolve_eval_frame_indices_reads_tuple_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            seq_root = data_root / "demo"
            seq_root.mkdir(parents=True)
            (seq_root / "tuple_meta.json").write_text('{"eval_frame_indices": [4, 5, 6]}\n', encoding="utf-8")

            indices = videodepth.resolve_eval_frame_indices_for_sequence(
                eval_frame_indices_spec="tuple_meta",
                data_root=data_root,
                seq="demo",
            )

            self.assertEqual(indices, [4, 5, 6])

    def test_videodepth_subset_depth_sequence_for_eval_selects_requested_frames(self) -> None:
        sequence = [
            [[0, 0]],
            [[1, 1]],
            [[2, 2]],
        ]

        subset = videodepth.subset_depth_sequence_for_eval(sequence, [0, 2], "demo", "prediction")

        self.assertEqual(subset.tolist(), [[[0, 0]], [[2, 2]]])

    def test_extended_datasets_registered_in_both_protocols(self) -> None:
        self.assertTrue(EXTENDED_DATASETS.issubset(monodepth.DATASET_SPECS.keys()))
        self.assertTrue(EXTENDED_DATASETS.issubset(videodepth.DATASET_SPECS.keys()))

    def test_monodepth_detects_eth3d_view_major_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _touch(root / "courtyard" / "images" / "000001" / "000000.jpg")
            _touch(root / "courtyard" / "depths" / "000001" / "000000.exr")

            spec = monodepth.DATASET_SPECS["eth3d"]
            layout = monodepth.detect_layout(spec, root, "02")
            self.assertEqual(layout, "evc")

            names = monodepth.get_sequence_names_for_layout(spec, root, layout, "02")
            self.assertEqual(names, ["courtyard"])

            image_files, gt_files = monodepth.list_sequence_files_for_layout(spec, root, layout, "courtyard", "02")
            self.assertEqual([p.as_posix() for p in image_files], [(root / "courtyard" / "images" / "000001" / "000000.jpg").as_posix()])
            self.assertEqual([p.as_posix() for p in gt_files], [(root / "courtyard" / "depths" / "000001" / "000000.exr").as_posix()])

    def test_videodepth_detects_mvs_synth_view_major_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _touch(root / "GTAV_001" / "images" / "0001" / "000000.png")
            _touch(root / "GTAV_001" / "depths" / "0001" / "000000.exr")

            spec = videodepth.DATASET_SPECS["mvs_synth"]
            layout = videodepth.detect_layout(spec, root, "02")
            self.assertEqual(layout, "evc")

            names = videodepth.get_sequence_names_for_layout(spec, root, layout, "02")
            self.assertEqual(names, ["GTAV_001"])

            image_files, gt_files = videodepth.list_sequence_files_for_layout(spec, root, layout, "GTAV_001", "02")
            self.assertEqual([p.as_posix() for p in image_files], [(root / "GTAV_001" / "images" / "0001" / "000000.png").as_posix()])
            self.assertEqual([p.as_posix() for p in gt_files], [(root / "GTAV_001" / "depths" / "0001" / "000000.exr").as_posix()])

    def test_videodepth_detects_nested_vkitti2_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _touch(root / "Scene01" / "clone" / "images" / "02" / "000000.jpg")
            _touch(root / "Scene01" / "clone" / "depths" / "02" / "000000.exr")

            spec = videodepth.DATASET_SPECS["vkitti2"]
            layout = videodepth.detect_layout(spec, root, "02")
            self.assertEqual(layout, "evc")

            names = videodepth.get_sequence_names_for_layout(spec, root, layout, "02")
            self.assertEqual(names, ["Scene01/clone"])

            image_files, gt_files = videodepth.list_sequence_files_for_layout(spec, root, layout, "Scene01/clone", "02")
            self.assertEqual([p.as_posix() for p in image_files], [(root / "Scene01" / "clone" / "images" / "02" / "000000.jpg").as_posix()])
            self.assertEqual([p.as_posix() for p in gt_files], [(root / "Scene01" / "clone" / "depths" / "02" / "000000.exr").as_posix()])

    def test_videodepth_detects_generic_npy_camera_evc_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _touch(root / "tuple_a" / "images" / "00" / "frame_0000.jpg")
            _touch(root / "tuple_a" / "depths" / "00" / "frame_0000.npy")

            spec = videodepth.DATASET_SPECS["generic_npy"]
            layout = videodepth.detect_layout(spec, root, "00")
            self.assertEqual(layout, "evc")

            names = videodepth.get_sequence_names_for_layout(spec, root, layout, "00")
            self.assertEqual(names, ["tuple_a"])

            image_files, gt_files = videodepth.list_sequence_files_for_layout(spec, root, layout, "tuple_a", "00")
            self.assertEqual([p.as_posix() for p in image_files], [(root / "tuple_a" / "images" / "00" / "frame_0000.jpg").as_posix()])
            self.assertEqual([p.as_posix() for p in gt_files], [(root / "tuple_a" / "depths" / "00" / "frame_0000.npy").as_posix()])

    def test_videodepth_detects_generic_png_mm_camera_evc_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _touch(root / "tuple_a" / "images" / "00" / "frame_0000.jpg")
            _touch(root / "tuple_a" / "depths" / "00" / "frame_0000.png")

            spec = videodepth.DATASET_SPECS["generic_png_mm"]
            layout = videodepth.detect_layout(spec, root, "00")
            self.assertEqual(layout, "evc")

            names = videodepth.get_sequence_names_for_layout(spec, root, layout, "00")
            self.assertEqual(names, ["tuple_a"])

            image_files, gt_files = videodepth.list_sequence_files_for_layout(spec, root, layout, "tuple_a", "00")
            self.assertEqual([p.as_posix() for p in image_files], [(root / "tuple_a" / "images" / "00" / "frame_0000.jpg").as_posix()])
            self.assertEqual([p.as_posix() for p in gt_files], [(root / "tuple_a" / "depths" / "00" / "frame_0000.png").as_posix()])

    def test_monodepth_scannetpp_evc_does_not_probe_every_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _touch(root / "scene_a" / "images" / "000001" / "000000.jpg")
            _touch(root / "scene_a" / "depths" / "000001" / "000000.exr")
            _touch(root / "scene_b" / "images" / "000002" / "000000.jpg")
            _touch(root / "scene_b" / "depths" / "000002" / "000000.exr")

            spec = monodepth.DATASET_SPECS["scannetpp"]
            with mock.patch.object(
                monodepth,
                "is_eth3d_evc_train_layout",
                side_effect=[True, AssertionError("should not probe scannetpp evc layout more than once")],
            ):
                names = monodepth.get_sequence_names_for_layout(spec, root, "evc", "02")

            self.assertEqual(names, ["scene_a", "scene_b"])

    def test_videodepth_scannetpp_evc_does_not_probe_every_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _touch(root / "scene_a" / "images" / "000001" / "000000.jpg")
            _touch(root / "scene_a" / "depths" / "000001" / "000000.exr")
            _touch(root / "scene_b" / "images" / "000002" / "000000.jpg")
            _touch(root / "scene_b" / "depths" / "000002" / "000000.exr")

            spec = videodepth.DATASET_SPECS["scannetpp"]
            with mock.patch.object(
                videodepth,
                "is_eth3d_evc_train_layout",
                side_effect=[True, AssertionError("should not probe scannetpp evc layout more than once")],
            ):
                names = videodepth.get_sequence_names_for_layout(spec, root, "evc", "02")

            self.assertEqual(names, ["scene_a", "scene_b"])


if __name__ == "__main__":
    unittest.main()
