import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class dotdict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _install_module_stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _load_dataset_module():
    stub_names = [
        "cv2",
        "torch",
        "torch.utils",
        "torch.utils.data",
        "easyvolcap.engine",
        "easyvolcap.utils.console_utils",
        "easyvolcap.utils.timer_utils",
        "easyvolcap.utils.base_utils",
        "easyvolcap.utils.ray_utils",
        "easyvolcap.utils.easy_utils",
        "easyvolcap.utils.parallel_utils",
        "easyvolcap.utils.vhull_utils",
        "easyvolcap.utils.dist_utils",
        "easyvolcap.utils.cam_utils",
        "easyvolcap.utils.math_utils",
        "easyvolcap.utils.bound_utils",
        "easyvolcap.utils.data_utils",
    ]
    missing = object()
    previous_modules = {name: sys.modules.get(name, missing) for name in stub_names}
    try:
        cv2 = _install_module_stub("cv2", setNumThreads=lambda _threads: None)
        cv2.Rodrigues = lambda *_args, **_kwargs: (_args[0], None)

        torch = _install_module_stub("torch")
        torch.Tensor = object
        torch.float = object()
        torch.long = object()
        _install_module_stub("torch.utils")
        _install_module_stub(
            "torch.utils.data",
            Dataset=type("Dataset", (), {}),
            get_worker_info=lambda: None,
        )

        class _Registry:
            def register_module(self):
                return lambda cls: cls

        class _EnumValue:
            def __init__(self, name):
                self.name = name

        class _EnumLike:
            TRAIN = _EnumValue("TRAIN")
            VAL = _EnumValue("VAL")
            TEST = _EnumValue("TEST")
            DISTANCE = _EnumValue("DISTANCE")
            ZIGZAG = _EnumValue("ZIGZAG")

            @classmethod
            def __class_getitem__(cls, key):
                return getattr(cls, key)

        _install_module_stub("easyvolcap.engine", DATASETS=_Registry(), cfg=dotdict(), args=dotdict(type=""))
        _install_module_stub(
            "easyvolcap.utils.console_utils",
            exists=lambda _path: False,
            log=lambda *_args, **_kwargs: None,
            yellow=lambda value: value,
            red=lambda value: value,
            tqdm=lambda values, **_kwargs: values,
            join=os.path.join,
        )
        _install_module_stub("easyvolcap.utils.timer_utils", timer=object())
        _install_module_stub("easyvolcap.utils.base_utils", dotdict=dotdict)
        _install_module_stub("easyvolcap.utils.ray_utils", get_rays=lambda *a, **k: None, weighted_sample_rays=lambda *a, **k: None)
        _install_module_stub("easyvolcap.utils.easy_utils", read_camera=lambda *a, **k: {})
        _install_module_stub("easyvolcap.utils.parallel_utils", parallel_execution=lambda *a, **k: None)
        _install_module_stub("easyvolcap.utils.vhull_utils", hierarchically_carve_vhull=lambda *a, **k: None)
        _install_module_stub(
            "easyvolcap.utils.dist_utils",
            get_rank=lambda: 0,
            get_world_size=lambda: 1,
            get_distributed=lambda: False,
        )
        _install_module_stub(
            "easyvolcap.utils.cam_utils",
            average_c2ws=lambda *a, **k: None,
            align_c2ws=lambda *a, **k: None,
            average_w2cs=lambda *a, **k: None,
            Sourcing=_EnumLike,
        )
        _install_module_stub(
            "easyvolcap.utils.math_utils",
            affine_inverse=lambda *a, **k: None,
            affine_padding=lambda *a, **k: None,
            torch_inverse_3x3=lambda *a, **k: None,
            point_padding=lambda *a, **k: None,
        )
        _install_module_stub(
            "easyvolcap.utils.bound_utils",
            get_bound_2d_bound=lambda *a, **k: None,
            get_bounds=lambda *a, **k: None,
            monotonic_near_far=lambda *a, **k: None,
            get_bound_3d_near_far=lambda *a, **k: None,
        )
        _install_module_stub(
            "easyvolcap.utils.data_utils",
            DataSplit=_EnumLike,
            UnstructuredTensors=object,
            load_resize_undist_ims_bytes=lambda *a, **k: None,
            load_image_from_bytes=lambda *a, **k: None,
            as_torch_func=lambda value: value,
            to_cuda=lambda value: value,
            to_cpu=lambda value: value,
            to_tensor=lambda value: value,
            export_pts=lambda *a, **k: None,
            load_pts=lambda *a, **k: None,
            decode_crop_fill_ims_bytes=lambda *a, **k: None,
            decode_fill_ims_bytes=lambda *a, **k: None,
            load_resize_undist_im_bytes=lambda *a, **k: None,
            decode_crop_fill_im_bytes=lambda *a, **k: None,
            normalize_image=lambda value: value,
        )

        module_name = "volumetric_video_dataset_under_test"
        module_path = Path(__file__).resolve().parents[1] / "easyvolcap/dataloaders/datasets/volumetric_video_dataset.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


dataset_module = _load_dataset_module()
VolumetricVideoDataset = dataset_module.VolumetricVideoDataset


class VolumetricVideoDatasetCameraProbeTests(unittest.TestCase):
    def _dataset(self, meta_data=None, n_frames_total=3):
        dataset = object.__new__(VolumetricVideoDataset)
        dataset.camera_root = ""
        dataset.data_root = "/dataset/scene"
        dataset.cameras_dir = "cameras"
        dataset.img_list_by_cam = None
        dataset.meta_data = meta_data
        dataset.intri_file = "intri.yml"
        dataset.n_frames_total = n_frames_total
        return dataset

    def test_meta_root_cameras_do_not_probe_cameras_dir(self):
        dataset = self._dataset(meta_data=dotdict({
            "intri": object(),
            "extri": object(),
        }))

        with mock.patch.object(VolumetricVideoDataset, "_probe_camera_dir", return_value=None) as probe:
            with mock.patch.object(dataset_module, "read_camera", return_value={"00": "cam00", "01": "cam01"}):
                cameras = dataset._VolumetricVideoDataset__load_cameras("extri.yml")

        probe.assert_not_called()
        self.assertEqual(list(cameras.keys()), ["00", "01"])
        self.assertEqual(cameras["00"], ["cam00", "cam00", "cam00"])

    def test_root_level_camera_files_do_not_probe_cameras_dir(self):
        dataset = self._dataset(meta_data=None)

        def fake_exists(path):
            return path in {
                "/dataset/scene/intri.yml",
                "/dataset/scene/extri.yml",
            }

        with mock.patch.object(VolumetricVideoDataset, "_probe_camera_dir", return_value=None) as probe:
            with mock.patch.object(dataset_module, "exists", side_effect=fake_exists):
                with mock.patch.object(dataset_module, "read_camera", return_value={"00": "cam00"}):
                    cameras = dataset._VolumetricVideoDataset__load_cameras("extri.yml")

        probe.assert_not_called()
        self.assertEqual(list(cameras.keys()), ["00"])
        self.assertEqual(cameras["00"], ["cam00", "cam00", "cam00"])

    def test_degenerated_meta_monocular_cameras_probe_once_then_fallback(self):
        dataset = self._dataset(meta_data=dotdict({
            "00/intri": object(),
            "00/extri": object(),
        }))

        def fake_read_camera(intri, extri, use_dict=False):
            if isinstance(intri, str) and intri.startswith("/dataset/scene/cameras/00/"):
                return {0: "disk0", 1: "disk1", 2: "disk2"}
            return {0: "meta0"}

        with mock.patch.object(VolumetricVideoDataset, "_probe_camera_dir", return_value=["00"]) as probe:
            with mock.patch.object(dataset_module, "read_camera", side_effect=fake_read_camera):
                cameras = dataset._VolumetricVideoDataset__load_cameras("extri.yml")

        probe.assert_called_once_with("/dataset/scene/cameras")
        self.assertEqual(list(cameras.keys()), ["00"])
        self.assertEqual(cameras["00"], ["disk0", "disk1", "disk2"])

    def test_monocular_camera_dir_fallback_still_probes_when_needed(self):
        dataset = self._dataset(meta_data=None)

        with mock.patch.object(VolumetricVideoDataset, "_probe_camera_dir", return_value=["00"]) as probe:
            with mock.patch.object(dataset_module, "exists", return_value=False):
                with mock.patch.object(dataset_module, "read_camera", return_value={0: "disk0", 1: "disk1"}):
                    cameras = dataset._VolumetricVideoDataset__load_cameras("extri.yml")

        probe.assert_called_once_with("/dataset/scene/cameras")
        self.assertEqual(list(cameras.keys()), ["00"])
        self.assertEqual(cameras["00"], ["disk0", "disk1"])


if __name__ == "__main__":
    unittest.main()
