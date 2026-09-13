from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from PIL import Image


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aidi"
    / "scripts"
    / "vggt"
    / "eval_da3_pose_benchmark.py"
)


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"Missing evaluator script: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("eval_da3_pose_benchmark", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_dtu_camera_file(path: Path, intrinsics: np.ndarray, extrinsics: np.ndarray) -> None:
    lines = [
        "extrinsic",
        *[" ".join(f"{value:.8f}" for value in row) for row in extrinsics],
        "",
        "intrinsic",
        *[" ".join(f"{value:.8f}" for value in row) for row in intrinsics],
        "",
        "0.0 0.0",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class VggtDa3PoseEvalTests(unittest.TestCase):
    def test_parse_args_accepts_pi3_model_family(self):
        module = load_module()

        original_argv = sys.argv[:]
        try:
            sys.argv = ["eval_da3_pose_benchmark.py", "--model-family", "pi3"]
            args = module.parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual(args.model_family, "pi3")

    def test_c2w_to_w2c_inverts_pose_stack(self):
        module = load_module()

        c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
        c2w[1, :3, 3] = np.array([1.0, -2.0, 3.0], dtype=np.float32)

        w2c = module.c2w_to_w2c(c2w)

        self.assertEqual(tuple(w2c.shape), (2, 4, 4))
        self.assertTrue(np.allclose(w2c[0], np.eye(4, dtype=np.float32)))
        self.assertTrue(np.allclose(w2c[1] @ c2w[1], np.eye(4, dtype=np.float32), atol=1e-5))

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

    def test_default_dataset_roots_match_da3_benchmark_layout(self):
        module = load_module()

        roots = module.DEFAULT_DATA_ROOTS
        self.assertTrue(str(roots["eth3d"]).endswith("workspace/benchmark_dataset/eth3d"))
        self.assertTrue(str(roots["hiroom"]).endswith("workspace/benchmark_dataset/hiroom/data"))
        self.assertTrue(str(roots["scannetpp"]).endswith("workspace/benchmark_dataset/scannetpp"))
        self.assertTrue(str(roots["dtu64"]).endswith("workspace/benchmark_dataset/dtu64"))
        self.assertTrue(str(roots["7scenes"]).endswith("workspace/benchmark_dataset/7scenes"))

    def test_da3_scene_lists_match_expected_counts(self):
        module = load_module()

        self.assertEqual(len(module.DA3_ETH3D_SCENES), 11)
        self.assertEqual(len(module.DA3_SCANNETPP_SCENES), 20)
        self.assertEqual(len(module.DA3_DTU64_SCENES), 13)
        self.assertIn("courtyard", module.DA3_ETH3D_SCENES)
        self.assertIn("09c1414f1b", module.DA3_SCANNETPP_SCENES)
        self.assertIn("scan105", module.DA3_DTU64_SCENES)

    def test_sample_frame_indices_uses_fixed_seed_42(self):
        module = load_module()

        actual = module.sample_frame_indices(num_frames=8, max_frames=4)
        expected = [3, 4, 6, 7]

        self.assertEqual(actual, expected)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch not installed in local test env")
    def test_preprocess_images_for_vggt_da3_bench_matches_benchmark_resize_and_normalize(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "frame.jpg"
            Image.new("RGB", (1000, 500), color=(255, 255, 255)).save(image_path)

            images = module.preprocess_images_for_vggt(
                [str(image_path)],
                image_preprocess_style="da3_bench",
                process_res=504,
                process_res_method="upper_bound_resize",
            )

        self.assertEqual(tuple(images.shape), (1, 3, 252, 504))
        expected = np.array(
            [
                (1.0 - 0.485) / 0.229,
                (1.0 - 0.456) / 0.224,
                (1.0 - 0.406) / 0.225,
            ],
            dtype=np.float32,
        )
        self.assertTrue(np.allclose(images[0, :, 0, 0].numpy(), expected, atol=1e-5))

    def test_compute_pose_metrics_returns_one_for_perfect_predictions(self):
        module = load_module()

        gt = np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0)
        gt[1, :3, 3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        gt[2, :3, 3] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        gt[3, :3, 3] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        metrics = module.compute_pose_metrics(pred_w2c=gt.copy(), gt_w2c=gt.copy())

        self.assertAlmostEqual(metrics["Auc3"], 1.0)
        self.assertAlmostEqual(metrics["Auc30"], 1.0)

    def test_build_summary_means_auc_fields(self):
        module = load_module()

        summary = module.build_summary(
            [
                {"Auc3": 0.8, "Auc30": 0.9},
                {"Auc3": 0.4, "Auc30": 0.7},
            ]
        )

        self.assertAlmostEqual(summary["Auc3"], 0.6)
        self.assertAlmostEqual(summary["Auc30"], 0.8)
        self.assertEqual(summary["num_scenes"], 2)

    def test_load_hiroom_scene_data_uses_sorted_images_and_shared_intrinsics(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "hiroom" / "data"
            scene = "floor1/room2/scan3"
            scene_root = root / scene
            (scene_root / "image").mkdir(parents=True)
            (scene_root / "pose").mkdir(parents=True)
            intrinsics = np.array(
                [[100.0, 0.0, 50.0], [0.0, 110.0, 40.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            np.save(scene_root / "cam_K.npy", intrinsics)
            for stem in ("000002", "000001"):
                (scene_root / "image" / f"{stem}.jpg").write_bytes(b"jpg")
                pose = np.eye(4, dtype=np.float32)
                pose[0, 3] = float(stem)
                np.save(scene_root / "pose" / f"{stem}.npy", pose)

            scene_data = module.load_hiroom_scene_data(root=root, scene=scene)

        self.assertEqual(
            [Path(path).name for path in scene_data.image_files],
            ["000001.jpg", "000002.jpg"],
        )
        self.assertEqual(tuple(scene_data.extrinsics.shape), (2, 4, 4))
        self.assertTrue(np.allclose(scene_data.intrinsics[0], intrinsics))
        self.assertTrue(np.allclose(scene_data.intrinsics[1], intrinsics))

    def test_load_eth3d_scene_data_accepts_colmap_names_prefixed_with_dslr_images(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "eth3d"
            scene_root = root / "courtyard"
            image_root = scene_root / "images" / "dslr_images"
            calib_root = scene_root / "dslr_calibration_jpg"
            image_root.mkdir(parents=True)
            calib_root.mkdir(parents=True)
            (image_root / "DSC_0307.JPG").write_bytes(b"jpg")
            (calib_root / "cameras.txt").write_text("stub", encoding="utf-8")
            (calib_root / "images.txt").write_text("stub", encoding="utf-8")

            fake_camera = SimpleNamespace(model="PINHOLE", params=np.array([100.0, 110.0, 50.0, 40.0], dtype=np.float32))
            fake_image = SimpleNamespace(
                name="dslr_images/DSC_0307.JPG",
                camera_id=1,
                qvec=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                tvec=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            )

            scene_data = module.load_eth3d_scene_data(
                root=root,
                scene="courtyard",
                read_cameras_text_fn=lambda _path: {1: fake_camera},
                read_images_text_fn=lambda _path: {1: fake_image},
                qvec2rotmat_fn=lambda _qvec: np.eye(3, dtype=np.float32),
            )

        self.assertEqual([Path(path).name for path in scene_data.image_files], ["DSC_0307.JPG"])
        self.assertEqual(tuple(scene_data.extrinsics.shape), (1, 4, 4))
        self.assertEqual(tuple(scene_data.intrinsics.shape), (1, 3, 3))

    def test_load_dtu64_scene_data_reorders_view_33_first(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dtu64"
            scene_root = root / "scan105" / "image"
            camera_root = root / "Cameras"
            scene_root.mkdir(parents=True)
            camera_root.mkdir(parents=True)

            intrinsics = np.eye(3, dtype=np.float32)
            extrinsics = np.eye(4, dtype=np.float32)
            for idx in range(40):
                (scene_root / f"{idx:08d}.png").write_bytes(b"png")
                cur_ext = extrinsics.copy()
                cur_ext[0, 3] = float(idx)
                write_dtu_camera_file(camera_root / f"{idx:08d}_cam.txt", intrinsics, cur_ext)

            scene_data = module.load_dtu64_scene_data(
                root=root,
                camera_root=camera_root,
                scene="scan105",
            )

        names = [Path(path).name for path in scene_data.image_files[:4]]
        self.assertEqual(names[0], "00000033.png")
        self.assertEqual(names[1], "00000000.png")
        self.assertEqual(names[2], "00000001.png")
        self.assertEqual(scene_data.extrinsics[0, 0, 3], 33.0)

    def test_load_dtu64_scene_data_supports_scene_local_camera_layout(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dtu_raw"
            scene_root = root / "scan105"
            (scene_root / "images" / "00").mkdir(parents=True)
            (scene_root / "cameras" / "00").mkdir(parents=True)
            (scene_root / "images" / "00" / "000001.jpg").write_bytes(b"jpg")
            (scene_root / "images" / "00" / "000000.jpg").write_bytes(b"jpg")
            (scene_root / "cameras" / "00" / "intri.yml").write_text("stub", encoding="utf-8")
            (scene_root / "cameras" / "00" / "extri.yml").write_text("stub", encoding="utf-8")

            fake_cameras = {
                "000001": SimpleNamespace(
                    K=np.array([[200.0, 0.0, 30.0], [0.0, 210.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                    RT=np.array(
                        [[1.0, 0.0, 0.0, 1.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                        dtype=np.float32,
                    ),
                ),
                "000000": SimpleNamespace(
                    K=np.array([[100.0, 0.0, 10.0], [0.0, 110.0, 20.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                    RT=np.array(
                        [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                        dtype=np.float32,
                    ),
                ),
            }

            scene_data = module.load_dtu64_scene_data(
                root=root,
                camera_root=root / "Cameras",
                scene="scan105",
                read_camera_fn=lambda *_args, **_kwargs: fake_cameras,
            )

        self.assertEqual(
            [Path(path).name for path in scene_data.image_files],
            ["000000.jpg", "000001.jpg"],
        )
        self.assertEqual(tuple(scene_data.extrinsics.shape), (2, 4, 4))
        self.assertEqual(scene_data.extrinsics[0, 0, 3], 0.5)
        self.assertEqual(scene_data.extrinsics[1, 0, 3], 1.5)
        self.assertAlmostEqual(scene_data.intrinsics[1, 0, 0], 200.0)

    def test_load_scannetpp_scene_data_filters_iphone_frames(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "scannetpp"
            image_root = root / "09c1414f1b" / "merge_dslr_iphone" / "images"
            colmap_root = root / "09c1414f1b" / "merge_dslr_iphone" / "colmap" / "sparse_render_rgb"
            image_root.mkdir(parents=True)
            colmap_root.mkdir(parents=True)

            for name in ("frame_iphone_001.jpg", "frame_dslr_000.jpg", "frame_iphone_000.jpg"):
                (image_root / name).write_bytes(b"jpg")

            fake_camera = SimpleNamespace(
                model="PINHOLE",
                params=np.array([500.0, 510.0, 256.0, 128.0], dtype=np.float32),
                height=480,
                width=640,
            )
            fake_images = {
                10: SimpleNamespace(
                    name="frame_iphone_001.jpg",
                    camera_id=1,
                    tvec=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    qvec2rotmat=lambda: np.eye(3, dtype=np.float32),
                ),
                11: SimpleNamespace(
                    name="frame_dslr_000.jpg",
                    camera_id=1,
                    tvec=np.array([2.0, 0.0, 0.0], dtype=np.float32),
                    qvec2rotmat=lambda: np.eye(3, dtype=np.float32),
                ),
                12: SimpleNamespace(
                    name="frame_iphone_000.jpg",
                    camera_id=1,
                    tvec=np.array([3.0, 0.0, 0.0], dtype=np.float32),
                    qvec2rotmat=lambda: np.eye(3, dtype=np.float32),
                ),
            }

            scene_data = module.load_scannetpp_scene_data(
                root=root,
                scene="09c1414f1b",
                read_model_fn=lambda _: ({1: fake_camera}, fake_images, {}),
            )

        self.assertEqual(
            [Path(path).name for path in scene_data.image_files],
            ["frame_iphone_000.jpg", "frame_iphone_001.jpg"],
        )
        self.assertEqual(tuple(scene_data.extrinsics.shape), (2, 4, 4))
        self.assertEqual(scene_data.extrinsics[0, 0, 3], 3.0)
        self.assertAlmostEqual(scene_data.intrinsics[0, 0, 2], 255.5)

    def test_load_scannetpp_scene_data_supports_evc_layout(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "scannetpp_evc"
            scene_root = root / "8b5caf3398"
            (scene_root / "images" / "000001").mkdir(parents=True)
            (scene_root / "images" / "000000").mkdir(parents=True)
            (scene_root / "images" / "000001" / "000000.jpg").write_bytes(b"jpg")
            (scene_root / "images" / "000000" / "000000.jpg").write_bytes(b"jpg")
            (scene_root / "intri.yml").write_text("stub", encoding="utf-8")
            (scene_root / "extri.yml").write_text("stub", encoding="utf-8")

            fake_cameras = {
                "000001": SimpleNamespace(
                    K=np.array([[200.0, 0.0, 30.0], [0.0, 210.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                    RT=np.array(
                        [[1.0, 0.0, 0.0, 1.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                        dtype=np.float32,
                    ),
                ),
                "000000": SimpleNamespace(
                    K=np.array([[100.0, 0.0, 10.0], [0.0, 110.0, 20.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                    RT=np.array(
                        [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                        dtype=np.float32,
                    ),
                ),
            }

            scene_data = module.load_scannetpp_scene_data(
                root=root,
                scene="8b5caf3398",
                read_camera_fn=lambda *_args, **_kwargs: fake_cameras,
            )

        self.assertEqual(
            [Path(path).as_posix().split("/")[-2:] for path in scene_data.image_files],
            [["000000", "000000.jpg"], ["000001", "000000.jpg"]],
        )
        self.assertEqual(tuple(scene_data.extrinsics.shape), (2, 4, 4))
        self.assertEqual(scene_data.extrinsics[0, 0, 3], 0.5)
        self.assertEqual(scene_data.extrinsics[1, 0, 3], 1.5)
        self.assertAlmostEqual(scene_data.intrinsics[1, 0, 0], 200.0)

    def test_select_scene_names_uses_eth3d_constants_when_present(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "eth3d"
            (root / "courtyard").mkdir(parents=True)
            (root / "pipes").mkdir(parents=True)
            (root / "misc").mkdir(parents=True)

            scene_names = module.select_scene_names(
                dataset_name="eth3d",
                roots={"eth3d": root},
                requested_scenes=[],
            )

        self.assertEqual(scene_names, ["courtyard", "pipes"])

    def test_load_eth3d_scene_data_supports_evc_layout(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "eth3d"
            scene_root = root / "courtyard"
            (scene_root / "images" / "000001").mkdir(parents=True)
            (scene_root / "images" / "000000").mkdir(parents=True)
            (scene_root / "images" / "000001" / "000000.jpg").write_bytes(b"jpg")
            (scene_root / "images" / "000000" / "000000.jpg").write_bytes(b"jpg")
            (scene_root / "intri.yml").write_text("stub", encoding="utf-8")
            (scene_root / "extri.yml").write_text("stub", encoding="utf-8")

            fake_cameras = {
                "000001": SimpleNamespace(
                    K=np.array([[200.0, 0.0, 30.0], [0.0, 210.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                    RT=np.array(
                        [[1.0, 0.0, 0.0, 1.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                        dtype=np.float32,
                    ),
                ),
                "000000": SimpleNamespace(
                    K=np.array([[100.0, 0.0, 10.0], [0.0, 110.0, 20.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                    RT=np.array(
                        [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                        dtype=np.float32,
                    ),
                ),
            }

            scene_data = module.load_eth3d_scene_data(
                root=root,
                scene="courtyard",
                read_camera_fn=lambda *_args, **_kwargs: fake_cameras,
            )

        self.assertEqual(
            [Path(path).as_posix().split("/")[-2:] for path in scene_data.image_files],
            [["000000", "000000.jpg"], ["000001", "000000.jpg"]],
        )
        self.assertEqual(tuple(scene_data.extrinsics.shape), (2, 4, 4))
        self.assertEqual(scene_data.extrinsics[0, 0, 3], 0.5)
        self.assertEqual(scene_data.extrinsics[1, 0, 3], 1.5)
        self.assertAlmostEqual(scene_data.intrinsics[1, 0, 0], 200.0)

    def test_load_eth3d_scene_data_supports_pi3_prepared_layout(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "eth3d"
            scene_root = root / "delivery_area"
            image_root = scene_root / "images" / "custom_undistorted"
            camera_root = scene_root / "custom_undistorted_cam"
            image_root.mkdir(parents=True)
            camera_root.mkdir(parents=True)

            (image_root / "710.JPG").write_bytes(b"jpg")
            (image_root / "711.JPG").write_bytes(b"jpg")
            np.savez(
                camera_root / "710.npz",
                K=np.array([[500.0, 0.0, 250.0], [0.0, 510.0, 120.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                R=np.eye(3, dtype=np.float32),
                T=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            )
            np.savez(
                camera_root / "711.npz",
                K=np.array([[900.0, 0.0, 250.0], [0.0, 910.0, 120.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                R=np.eye(3, dtype=np.float32),
                T=np.array([9.0, 9.0, 9.0], dtype=np.float32),
            )

            scene_data = module.load_eth3d_scene_data(root=root, scene="delivery_area")

        self.assertEqual([Path(path).name for path in scene_data.image_files], ["710.JPG"])
        self.assertEqual(tuple(scene_data.extrinsics.shape), (1, 4, 4))
        self.assertEqual(scene_data.extrinsics[0, 0, 3], 1.0)
        self.assertAlmostEqual(scene_data.intrinsics[0, 0, 0], 500.0)

    def test_load_eth3d_scene_data_supports_prepared_intrinsics_extrinsics_schema(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "eth3d"
            scene_root = root / "courtyard"
            image_root = scene_root / "images" / "custom_undistorted"
            camera_root = scene_root / "custom_undistorted_cam"
            image_root.mkdir(parents=True)
            camera_root.mkdir(parents=True)

            (image_root / "DSC_0001.JPG").write_bytes(b"jpg")
            extrinsics = np.eye(4, dtype=np.float32)
            extrinsics[0, 3] = 2.5
            np.savez(
                camera_root / "DSC_0001.npz",
                intrinsics=np.array([[700.0, 0.0, 300.0], [0.0, 710.0, 200.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                extrinsics=extrinsics,
            )

            scene_data = module.load_eth3d_scene_data(root=root, scene="courtyard")

        self.assertEqual([Path(path).name for path in scene_data.image_files], ["DSC_0001.JPG"])
        self.assertEqual(tuple(scene_data.extrinsics.shape), (1, 4, 4))
        self.assertEqual(scene_data.extrinsics[0, 0, 3], 2.5)
        self.assertAlmostEqual(scene_data.intrinsics[0, 0, 0], 700.0)

    def test_select_scene_names_discovers_7scenes_sequences(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "7scenes"
            (root / "chess_seq-01").mkdir(parents=True)
            (root / "fire_seq-02").mkdir(parents=True)
            (root / "misc").mkdir(parents=True)

            scene_names = module.select_scene_names(
                dataset_name="7scenes",
                roots={"7scenes": root},
                requested_scenes=[],
            )

        self.assertEqual(scene_names, ["chess", "fire"])

    def test_load_7scenes_scene_data_supports_evc_layout(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "7scenes"
            scene_root = root / "chess_seq-01"
            (scene_root / "images" / "000001").mkdir(parents=True)
            (scene_root / "images" / "000000").mkdir(parents=True)
            (scene_root / "images" / "000001" / "000001.png").write_bytes(b"png")
            (scene_root / "images" / "000000" / "000000.png").write_bytes(b"png")
            (scene_root / "intri.yml").write_text("stub", encoding="utf-8")
            (scene_root / "extri.yml").write_text("stub", encoding="utf-8")

            fake_cameras = {
                "000001": SimpleNamespace(
                    K=np.array([[200.0, 0.0, 30.0], [0.0, 210.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                    RT=np.array(
                        [[1.0, 0.0, 0.0, 1.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                        dtype=np.float32,
                    ),
                ),
                "000000": SimpleNamespace(
                    K=np.array([[100.0, 0.0, 10.0], [0.0, 110.0, 20.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                    RT=np.array(
                        [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                        dtype=np.float32,
                    ),
                ),
            }

            scene_data = module.load_7scenes_scene_data(
                root=root,
                scene="chess_seq-01",
                read_camera_fn=lambda *_args, **_kwargs: fake_cameras,
            )

        self.assertEqual(
            [Path(path).as_posix().split("/")[-2:] for path in scene_data.image_files],
            [["000000", "000000.png"], ["000001", "000001.png"]],
        )
        self.assertEqual(tuple(scene_data.extrinsics.shape), (2, 4, 4))
        self.assertEqual(scene_data.extrinsics[0, 0, 3], 0.5)
        self.assertEqual(scene_data.extrinsics[1, 0, 3], 1.5)
        self.assertAlmostEqual(scene_data.intrinsics[1, 0, 0], 200.0)

    def test_load_7scenes_scene_data_aggregates_evc_sequences_by_scene_prefix(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "7scenes" / "test"
            scene_root_a = root / "chess_seq-01"
            scene_root_b = root / "chess_seq-02"
            for scene_root in (scene_root_a, scene_root_b):
                (scene_root / "images" / "000000").mkdir(parents=True)
                (scene_root / "images" / "000001").mkdir(parents=True)
                (scene_root / "images" / "000000" / "000000.png").write_bytes(b"png")
                (scene_root / "images" / "000001" / "000001.png").write_bytes(b"png")
                (scene_root / "intri.yml").write_text("stub", encoding="utf-8")
                (scene_root / "extri.yml").write_text("stub", encoding="utf-8")

            fake_camera_roots = {
                str(scene_root_a): {
                    "000000": SimpleNamespace(
                        K=np.array([[100.0, 0.0, 10.0], [0.0, 110.0, 20.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                        RT=np.array(
                            [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                            dtype=np.float32,
                        ),
                    ),
                    "000001": SimpleNamespace(
                        K=np.array([[200.0, 0.0, 30.0], [0.0, 210.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                        RT=np.array(
                            [[1.0, 0.0, 0.0, 1.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                            dtype=np.float32,
                        ),
                    ),
                },
                str(scene_root_b): {
                    "000000": SimpleNamespace(
                        K=np.array([[300.0, 0.0, 50.0], [0.0, 310.0, 60.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                        RT=np.array(
                            [[1.0, 0.0, 0.0, 2.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                            dtype=np.float32,
                        ),
                    ),
                    "000001": SimpleNamespace(
                        K=np.array([[400.0, 0.0, 70.0], [0.0, 410.0, 80.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                        RT=np.array(
                            [[1.0, 0.0, 0.0, 3.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                            dtype=np.float32,
                        ),
                    ),
                },
            }

            def fake_read_camera(intri_path, *_args, **_kwargs):
                return fake_camera_roots[str(Path(intri_path).parent)]

            scene_data = module.load_7scenes_scene_data(
                root=root,
                scene="chess",
                read_camera_fn=fake_read_camera,
            )

        self.assertEqual(len(scene_data.image_files), 4)
        self.assertEqual(tuple(scene_data.extrinsics.shape), (4, 4, 4))
        self.assertAlmostEqual(scene_data.extrinsics[0, 0, 3], 0.5)
        self.assertAlmostEqual(scene_data.extrinsics[-1, 0, 3], 3.5)
        self.assertAlmostEqual(scene_data.intrinsics[-1, 0, 0], 400.0)

    def test_load_7scenes_scene_data_supports_official_raw_layout(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "7scenes"
            scene_root = root / "7Scenes" / "chess"
            seq_root = scene_root / "seq-01"
            seq_root.mkdir(parents=True)
            (scene_root / "TestSplit.txt").write_text("sequence01\n", encoding="utf-8")
            (seq_root / "frame-000000.color.png").write_bytes(b"png")
            (seq_root / "frame-000001.color.png").write_bytes(b"png")
            np.savetxt(seq_root / "frame-000000.pose.txt", np.eye(4, dtype=np.float32))
            pose_c2w = np.eye(4, dtype=np.float32)
            pose_c2w[0, 3] = 2.0
            np.savetxt(seq_root / "frame-000001.pose.txt", pose_c2w)

            scene_data = module.load_7scenes_scene_data(root=root, scene="chess")

        self.assertEqual(
            [Path(path).name for path in scene_data.image_files],
            ["frame-000000.color.png", "frame-000001.color.png"],
        )
        self.assertEqual(tuple(scene_data.extrinsics.shape), (2, 4, 4))
        self.assertTrue(np.allclose(scene_data.extrinsics[0], np.eye(4, dtype=np.float32)))
        self.assertAlmostEqual(scene_data.extrinsics[1, 0, 3], -2.0)
        self.assertAlmostEqual(scene_data.intrinsics[0, 0, 0], 585.0)


if __name__ == "__main__":
    unittest.main()
