from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "aidi"
    / "scripts"
    / "vggt"
    / "eval_vggt_re10k_pose_lightweight.py"
)


def load_module():
    if not SCRIPT_PATH.is_file():
        raise AssertionError(f"Missing RE10K lightweight VGGT evaluator: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("eval_vggt_re10k_pose_lightweight", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VggtRe10kLightweightEvalTests(unittest.TestCase):
    def test_default_output_path_uses_tmp_json(self):
        module = load_module()
        args = module.setup_args(
            [
                "--re10k-root",
                "/tmp/re10k",
            ]
        )
        output_path = module.default_output_path(args)
        self.assertTrue(str(output_path).endswith("re10k_vggt_lightweight_vggt_official_seed20260215.json"))

    def test_build_result_payload_keeps_summary_and_failures(self):
        module = load_module()
        payload = module.build_result_payload(
            args=module.setup_args(
                [
                    "--re10k-root",
                    "/tmp/re10k",
                    "--use-scene-order",
                    "--eval-frame-indices",
                    "0,1,2,3,4,5",
                    "--translation-error-mode",
                    "centers",
                    "--auc-combine",
                    "min",
                ]
            ),
            loaded_model="official://VGGT-1B",
            device="cuda:0",
            dtype="bfloat16",
            metrics=[{"cam:pose_auc_10": 0.1, "cam:pose_auc_20": 0.2, "cam:pose_auc_30": 0.3}],
            failures=[{"path": "scene0", "error": "boom"}],
        )
        self.assertEqual(payload["summary"]["cam:pose_auc_30_mean"], 0.3)
        self.assertEqual(payload["failures"][0]["path"], "scene0")
        self.assertEqual(payload["implementation"], "vggt_re10k_lightweight")
        self.assertTrue(payload["use_scene_order"])
        self.assertEqual(payload["eval_frame_indices"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(payload["translation_error_mode"], "centers")
        self.assertEqual(payload["auc_combine"], "min")

    def test_parse_and_subset_eval_frame_indices(self):
        module = load_module()

        poses = np.arange(8 * 3 * 4, dtype=np.float64).reshape(8, 3, 4)
        indices = module.parse_eval_frame_indices("0,2,5")
        subset = module.subset_pose_array_for_eval(poses, indices)

        self.assertEqual(indices, [0, 2, 5])
        self.assertTrue(np.array_equal(subset, poses[[0, 2, 5]]))

    def test_camera_cache_root_defaults_to_process_specific_path(self):
        module = load_module()

        default_args = module.setup_args(["--re10k-root", "/tmp/re10k"])
        custom_args = module.setup_args(
            [
                "--re10k-root",
                "/tmp/re10k",
                "--camera-cache-root",
                "/tmp/custom_re10k_cache",
            ]
        )

        self.assertEqual(
            module.resolve_camera_cache_root(default_args, pid=12345),
            Path("/tmp/re10k_vggt_cam_cache_12345"),
        )
        self.assertEqual(
            module.resolve_camera_cache_root(custom_args, pid=12345),
            Path("/tmp/custom_re10k_cache"),
        )

    def test_setup_args_accepts_candidate_pool_eval_flags(self):
        module = load_module()

        args = module.setup_args(
            [
                "--re10k-root",
                "/tmp/re10k",
                "--use-scene-order",
                "--pool-size",
                "0",
                "--eval-frame-indices",
                "0,1,2,3,4,5",
            ]
        )

        self.assertTrue(args.use_scene_order)
        self.assertEqual(args.pool_size, 0)
        self.assertEqual(args.eval_frame_indices, "0,1,2,3,4,5")

    def test_setup_args_accepts_explicit_pose_metric_mode(self):
        module = load_module()

        args = module.setup_args(
            [
                "--re10k-root",
                "/tmp/re10k",
                "--translation-error-mode",
                "centers",
                "--auc-combine",
                "min",
            ]
        )

        self.assertEqual(args.translation_error_mode, "centers")
        self.assertEqual(args.auc_combine, "min")

    def test_parse_devices_and_shard_scene_roots(self):
        module = load_module()
        self.assertEqual(module.parse_devices("0,2,cuda:5"), ["cuda:0", "cuda:2", "cuda:5"])

        scene_roots = [Path(f"/tmp/scene_{idx:02d}") for idx in range(7)]
        self.assertEqual(
            module.shard_scene_roots(scene_roots, num_shards=3, shard_index=1),
            [scene_roots[1], scene_roots[4]],
        )

    def test_should_launch_multi_gpu_only_for_parent_process(self):
        module = load_module()
        parent_args = module.setup_args(["--re10k-root", "/tmp/re10k", "--devices", "0,1,2"])
        self.assertTrue(module.should_launch_multi_gpu(parent_args))

        worker_args = module.setup_args(
            [
                "--re10k-root",
                "/tmp/re10k",
                "--devices",
                "0,1,2",
                "--num-shards",
                "3",
                "--shard-index",
                "1",
            ]
        )
        self.assertFalse(module.should_launch_multi_gpu(worker_args))

    def test_merge_worker_payloads_combines_metrics_and_devices(self):
        module = load_module()
        args = module.setup_args(["--re10k-root", "/tmp/re10k", "--devices", "0,1"])
        merged = module.merge_worker_payloads(
            args,
            [
                {
                    "loaded_model": "official://VGGT-1B",
                    "dtype": "bfloat16",
                    "device": "cuda:0",
                    "devices": ["cuda:0"],
                    "metrics": [{"cam:pose_auc_30": 0.3, "cam:pose_auc_20": 0.2, "cam:pose_auc_10": 0.1}],
                    "failures": [],
                },
                {
                    "loaded_model": "official://VGGT-1B",
                    "dtype": "bfloat16",
                    "device": "cuda:1",
                    "devices": ["cuda:1"],
                    "metrics": [{"cam:pose_auc_30": 0.6, "cam:pose_auc_20": 0.4, "cam:pose_auc_10": 0.2}],
                    "failures": [{"path": "scene_bad", "error": "boom"}],
                },
            ],
        )
        self.assertEqual(merged["summary"]["metrics_count"], 2)
        self.assertAlmostEqual(merged["summary"]["cam:pose_auc_30_mean"], 0.45, places=8)
        self.assertEqual(merged["devices"], ["cuda:0", "cuda:1"])
        self.assertEqual(merged["failures"][0]["path"], "scene_bad")

    def test_build_worker_command_sets_shard_specific_args(self):
        module = load_module()
        args = module.setup_args(
            [
                "--re10k-root",
                "/tmp/re10k",
                "--official-ckpt-root",
                "/tmp/official",
                "--devices",
                "0,1",
                "--output",
                "/tmp/merged.json",
            ]
        )
        command = module.build_worker_command(
            args,
            device="cuda:1",
            shard_index=1,
            num_shards=2,
            output_path=Path("/tmp/shard01.json"),
        )
        self.assertIn("--device", command)
        self.assertIn("cuda:1", command)
        self.assertIn("--num-shards", command)
        self.assertIn("2", command)
        self.assertIn("--shard-index", command)
        self.assertIn("1", command)
        self.assertIn("/tmp/shard01.json", command)

    def test_build_worker_launch_spec_masks_single_physical_gpu(self):
        module = load_module()
        env, worker_device = module.build_worker_launch_spec("cuda:7", {"PYTHONUNBUFFERED": "1"})
        self.assertEqual(env["PYTHONUNBUFFERED"], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "7")
        self.assertEqual(worker_device, "cuda:0")

    def test_evc_compatible_preprocess_matches_reference_crop(self):
        import cv2

        from easyvolcap.official_vggt.utils.load_fn import load_and_preprocess_images_evc_compatible
        from easyvolcap.utils.data_utils import as_torch_func, load_image_file
        from easyvolcap.utils.image_utils import fill_nhwc_image

        def reference_crop(paths: list[str], target_size: int) -> torch.Tensor:
            rgb = [torch.as_tensor(load_image_file(path)) for path in paths]
            heights = [int(im.shape[0]) for im in rgb]
            widths = [int(im.shape[1]) for im in rgb]
            div = 14

            def resize_nhwc(im: torch.Tensor, new_h: int, new_w: int):
                return as_torch_func(
                    lambda arr: cv2.resize(arr, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)
                )(im)

            for idx, im in enumerate(rgb):
                h0, w0 = heights[idx], widths[idx]
                new_w = int(target_size)
                new_h = max(div, int(round((h0 * (new_w / w0)) / div) * div))
                if new_h != h0 or new_w != w0:
                    im = resize_nhwc(im, new_h, new_w)
                if new_h > target_size:
                    start_y = (new_h - target_size) // 2
                    im = im[start_y:start_y + target_size, :, :]
                    new_h = target_size
                rgb[idx] = im
                heights[idx], widths[idx] = int(new_h), int(new_w)

            max_h, max_w = max(heights), max(widths)
            for idx, im in enumerate(rgb):
                if heights[idx] == max_h and widths[idx] == max_w:
                    continue
                rgb[idx] = fill_nhwc_image(im, size=(max_h, max_w), value=1.0, center=True)

            return torch.stack([im.permute(2, 0, 1).contiguous() for im in rgb], dim=0)

        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths: list[str] = []
            for idx, shape in enumerate(((365, 641), (379, 511))):
                height, width = shape
                pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
                path = Path(tmpdir) / f"image_{idx}.jpg"
                Image.fromarray(pixels, mode="RGB").save(path, quality=95)
                image_paths.append(str(path))

            expected = reference_crop(image_paths, target_size=518)
            actual = load_and_preprocess_images_evc_compatible(
                image_paths,
                mode="crop",
                target_size=518,
            )
            self.assertEqual(actual.shape, expected.shape)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
