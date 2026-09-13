import argparse
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "aidi" / "scripts" / "vggt" / "vggt_global_attention_analyzer.py"
MODULE_NAME = "codex_vggt_global_attention_analyzer"


def _load_module():
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


class VggtGlobalAttentionAnalyzerTests(unittest.TestCase):
    def test_discover_official_seq_images_and_parse_attention_noise_name(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_root = Path(tmpdir) / "scene0707_00__anchor0066__noise"
            color_dir = seq_root / "color_90"
            color_dir.mkdir(parents=True)
            for name in ("frame_0002.jpg", "frame_0000.jpg", "frame_0001.jpg"):
                (color_dir / name).write_bytes(b"jpg")

            image_paths = module._discover_official_seq_image_paths(seq_root, "color_90")
            parsed = module._parse_attention_noise_tuple_name(seq_root.name)

            self.assertEqual([path.name for path in image_paths], ["frame_0000.jpg", "frame_0001.jpg", "frame_0002.jpg"])
            self.assertEqual(parsed["scene_id"], "scene0707_00")
            self.assertEqual(parsed["anchor_id"], "anchor0066")
            self.assertEqual(parsed["tuple_kind"], "noise")
            self.assertEqual(
                module._default_attention_noise_sample_note(parsed["tuple_kind"]),
                "attention-noise noise; target views 0-5; support views 6-9",
            )

    def test_build_minimal_analyzer_batch_from_images(self):
        module = _load_module()
        images = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)

        batch = module._build_minimal_analyzer_batch(images)

        self.assertEqual(tuple(batch["images"].shape), (1, 2, 3, 4, 5))
        self.assertEqual(tuple(batch["rgb"].shape), (1, 2, 20, 3))
        self.assertEqual(int(batch["meta"]["H"][0]), 4)
        self.assertEqual(int(batch["meta"]["W"][0]), 5)
        self.assertEqual(int(batch["meta"]["iter"]), 0)

    def test_build_seq_bundle_meta_defaults_from_tuple_name(self):
        module = _load_module()

        class _Args:
            bundle_id = ""
            bundle_label = ""
            model_id = "pt44"
            model_name = "Indexer VGGT PT44"
            sample_id = ""
            sample_name = ""
            sample_scene = ""
            sample_stride = 0
            sample_views = 0
            sample_note = ""

        images_uint8 = np.zeros((10, 392, 518, 3), dtype=np.uint8)
        parsed = {
            "scene_id": "scene0707_00",
            "anchor_id": "anchor0066",
            "tuple_kind": "noise",
        }

        meta = module._build_seq_bundle_meta(_Args(), "scene0707_00__anchor0066__noise", parsed, images_uint8)

        self.assertEqual(meta["sample_id"], "scene0707_00__anchor0066__noise")
        self.assertEqual(meta["sample_scene"], "scene0707_00")
        self.assertEqual(meta["bundle_id"], "scene0707_00__anchor0066__noise__pt44")
        self.assertEqual(meta["sample_note"], "attention-noise noise; target views 0-5; support views 6-9")
        self.assertEqual(meta["sample_views"], 10)

    def test_main_registers_capture_official_seq_subcommand(self):
        module = _load_module()
        parser = argparse.ArgumentParser("VGGT global attention analyzer")
        subparsers = parser.add_subparsers(dest="command", required=True)

        module._register_cli_commands(parser, subparsers)
        args = parser.parse_args([
            "capture-official-seq",
            "--config", "dummy.yaml",
            "--seq-root", "/tmp/scene0707_00__anchor0066__noise",
            "--out-dir", "/tmp/out",
        ])

        self.assertEqual(args.command, "capture-official-seq")
        self.assertIs(args.func, module.cmd_capture_official_seq)
        self.assertEqual(int(args.force_dense), 0)

    def test_load_repo_config_resolves_legacy_base_yaml(self):
        module = _load_module()
        fake_resolved = {"resolved": True}

        class _FakeConfig:
            @staticmethod
            def fromfile(path: str):
                return {"_base_": "./base.yaml", "model_cfg": {"vggt_cfg": {"indexer_cfg": {"topk": 1024}}}}

        with tempfile.TemporaryDirectory() as tmpdir:
            child_cfg = Path(tmpdir) / "child.yaml"
            child_cfg.write_text("_base_: ./base.yaml\n", encoding="utf-8")
            helper = types.SimpleNamespace(load_resolved_config=mock.Mock(return_value=fake_resolved))
            with mock.patch.object(module.importlib, "import_module", return_value=helper) as mocked_import:
                cfg = module._load_repo_config(child_cfg, _FakeConfig)

        self.assertIs(cfg, fake_resolved)
        mocked_import.assert_called_once_with("aidi.scripts.baselines.eval_config_utils")
        helper.load_resolved_config.assert_called_once_with(child_cfg)

    def test_ensure_pi3_weights_redownloads_invalid_existing_checkpoint(self):
        module = _load_module()

        class _Response(io.BytesIO):
            def __init__(self, payload: bytes, content_length: Optional[int] = None):
                super().__init__(payload)
                self.headers = {}
                if content_length is not None:
                    self.headers["Content-Length"] = str(content_length)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self.close()
                return False

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "model.safetensors"
            ckpt_path.write_bytes(b"broken")
            resolved_ckpt_path = ckpt_path.resolve()

            def _fake_validate(path: Path):
                return "corrupt header" if Path(path) == resolved_ckpt_path else None

            with mock.patch.object(module, "_validate_safetensors_file", side_effect=_fake_validate):
                with mock.patch.object(module.urllib.request, "urlopen", return_value=_Response(b"fresh-weights", len(b"fresh-weights"))):
                    resolved = module._ensure_pi3_weights(ckpt_path)

            self.assertEqual(resolved, ckpt_path.resolve())
            self.assertEqual(ckpt_path.read_bytes(), b"fresh-weights")

    def test_ensure_pi3_weights_rejects_truncated_download(self):
        module = _load_module()

        class _Response(io.BytesIO):
            def __init__(self, payload: bytes, content_length: int):
                super().__init__(payload)
                self.headers = {"Content-Length": str(content_length)}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self.close()
                return False

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "model.safetensors"
            with mock.patch.object(module.urllib.request, "urlopen", return_value=_Response(b"short", 12)):
                with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                    module._ensure_pi3_weights(ckpt_path)

    def test_frontend_registry_groups_compare_bundles_for_same_sample(self):
        module = _load_module()
        entries = {
            "scene01__official": {
                "bundle_id": "scene01__official",
                "bundle_label": "Scene01 | Official",
                "model_id": "official",
                "model_name": "Official VGGT",
                "sample_id": "scene01",
                "sample_name": "Scene01",
                "sample_scene": "scene-01",
                "sample_stride": 1,
                "sample_views": 12,
                "sample_note": "same input",
                "num_views": 12,
                "grid_h": 21,
                "grid_w": 37,
                "captured_layers": [0, 1],
                "layer_modes": {"00": "dense_qk", "01": "dense_qk"},
                "layer_num_heads": {"00": 16, "01": 16},
                "layer_indexer_num_heads": {"00": 0, "01": 0},
                "layer_has_indexer_scores": {"00": False, "01": False},
            },
            "scene01__pi3": {
                "bundle_id": "scene01__pi3",
                "bundle_label": "Scene01 | PI3",
                "model_id": "pi3",
                "model_name": "PI3",
                "sample_id": "scene01",
                "sample_name": "Scene01",
                "sample_scene": "scene-01",
                "sample_stride": 1,
                "sample_views": 12,
                "sample_note": "same input",
                "num_views": 12,
                "grid_h": 21,
                "grid_w": 37,
                "captured_layers": [1, 3],
                "layer_modes": {"01": "dense_qk", "03": "dense_qk"},
                "layer_num_heads": {"01": 16, "03": 16},
                "layer_indexer_num_heads": {"01": 0, "03": 0},
                "layer_has_indexer_scores": {"01": False, "03": False},
            },
            "scene01__pt34": {
                "bundle_id": "scene01__pt34",
                "bundle_label": "Scene01 | PT34",
                "model_id": "pt34",
                "model_name": "Indexer VGGT PT34",
                "sample_id": "scene01",
                "sample_name": "Scene01",
                "sample_scene": "scene-01",
                "sample_stride": 1,
                "sample_views": 12,
                "sample_note": "same input",
                "num_views": 12,
                "grid_h": 21,
                "grid_w": 37,
                "captured_layers": [0, 1],
                "layer_modes": {"00": "dense_qk", "01": "sparse_qk_topk"},
                "layer_num_heads": {"00": 16, "01": 16},
                "layer_indexer_num_heads": {"00": 0, "01": 4},
                "layer_has_indexer_scores": {"00": False, "01": True},
            },
        }

        registry = module._frontend_registry(entries)

        self.assertIn("sample_compare_bundle_ids", registry)
        self.assertEqual(
            registry["sample_compare_bundle_ids"]["scene01"],
            ["scene01__official", "scene01__pt34", "scene01__pi3"],
        )

    def test_format_token_descriptor_respects_pi3_register_tokens(self):
        module = _load_module()
        meta = {
            "tokens_per_view": 9,
            "patch_start_idx": 5,
            "patch_grid_width": 2,
            "num_views": 1,
            "camera_tokens_per_view": 0,
            "register_tokens_per_view": 5,
        }

        first_special = module._format_token_descriptor(meta, 0)
        first_patch = module._format_token_descriptor(meta, 5)

        self.assertEqual(first_special["kind"], "special")
        self.assertEqual(first_special["name"], "register[0]")
        self.assertEqual(first_patch["kind"], "patch")
        self.assertEqual(first_patch["row"], 0)
        self.assertEqual(first_patch["col"], 0)

    def test_load_bundle_registry_skips_non_bundle_summaries(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "benchmark").mkdir()
            (root / "benchmark" / "summary.json").write_text(json.dumps({"num_scenes_kept": 1}), encoding="utf-8")

            bundle_dir = root / "scene0707_00__anchor0066__noise__pt44"
            (bundle_dir / "layers").mkdir(parents=True)
            np.save(bundle_dir / "images_uint8.npy", np.zeros((10, 8, 8, 3), dtype=np.uint8))
            (bundle_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "num_views": 10,
                        "patch_grid_height": 2,
                        "patch_grid_width": 2,
                        "patch_start_idx": 5,
                        "captured_layers": [],
                        "layer_modes": {},
                        "layer_num_heads": {},
                        "layer_indexer_num_heads": {},
                        "layer_has_indexer_scores": {},
                        "bundle_meta": {
                            "bundle_id": "scene0707_00__anchor0066__noise__pt44",
                            "model_id": "pt44",
                            "model_name": "Indexer VGGT PT44",
                            "sample_id": "scene0707_00__anchor0066__noise",
                            "sample_name": "scene0707_00__anchor0066__noise",
                        },
                    }
                ),
                encoding="utf-8",
            )

            entries, default_bundle_id = module._load_bundle_registry("", str(root))

            self.assertEqual(list(entries.keys()), ["scene0707_00__anchor0066__noise__pt44"])
            self.assertEqual(default_bundle_id, "scene0707_00__anchor0066__noise__pt44")

    def test_frontend_registry_can_filter_pi3_only_bundles(self):
        module = _load_module()
        entries = {
            "scene01__official": {
                "bundle_id": "scene01__official",
                "bundle_label": "Scene01 | Official",
                "model_id": "official",
                "model_name": "Official VGGT",
                "sample_id": "scene01",
                "sample_name": "Scene01",
                "sample_scene": "scene-01",
                "sample_stride": 1,
                "sample_views": 12,
                "sample_note": "same input",
                "num_views": 12,
                "grid_h": 21,
                "grid_w": 37,
                "captured_layers": [0, 1],
                "layer_modes": {"00": "dense_qk", "01": "dense_qk"},
                "layer_num_heads": {"00": 16, "01": 16},
                "layer_indexer_num_heads": {"00": 0, "01": 0},
                "layer_has_indexer_scores": {"00": False, "01": False},
            },
            "scene01__pi3": {
                "bundle_id": "scene01__pi3",
                "bundle_label": "Scene01 | PI3",
                "model_id": "pi3",
                "model_name": "PI3",
                "sample_id": "scene01",
                "sample_name": "Scene01",
                "sample_scene": "scene-01",
                "sample_stride": 1,
                "sample_views": 12,
                "sample_note": "same input",
                "num_views": 12,
                "grid_h": 21,
                "grid_w": 37,
                "captured_layers": [1, 3, 5],
                "layer_modes": {"01": "dense_qk", "03": "dense_qk", "05": "dense_qk"},
                "layer_num_heads": {"01": 16, "03": 16, "05": 16},
                "layer_indexer_num_heads": {"01": 0, "03": 0, "05": 0},
                "layer_has_indexer_scores": {"01": False, "03": False, "05": False},
            },
        }

        registry = module._frontend_registry(entries, model_filter="pi3")

        self.assertEqual(sorted(registry["bundles"].keys()), ["scene01__pi3"])
        self.assertEqual(registry["sample_compare_bundle_ids"], {})

    def test_build_pi3_lite_html_uses_actual_pi3_layers(self):
        module = _load_module()
        registry = {
            "bundles": {
                "scene01__pi3": {
                    "bundle_id": "scene01__pi3",
                    "bundle_label": "Scene01 | PI3",
                    "model_id": "pi3",
                    "model_name": "PI3",
                    "sample_id": "scene01",
                    "sample_name": "Scene01",
                    "sample_scene": "scene-01",
                    "sample_stride": 1,
                    "sample_views": 12,
                    "sample_note": "same input",
                    "num_views": 12,
                    "grid_h": 21,
                    "grid_w": 37,
                    "camera_tokens_per_view": 0,
                    "register_tokens_per_view": 5,
                    "captured_layers": [1, 3, 5],
                    "layer_modes": {"01": "dense_qk", "03": "dense_qk", "05": "dense_qk"},
                    "layer_num_heads": {"01": 16, "03": 16, "05": 16},
                    "layer_indexer_num_heads": {"01": 0, "03": 0, "05": 0},
                    "layer_has_indexer_scores": {"01": False, "03": False, "05": False},
                }
            },
            "sample_compare_bundle_ids": {},
        }
        default_entry = dict(registry["bundles"]["scene01__pi3"])
        default_query = np.zeros((32, 32, 3), dtype=np.uint8)
        default_panel = np.zeros((32, 32, 3), dtype=np.uint8)

        html = module._build_pi3_lite_html(
            registry=registry,
            default_bundle_id="scene01__pi3",
            default_entry=default_entry,
            default_query=default_query,
            default_panel=default_panel,
            default_stats="ok",
        )

        self.assertIn("PI3 Layer", html)
        self.assertIn("Decoder Depth", html)
        self.assertIn("Visualized Global Layers", html)
        self.assertIn("No fair layer alignment", html)
        self.assertNotIn("Aligned Layer", html)
        self.assertNotIn("Fair Compare Panels", html)

    def test_topk_paper_display_uses_two_by_five_lightweight_layout(self):
        module = _load_module()
        num_views = 10
        grid_h = 2
        grid_w = 2
        patch_start_idx = 1
        tokens_per_view = patch_start_idx + grid_h * grid_w
        total_tokens = num_views * tokens_per_view
        query_token = patch_start_idx
        meta = {
            "num_views": num_views,
            "tokens_per_view": tokens_per_view,
            "patch_start_idx": patch_start_idx,
            "patch_grid_height": grid_h,
            "patch_grid_width": grid_w,
        }
        images_uint8 = np.full((num_views, 20, 40, 3), 120, dtype=np.uint8)
        q = torch.ones((1, total_tokens, 4), dtype=torch.float32)
        k = torch.ones((1, total_tokens, 4), dtype=torch.float32)
        topk_indices = torch.zeros((1, total_tokens, 8), dtype=torch.int64)
        topk_indices[0, query_token] = torch.tensor(
            [1, 2, 3, 4, 6, 7, 8, 9],
            dtype=torch.int64,
        )
        payload = {
            "q": q,
            "k": k,
            "scale": 1.0,
            "mode": "sparse_qk_topk",
            "topk_indices": topk_indices,
            "topk_scores": torch.ones((1, total_tokens, 8), dtype=torch.float32),
        }

        with mock.patch.object(module, "_load_layer_bundle", return_value=payload):
            _, panel, stats = module._build_attention_outputs(
                bundle_dir=Path("/tmp/fake"),
                meta=meta,
                images_uint8=images_uint8,
                layer_idx=31,
                head_mode="sum",
                query_view=0,
                query_row=0,
                query_col=0,
                display_mode="topk_paper",
            )

        self.assertIn("`topk_paper`", stats)
        self.assertEqual(panel.size, (1600, 450))
        panel_np = np.asarray(panel)
        self.assertGreater(panel_np[10, 10].mean(), 235.0)
        self.assertGreater(panel_np[132, 118].mean(), 105.0)
        self.assertLess(panel_np[132, 118].mean(), 170.0)


if __name__ == "__main__":
    unittest.main()
