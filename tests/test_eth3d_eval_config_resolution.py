import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aidi.scripts.baselines.eval_config_utils import load_resolved_config


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestEth3DEvalConfigResolution(unittest.TestCase):
    def test_missing_absolute_legacy_base_falls_back_to_repo_relative_path(self):
        with TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "legacy_eval.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "_base_: /home/feng01.zhou/workspace/meshx_5090/configs/exps/vggt/vggt_official_finetune_5090.yaml",
                        "",
                        "model_cfg:",
                        "  vggt_cfg:",
                        "    enable_point: true",
                        "    indexer_cfg:",
                        "      topk: 1024",
                        "      indexer_layers: 9-19",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = load_resolved_config(cfg_path)
            model_cfg = cfg["model_cfg"]
            self.assertEqual(model_cfg["type"], "OfficialVGGTModel")
            self.assertEqual(model_cfg["vggt_cfg"]["indexer_cfg"]["topk"], 1024)

    def test_legacy_base_yaml_is_fully_resolved(self):
        cfg = load_resolved_config(
            REPO_ROOT / "aidi/configs/vggt/finetune_5090_sparse_topk1024_l9_19_p34_record.yaml"
        )

        model_cfg = cfg["model_cfg"]
        self.assertEqual(model_cfg["type"], "OfficialVGGTModel")
        self.assertIn("agg_ckpt", model_cfg)
        self.assertIn("cam_ckpt", model_cfg)
        self.assertIn("xyz_ckpt", model_cfg)
        self.assertEqual(model_cfg["vggt_cfg"]["indexer_cfg"]["topk"], 1024)

    def test_recent_sparse_short_configs_are_fully_resolved(self):
        cfg_specs = [
            ("aidi/configs/vggt/finetune_5090_sparse_topk512_l9_19_p34_record.yaml", 512),
            ("aidi/configs/vggt/finetune_5090_sparse_topk1024_layersall_p34_record.yaml", 1024),
            ("aidi/configs/vggt/finetune_5090_sparse_topk1024_l9_19_softviewbias_trainkernel_lossw02_p34_record.yaml", 1024),
        ]
        for rel_path, topk in cfg_specs:
            with self.subTest(config=rel_path):
                cfg = load_resolved_config(REPO_ROOT / rel_path)
                model_cfg = cfg["model_cfg"]
                self.assertEqual(model_cfg["type"], "OfficialVGGTModel")
                self.assertIn("agg_ckpt", model_cfg)
                self.assertEqual(model_cfg["vggt_cfg"]["indexer_cfg"]["topk"], topk)

    def test_native_official_config_still_loads(self):
        cfg = load_resolved_config(REPO_ROOT / "configs/exps/vggt/vggt_official_eval_paper.yaml")
        self.assertIn("model_cfg", cfg)
        self.assertIn("type", cfg["model_cfg"])


if __name__ == "__main__":
    unittest.main()
