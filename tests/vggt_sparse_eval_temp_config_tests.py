import os
import tempfile
import unittest

import yaml

from easyvolcap.engine.config import Config

from aidi.scripts.vggt.sparse_eval_temp_config import build_temp_eval_payload, resolve_base_config_chain


class VGGTSparseEvalTempConfigTests(unittest.TestCase):
    def test_resolve_base_config_chain_keeps_overlay_order(self):
        chain = resolve_base_config_chain(
            "aidi/configs/vggt/finetune_5090_sparse_topk1024_l9_19_softviewbias_trainkernel_lossw02_p34_record.yaml"
        )
        self.assertEqual(
            chain,
            [
                "aidi/configs/vggt/finetune_5090_sparse_topk2048_l9_19_p34_record.yaml",
                "aidi/configs/vggt/finetune_5090_sparse_topk1024_l9_19_p34_record.yaml",
                "aidi/configs/vggt/finetune_5090_sparse_topk1024_l9_19_softviewbias_trainkernel_lossw02_p34_record.yaml",
            ],
        )

    def test_build_temp_eval_payload_preserves_model_type_from_base_chain(self):
        env = {
            "BASE_CONFIG": "aidi/configs/vggt/finetune_5090_sparse_topk1024_l9_19_softviewbias_trainkernel_lossw02_p34_record.yaml",
            "EXP_NAME": "tmp/eval",
            "CKPT_PATH": "/tmp/fake.pt",
            "DPT_CKPT": "/tmp/depth.pt",
            "AGG_CKPT": "/tmp/aggregator.pt",
            "CAM_CKPT": "/tmp/camera.pt",
            "XYZ_CKPT": "/tmp/point.pt",
            "TRA_CKPT": "/tmp/track.pt",
            "METRICS_FILE": "metrics_eth3d.json",
        }
        payload = build_temp_eval_payload("configs/exps/vggt/evaluation/eth3d.yaml", env)

        merged = {}
        for cfg_path in payload["configs"]:
            merged = Config._merge_a_into_b(Config.fromfile(cfg_path)._cfg_dict, merged)
        final_cfg = Config._merge_a_into_b(
            {k: v for k, v in payload.items() if k != "configs"},
            merged,
        )

        self.assertEqual(final_cfg["model_cfg"]["type"], "OfficialVGGTModel")
        self.assertEqual(final_cfg["model_cfg"]["vggt_cfg"]["indexer_cfg"]["topk"], 1024)
        self.assertTrue(final_cfg["model_cfg"]["vggt_cfg"]["indexer_cfg"]["soft_view_bias_enabled"])


if __name__ == "__main__":
    unittest.main()
