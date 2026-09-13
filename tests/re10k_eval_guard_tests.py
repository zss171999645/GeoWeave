import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Re10kEvalGuardTests(unittest.TestCase):
    def test_clusterfix_root_is_allowed(self):
        from aidi.scripts.vggt import re10k_eval_guard as guard

        self.assertTrue(
            guard.is_fair_re10k_root(
                "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/datasets/re10k/processed_pose1800_clusterfix/test"
            )
        )

    def test_compat_root_is_allowed(self):
        from aidi.scripts.vggt import re10k_eval_guard as guard

        self.assertTrue(
            guard.is_fair_re10k_root(
                "/home/feng01.zhou/workspace/meshx_5090/tmp/re10k_compatmono_old5090_full/test"
            )
        )

    def test_legacy_root_is_rejected(self):
        from aidi.scripts.vggt import re10k_eval_guard as guard

        self.assertFalse(
            guard.is_fair_re10k_root(
                "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/datasets/re10k/processed_pose1800/test"
            )
        )

    def test_topk1024_family_requires_topk_override(self):
        from aidi.scripts.vggt import re10k_eval_guard as guard

        issues = guard.validate_re10k_eval_request(
            exp_name="vggt/official/re10k_old5090_topk1024_59_recheckhost_20260402",
            ckpt_path="/tmp/resumept30_topk1024_pt59.pt",
            base_config="aidi/configs/vggt/finetune_5090_sparse_topk2048_l9_19_p34_record.yaml",
            extra_overrides="runner_cfg.pretrained_model=null",
            re10k_root="/tmp/re10k_compatmono_old5090_full/test",
            official_baseline=False,
        )

        self.assertTrue(any("topk=1024" in issue for issue in issues))

    def test_l917_family_requires_layer_override(self):
        from aidi.scripts.vggt import re10k_eval_guard as guard

        issues = guard.validate_re10k_eval_request(
            exp_name="vggt/official/re10k_old5090_l917_39_recheckhost_20260402",
            ckpt_path="/tmp/resumept30_l917_pt39.pt",
            base_config="aidi/configs/vggt/finetune_5090_sparse_topk2048_l9_19_p34_record.yaml",
            extra_overrides="runner_cfg.pretrained_model=null",
            re10k_root="/tmp/re10k_compatmono_old5090_full/test",
            official_baseline=False,
        )

        self.assertTrue(any("indexer_layers=9-17" in issue for issue in issues))

    def test_topk512_family_requires_topk512_override(self):
        from aidi.scripts.vggt import re10k_eval_guard as guard

        issues = guard.validate_re10k_eval_request(
            exp_name="vggt/official/re10k_4090_topk512_l919_lossw05_ep80_pt14_20260403",
            ckpt_path="/tmp/topk512_l919_lossw05_ep80_pt14.pt",
            base_config="aidi/configs/vggt/finetune_5090_sparse_topk2048_l9_19_p34_record.yaml",
            extra_overrides="runner_cfg.pretrained_model=null",
            re10k_root="/tmp/re10k_compatmono_4090_full/test",
            official_baseline=False,
        )

        self.assertTrue(any("topk=512" in issue for issue in issues))

    def test_layersall_family_requires_layersall_override(self):
        from aidi.scripts.vggt import re10k_eval_guard as guard

        issues = guard.validate_re10k_eval_request(
            exp_name="vggt/official/re10k_4090_topk1024_layersall_lossw05_ep80_pt4_20260403",
            ckpt_path="/tmp/topk1024_layersall_lossw05_ep80_pt4.pt",
            base_config="aidi/configs/vggt/finetune_5090_sparse_topk2048_l9_19_p34_record.yaml",
            extra_overrides="model_cfg.vggt_cfg.indexer_cfg.topk=1024",
            re10k_root="/tmp/re10k_compatmono_4090_full/test",
            official_baseline=False,
        )

        self.assertTrue(any("indexer_layers=all" in issue for issue in issues))

    def test_topk512_lossw05_is_not_misclassified_as_topk1024(self):
        from aidi.scripts.vggt import re10k_eval_guard as guard

        issues = guard.validate_re10k_eval_request(
            exp_name="vggt/official/re10k_4090_topk512_l919_lossw05_ep80_pt14_20260403",
            ckpt_path="/tmp/topk512_l919_lossw05_ep80_pt14.pt",
            base_config="aidi/configs/vggt/finetune_5090_sparse_topk2048_l9_19_p34_record.yaml",
            extra_overrides="model_cfg.vggt_cfg.indexer_cfg.topk=512;model_cfg.vggt_cfg.indexer_cfg.indexer_layers=9-19",
            re10k_root="/tmp/re10k_compatmono_4090_full/test",
            official_baseline=False,
        )

        self.assertEqual([], issues)

    def test_unexpected_l917_override_is_rejected_for_topk2048(self):
        from aidi.scripts.vggt import re10k_eval_guard as guard

        issues = guard.validate_re10k_eval_request(
            exp_name="vggt/official/re10k_old5090_resumept30_topk2048_pt74_fixmono_20260402",
            ckpt_path="/tmp/resumept30_topk2048_pt74.pt",
            base_config="aidi/configs/vggt/finetune_5090_sparse_topk2048_l9_19_p34_record.yaml",
            extra_overrides="model_cfg.vggt_cfg.indexer_cfg.indexer_layers=9-17",
            re10k_root="/tmp/re10k_compatmono_old5090_full/test",
            official_baseline=False,
        )

        self.assertTrue(any("unexpected" in issue.lower() for issue in issues))

    def test_unexpected_topk1024_override_is_rejected_for_topk512(self):
        from aidi.scripts.vggt import re10k_eval_guard as guard

        issues = guard.validate_re10k_eval_request(
            exp_name="vggt/official/re10k_4090_topk512_l919_lossw05_ep80_pt14_20260403",
            ckpt_path="/tmp/topk512_l919_lossw05_ep80_pt14.pt",
            base_config="aidi/configs/vggt/finetune_5090_sparse_topk2048_l9_19_p34_record.yaml",
            extra_overrides="model_cfg.vggt_cfg.indexer_cfg.topk=1024;model_cfg.vggt_cfg.indexer_cfg.indexer_layers=9-19",
            re10k_root="/tmp/re10k_compatmono_4090_full/test",
            official_baseline=False,
        )

        self.assertTrue(any("Unexpected topk1024" in issue for issue in issues))

    def test_runtime_log_must_contain_loaded_network(self):
        from aidi.scripts.vggt import re10k_eval_guard as guard

        issues = guard.validate_runtime_log(
            log_text="Evaluating epoch 0, 1 / 1554\n",
            official_baseline=False,
        )

        self.assertTrue(any("Loaded network" in issue for issue in issues))

    def test_runtime_log_rejects_epoch_zero_load(self):
        from aidi.scripts.vggt import re10k_eval_guard as guard

        issues = guard.validate_runtime_log(
            log_text="Loaded network /tmp/foo.pt at epoch 0\n",
            official_baseline=False,
        )

        self.assertTrue(any("epoch 0" in issue for issue in issues))

    def test_runtime_log_accepts_positive_loaded_epoch(self):
        from aidi.scripts.vggt import re10k_eval_guard as guard

        issues = guard.validate_runtime_log(
            log_text="Loaded network /tmp/foo.pt at epoch 60\n",
            official_baseline=False,
        )

        self.assertEqual([], issues)

    def test_runtime_log_accepts_wrapped_loaded_epoch(self):
        from aidi.scripts.vggt import re10k_eval_guard as guard

        issues = guard.validate_runtime_log(
            log_text=(
                "04-02 14:38:37 Loaded network net_utils.py:460\n"
                "    /tmp/vggt_eval_cache/inputs/c\n"
                "    kpt_path_deadbeef.pt at epoch 14\n"
                "04-02 14:38:38 Evaluating epoch 15, 1 / 1554\n"
            ),
            official_baseline=False,
        )

        self.assertEqual([], issues)


if __name__ == "__main__":
    unittest.main()
