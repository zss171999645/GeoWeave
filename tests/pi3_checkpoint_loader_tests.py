import os
import unittest
from unittest import mock

from aidi.scripts.baselines import pi3_checkpoint_loader as loader


class FakePi3Model:
    def __init__(self, indexer_cfg):
        self.indexer_cfg = indexer_cfg
        self.indexer_state = {"enabled": False}
        self.calls = []

    def set_indexer_state(self, state):
        self.indexer_state = dict(state)
        self.calls.append(("set", dict(state)))

    def set_indexer_state_by_step(self, step, training=True):
        warmup_steps = int(self.indexer_cfg.get("warmup_steps", 0))
        sparse_start = int(self.indexer_cfg.get("sparse_start_step", warmup_steps))
        state = {
            "enabled": bool(self.indexer_cfg.get("enabled", False)),
            "warmup": bool(training) and step < warmup_steps,
            "sparse": bool(self.indexer_cfg.get("enable_sparse", True)) and step >= sparse_start,
            "compute_loss": bool(training),
            "topk": int(self.indexer_cfg.get("topk", 512)),
        }
        self.set_indexer_state(state)
        self.calls.append(("by_step", step, training, dict(state)))
        return state


class Pi3CheckpointLoaderTests(unittest.TestCase):
    def test_native_sparse_eval_sets_sparse_indexer_state_from_config(self):
        model = FakePi3Model(
            {
                "enabled": True,
                "enable_sparse": True,
                "warmup_steps": 100,
                "sparse_start_step": 200,
                "topk": 1024,
            }
        )

        state = loader.configure_native_pi3_eval_indexer(model, model.indexer_cfg)

        self.assertEqual(state["enabled"], True)
        self.assertEqual(state["warmup"], False)
        self.assertEqual(state["sparse"], True)
        self.assertEqual(state["compute_loss"], False)
        self.assertEqual(state["topk"], 1024)
        self.assertIn(("by_step", 200, False, state), model.calls)

    def test_native_sparse_eval_can_force_dense_for_ablation(self):
        model = FakePi3Model(
            {
                "enabled": True,
                "enable_sparse": True,
                "warmup_steps": 0,
                "sparse_start_step": 0,
                "topk": 1024,
            }
        )

        with mock.patch.dict(os.environ, {"PI3_INDEXER_EVAL_MODE": "dense"}):
            state = loader.configure_native_pi3_eval_indexer(model, model.indexer_cfg)

        self.assertEqual(state["enabled"], True)
        self.assertEqual(state["warmup"], False)
        self.assertEqual(state["sparse"], False)
        self.assertEqual(state["compute_loss"], False)


if __name__ == "__main__":
    unittest.main()
