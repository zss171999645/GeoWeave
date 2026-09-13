from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PI3_MODEL_PATH = (
    REPO_ROOT
    / "aidi"
    / "third_party"
    / "pi3_training"
    / "pi3"
    / "models"
    / "pi3_training.py"
)
PI3_MODEL_CFG_PATH = (
    REPO_ROOT
    / "aidi"
    / "third_party"
    / "pi3_training"
    / "configs"
    / "model"
    / "pi3.yaml"
)
TRAIN_SCRIPT_PATH = REPO_ROOT / "aidi" / "scripts" / "pi3" / "train_pi3_official.sh"


class Pi3IndexerInitCheckpointTests(unittest.TestCase):
    def test_pi3_constructor_declares_indexer_init_checkpoint(self):
        tree = ast.parse(PI3_MODEL_PATH.read_text())
        pi3_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Pi3"
        )
        init_func = next(
            node
            for node in pi3_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        args = [arg.arg for arg in init_func.args.args]

        self.assertIn("indexer_init_ckpt", args)

    def test_indexer_init_loader_handles_evc_pi3_prefix(self):
        source = PI3_MODEL_PATH.read_text()

        self.assertIn("pi3.decoder.", source)
        self.assertIn("decoder.", source)
        self.assertIn("indexer_init_ckpt", PI3_MODEL_CFG_PATH.read_text())
        self.assertIn("model.indexer_init_ckpt=", TRAIN_SCRIPT_PATH.read_text())
        self.assertNotIn("++model.indexer_init_ckpt=", TRAIN_SCRIPT_PATH.read_text())

    def test_warmup_submit_script_supports_explicit_warmup_lr(self):
        script_path = REPO_ROOT / "aidi" / "scripts" / "pi3" / "submit_pi3_5090_warmup.sh"
        source = script_path.read_text()

        self.assertIn("WARMUP_LR=", source)
        self.assertIn("model.indexer_cfg.warmup_lr=${WARMUP_LR}", source)
        self.assertIn("WARMUP_LR=${WARMUP_LR:-<config-default>}", source)

    def test_warmup_dataset_cache_overrides_are_idempotent(self):
        script_path = REPO_ROOT / "aidi" / "scripts" / "pi3" / "submit_pi3_5090_warmup.sh"
        source = script_path.read_text()

        self.assertIn('"++train_dataset.${dataset_key}.use_index_cache=${PI3_USE_INDEX_CACHE}"', source)
        self.assertIn('"++test_dataset.${dataset_key}.index_cache_dir=${PI3_DATASET_CACHE_DIR}"', source)
        self.assertNotIn('"+train_dataset.${dataset_key}.use_index_cache=${PI3_USE_INDEX_CACHE}"', source)
        self.assertNotIn('"+test_dataset.${dataset_key}.index_cache_dir=${PI3_DATASET_CACHE_DIR}"', source)

    def test_warmup_streaming_kl_checkpoint_requires_static_graph(self):
        script_path = REPO_ROOT / "aidi" / "scripts" / "pi3" / "submit_pi3_5090_warmup.sh"
        source = script_path.read_text()

        self.assertIn("PI3_STATIC_GRAPH=${PI3_STATIC_GRAPH:-1}", source)
        self.assertIn("streaming KL autograd with decoder activation checkpointing requires PI3_STATIC_GRAPH=1", source)
        self.assertIn("PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT", source)


if __name__ == "__main__":
    unittest.main()
