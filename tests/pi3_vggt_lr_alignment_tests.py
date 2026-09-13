from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_PATH = ROOT / "aidi/third_party/pi3_training/utils/scheduler.py"
LAUNCHER_PATH = ROOT / "aidi/scripts/pi3/train_pi3_official.sh"
WRAPPER_PATH = ROOT / "aidi/scripts/pi3/submit_pi3_5090_training.sh"


def load_warmup_cosine_helper():
    source = SCHEDULER_PATH.read_text(encoding="utf-8")
    module_ast = ast.parse(source)
    helper_node = next(
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name == "_warmup_cosine_lr"
    )
    code = compile(ast.Module(body=[helper_node], type_ignores=[]), str(SCHEDULER_PATH), "exec")
    namespace = {"math": math}
    exec(code, namespace)
    return namespace["_warmup_cosine_lr"]


class Pi3VggtLrAlignmentTests(unittest.TestCase):
    def test_warmup_cosine_lr_matches_vggt_standard_160k_curve(self):
        warmup_cosine_lr = load_warmup_cosine_helper()

        def lr_at(step: int, base_lr: float = 2.0e-5) -> float:
            return warmup_cosine_lr(
                base_lr=base_lr,
                step=step,
                decay_iter=160000,
                warmup_iters=10000,
                warmup_start_lr=1.0e-8,
                min_lr=1.0e-8,
            )

        self.assertAlmostEqual(lr_at(50), 1.0995e-7, places=12)
        self.assertAlmostEqual(lr_at(10000), 2.0e-5, places=12)
        self.assertAlmostEqual(lr_at(80000), 1.1045284632676535e-5, places=12)
        self.assertAlmostEqual(lr_at(160000), 1.0e-8, places=12)
        self.assertAlmostEqual(lr_at(10000, base_lr=4.0e-6), 4.0e-6, places=12)

    def test_pi3_launcher_exposes_vggt_style_scheduler_overrides(self):
        launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
        wrapper = WRAPPER_PATH.read_text(encoding="utf-8")

        for expected in [
            "append_override_from_env TRAIN_SCHEDULER_TYPE train.lr_scheduler.type",
            'append_override_from_env TRAIN_SCHEDULER_DECAY_ITER "++train.lr_scheduler.decay_iter"',
            'append_override_from_env TRAIN_SCHEDULER_WARMUP_ITERS "++train.lr_scheduler.warmup_iters"',
            'append_override_from_env TRAIN_SCHEDULER_WARMUP_START_LR "++train.lr_scheduler.warmup_start_lr"',
            'append_override_from_env TRAIN_SCHEDULER_MIN_LR "++train.lr_scheduler.min_lr"',
        ]:
            self.assertIn(expected, launcher)

        for expected in [
            "export TRAIN_SCHEDULER_TYPE TRAIN_SCHEDULER_DECAY_ITER TRAIN_SCHEDULER_WARMUP_ITERS",
            "TRAIN_SCHEDULER_TYPE=${TRAIN_SCHEDULER_TYPE:-<default>}",
        ]:
            self.assertIn(expected, wrapper)


if __name__ == "__main__":
    unittest.main()
