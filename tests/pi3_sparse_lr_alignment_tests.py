from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aidi"
    / "third_party"
    / "pi3_training"
    / "trainers"
    / "pi3_trainer.py"
)


def load_pi3_trainer_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"Missing Pi3 trainer module: {MODULE_PATH}")

    saved = {}

    def install(name: str, module: types.ModuleType) -> None:
        saved[name] = sys.modules.get(name)
        sys.modules[name] = module

    trainers_pkg = types.ModuleType("trainers")
    base_trainer_module = types.ModuleType("trainers.base_trainer_accelerate")

    class BaseTrainer:
        def __init__(self, *args, **kwargs):
            pass

    base_trainer_module.BaseTrainer = BaseTrainer
    trainers_pkg.base_trainer_accelerate = base_trainer_module

    datasets_pkg = types.ModuleType("datasets")
    datasets_base_pkg = types.ModuleType("datasets.base")
    datasets_base_module = types.ModuleType("datasets.base.base_dataset")
    datasets_base_module.sample_resolutions = lambda *args, **kwargs: []
    datasets_base_pkg.base_dataset = datasets_base_module
    datasets_pkg.base = datasets_base_pkg

    hydra_module = types.ModuleType("hydra")
    hydra_module.utils = SimpleNamespace(instantiate=lambda cfg: cfg)

    torch_module = types.ModuleType("torch")

    easydict_module = types.ModuleType("easydict")

    class EasyDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__

    easydict_module.EasyDict = EasyDict

    install("trainers", trainers_pkg)
    install("trainers.base_trainer_accelerate", base_trainer_module)
    install("datasets", datasets_pkg)
    install("datasets.base", datasets_base_pkg)
    install("datasets.base.base_dataset", datasets_base_module)
    install("hydra", hydra_module)
    install("torch", torch_module)
    install("easydict", easydict_module)

    spec = importlib.util.spec_from_file_location("pi3_trainer_under_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading trainer from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


class Pi3SparseLrAlignmentTests(unittest.TestCase):
    def test_indexer_params_are_split_out_of_other_params(self):
        module = load_pi3_trainer_module()
        named_params = [
            ("encoder.block.weight", object()),
            ("decoder.0.attn.indexer.q_proj.weight", object()),
            ("decoder.0.mlp.weight", object()),
        ]

        grouped = module._split_pi3_optimizer_named_parameters(named_params)

        self.assertEqual([name for name, _ in grouped["encoder"]], ["encoder.block.weight"])
        self.assertEqual([name for name, _ in grouped["indexer"]], ["decoder.0.attn.indexer.q_proj.weight"])
        self.assertEqual([name for name, _ in grouped["other"]], ["decoder.0.mlp.weight"])

    def _build_trainer(self, indexer_cfg: dict):
        module = load_pi3_trainer_module()
        trainer = module.Pi3Trainer.__new__(module.Pi3Trainer)
        trainer.cfg = SimpleNamespace(model=SimpleNamespace(indexer_cfg=indexer_cfg))
        trainer.optimizer = SimpleNamespace(
            param_groups=[
                {"lr": 3.0e-4, "is_indexer": False},
                {"lr": 1.0e-4, "is_indexer": True},
            ]
        )
        return trainer

    def test_sparse_lr_scheduler_mode_keeps_scheduler_lrs(self):
        trainer = self._build_trainer(
            {
                "sparse_lr": 1.0e-5,
                "sparse_lr_decay": "scheduler",
            }
        )

        trainer._apply_indexer_stage_lr({"sparse": True})

        self.assertEqual(
            [group["lr"] for group in trainer.optimizer.param_groups],
            [3.0e-4, 1.0e-4],
        )

    def test_sparse_min_lr_floors_only_indexer_groups(self):
        trainer = self._build_trainer(
            {
                "sparse_lr": 1.0e-5,
                "sparse_lr_decay": "scheduler",
                "sparse_min_lr": 1.0e-6,
            }
        )
        trainer.optimizer.param_groups = [
            {"lr": 3.0e-4, "is_indexer": False},
            {"lr": 2.0e-7, "is_indexer": True},
            {"lr": 2.0e-6, "is_indexer": True},
        ]

        trainer._apply_indexer_stage_lr({"sparse": True})

        self.assertEqual(
            [group["lr"] for group in trainer.optimizer.param_groups],
            [3.0e-4, 1.0e-6, 2.0e-6],
        )

    def test_sparse_lr_constant_mode_overrides_only_indexer_groups(self):
        trainer = self._build_trainer(
            {
                "sparse_lr": 1.0e-5,
                "sparse_lr_decay": "constant",
            }
        )

        trainer._apply_indexer_stage_lr({"sparse": True})

        self.assertEqual(
            [group["lr"] for group in trainer.optimizer.param_groups],
            [3.0e-4, 1.0e-5],
        )

    def test_sparse_lr_cap_mode_clamps_only_higher_indexer_groups(self):
        trainer = self._build_trainer(
            {
                "sparse_lr": 1.0e-5,
                "sparse_lr_decay": "cap",
            }
        )
        trainer.optimizer.param_groups = [
            {"lr": 3.0e-4, "is_indexer": False},
            {"lr": 2.0e-5, "is_indexer": True},
            {"lr": 5.0e-6, "is_indexer": True},
        ]

        trainer._apply_indexer_stage_lr({"sparse": True})

        self.assertEqual(
            [group["lr"] for group in trainer.optimizer.param_groups],
            [3.0e-4, 1.0e-5, 5.0e-6],
        )

    def test_sparse_lr_noops_when_no_indexer_group_exists(self):
        trainer = self._build_trainer(
            {
                "sparse_lr": 1.0e-5,
                "sparse_lr_decay": "constant",
            }
        )
        trainer.optimizer.param_groups = [
            {"lr": 3.0e-4, "is_indexer": False},
            {"lr": 1.0e-4, "is_indexer": False},
        ]

        trainer._apply_indexer_stage_lr({"sparse": True})

        self.assertEqual(
            [group["lr"] for group in trainer.optimizer.param_groups],
            [3.0e-4, 1.0e-4],
        )

    def test_warmup_only_indexer_loss_skips_pi3_loss(self):
        module = load_pi3_trainer_module()
        trainer = module.Pi3Trainer.__new__(module.Pi3Trainer)

        class FakeLoss:
            def detach(self):
                return self

        class FakeModel:
            indexer_loss = FakeLoss()
            indexer_cfg = {"warmup_only_indexer_loss": True}

            def get_indexer_state(self):
                return {"warmup": True, "sparse": False}

        def fail_if_called(*args, **kwargs):
            raise AssertionError("Pi3Loss must not run during KL indexer warm-up")

        trainer.train_loss = fail_if_called
        trainer.test_loss = fail_if_called
        trainer._unwrap_model = lambda: FakeModel()

        output = trainer.calculate_loss([{"unused": True}, [], True], [], mode="train")

        self.assertIs(output.loss, FakeModel.indexer_loss)
        self.assertIn("indexer_loss", output)

    def test_warmup_config_freezes_non_indexer_params_by_default(self):
        cfg_path = (
            Path(__file__).resolve().parents[1]
            / "aidi"
            / "third_party"
            / "pi3_training"
            / "configs"
            / "train"
            / "train_pi3_lowres_indexer_warmup.yaml"
        )
        source = cfg_path.read_text()

        self.assertIn("warmup_only_indexer_train: true", source)


if __name__ == "__main__":
    unittest.main()
