import unittest


class _TrainableDummy:
    def __init__(self):
        self.training = True

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)


class _DummyAggregator:
    def __init__(self):
        self.states = []

    def set_indexer_state(self, state):
        self.states.append(dict(state))


class _DummyVggtWrapper(_TrainableDummy):
    def __init__(self):
        super().__init__()
        self.vggt = type("DummyVggt", (), {"aggregator": _DummyAggregator()})()
        self.requested_steps = []

    def _indexer_state_from_step(self, step):
        self.requested_steps.append(int(step))
        return {"family": "vggt", "step": int(step)}


class _DummyPi3(_TrainableDummy):
    def __init__(self):
        super().__init__()
        self.encoder = object()
        self.decoder = object()
        self.point_head = object()
        self.camera_head = object()
        self.calls = []

    def set_indexer_state_by_step(self, step, training=True):
        self.calls.append((int(step), bool(training)))
        return {"family": "pi3", "step": int(step), "training": bool(training)}


class _ModuleWrapper:
    def __init__(self, module):
        self.module = module


class Core4ModelAdapterTests(unittest.TestCase):
    def test_prepare_vggt_like_model_sets_eval_state_and_restores_training_state(self):
        from aidi.utils.core4_model_adapter import (
            prepare_live_model_for_core4_eval,
            restore_live_model_after_core4_eval,
        )

        model = _DummyVggtWrapper()
        model.train()

        prepared = prepare_live_model_for_core4_eval(_ModuleWrapper(model), global_step=7)

        self.assertIs(prepared.live_model, model)
        self.assertIs(prepared.inference_model, model.vggt)
        self.assertEqual(prepared.model_family, "vggt")
        self.assertTrue(prepared.was_training)
        self.assertFalse(model.training)
        self.assertEqual(model.vggt.aggregator.states[-1], {"family": "vggt", "step": 7})

        restore_live_model_after_core4_eval(prepared, global_step=8)

        self.assertTrue(model.training)
        self.assertEqual(model.vggt.aggregator.states[-1], {"family": "vggt", "step": 8})

    def test_prepare_pi3_like_model_sets_eval_state_and_restores_training_state(self):
        from aidi.utils.core4_model_adapter import (
            prepare_live_model_for_core4_eval,
            restore_live_model_after_core4_eval,
        )

        model = _DummyPi3()
        model.train()

        prepared = prepare_live_model_for_core4_eval(_ModuleWrapper(model), global_step=11)

        self.assertIs(prepared.live_model, model)
        self.assertIs(prepared.inference_model, model)
        self.assertEqual(prepared.model_family, "pi3")
        self.assertTrue(prepared.was_training)
        self.assertFalse(model.training)
        self.assertEqual(model.calls[-1], (11, False))

        restore_live_model_after_core4_eval(prepared, global_step=12)

        self.assertTrue(model.training)
        self.assertEqual(model.calls[-1], (12, True))


if __name__ == "__main__":
    unittest.main()
