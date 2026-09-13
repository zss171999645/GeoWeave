from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'aidi'
    / 'scripts'
    / 'baselines'
    / 'eval_mv_recon_summary_utils.py'
)


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f'Missing helper module: {MODULE_PATH}')
    spec = importlib.util.spec_from_file_location('eval_mv_recon_summary_utils', MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed loading module from {MODULE_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Pi3MvReconCoverageTests(unittest.TestCase):
    def test_full_coverage_without_skips_is_valid(self):
        module = load_module()

        result = module.compute_protocol_coverage(
            metric={'num_sequences': 22},
            skipped=[],
            seq_total=22,
            max_sequences=0,
        )

        self.assertTrue(result['valid'])
        self.assertEqual(result['expected_sequences'], 22)
        self.assertEqual(result['evaluated_sequences'], 22)
        self.assertEqual(result['num_skipped'], 0)
        self.assertEqual(result['failure_reason'], '')

    def test_skipped_sequences_make_result_invalid(self):
        module = load_module()

        result = module.compute_protocol_coverage(
            metric={'num_sequences': 20},
            skipped=[('scan48', 'infer_error:OutOfMemoryError:oom')],
            seq_total=22,
            max_sequences=0,
        )

        self.assertFalse(result['valid'])
        self.assertEqual(result['expected_sequences'], 22)
        self.assertEqual(result['evaluated_sequences'], 20)
        self.assertEqual(result['num_skipped'], 1)
        self.assertIn('skipped=1', result['failure_reason'])

    def test_sequence_gap_without_explicit_skip_is_invalid(self):
        module = load_module()

        result = module.compute_protocol_coverage(
            metric={'num_sequences': 21},
            skipped=[],
            seq_total=22,
            max_sequences=0,
        )

        self.assertFalse(result['valid'])
        self.assertIn('evaluated=21/22', result['failure_reason'])

    def test_max_sequence_cap_adjusts_expected_coverage(self):
        module = load_module()

        result = module.compute_protocol_coverage(
            metric={'num_sequences': 5},
            skipped=[],
            seq_total=22,
            max_sequences=5,
        )

        self.assertTrue(result['valid'])
        self.assertEqual(result['expected_sequences'], 5)
        self.assertEqual(result['evaluated_sequences'], 5)


if __name__ == '__main__':
    unittest.main()
