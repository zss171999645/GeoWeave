import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "aidi"
    / "scripts"
    / "vggt"
    / "eval_official_vggt_sparse_datasets_plus.sh"
)


class EvalOfficialVggtSparseDatasetsPlusScriptTests(unittest.TestCase):
    def test_easyvolcap_main_uses_short_config_flag(self):
        text = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn('-m easyvolcap.scripts.main --type test -c "${TMP_CFG}"', text)
        self.assertNotIn('--type test --config "${TMP_CFG}"', text)


if __name__ == "__main__":
    unittest.main()
