from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "aidi/scripts/pi3/submit_pi3_business_lr_camemb_16gpu_matrix.sh"


class Pi3LrCameraEmbeddingMatrixTests(unittest.TestCase):
    def test_matrix_has_expected_six_variants(self):
        source = SCRIPT.read_text()
        for variant in [
            "ctrl_lr2e5_w2c10|2e-5|4e-6|1|1|vggt_w2c_quat10",
            "lr4e5_w2c10|4e-5|8e-6|1|1|vggt_w2c_quat10",
            "lr8e5_w2c10|8e-5|1.6e-5|1|1|vggt_w2c_quat10",
            "lr1e4_w2c10|1e-4|2e-5|1|1|vggt_w2c_quat10",
            "camc2w10_lr4e5|4e-5|8e-6|1|1|c2w_quat10",
            "camoff_lr4e5|4e-5|8e-6|0|0|vggt_w2c_quat10",
        ]:
            self.assertIn(variant, source)

    def test_matrix_uses_16_gpu_langfang_and_vggt_style_val(self):
        source = SCRIPT.read_text()
        for expected in [
            "CLUSTER=${CLUSTER:-project-5090-4dlabel-perception-v2-acloud-langfang}",
            "NUM_NODES=${NUM_NODES:-2}",
            "GPU_PER_NODE=${GPU_PER_NODE:-8}",
            "NUM_PROCESSES=${NUM_PROCESSES:-16}",
            "TEST_IMAGE_NUM_RANGE=[36,36]",
            "PI3_VGGT_VAL_METRICS=1",
            "TRAIN_SCHEDULER_TYPE=WarmupCosineLR",
            'local job_name="${run_name}"',
            "PI3_POSE_PRIOR_FORMAT=\"${pose_prior_format}\"",
            "TRAIN_GRAD_ACCUM_STEPS=1",
        ]:
            self.assertIn(expected, source)
        self.assertNotIn('local job_name="5090x16_${run_name}"', source)


if __name__ == "__main__":
    unittest.main()
