import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - local static test env may not include torch.
    torch = None


ROOT = Path(__file__).resolve().parents[1]
PI3_ROOT = ROOT / "aidi" / "third_party" / "pi3_training"
if str(PI3_ROOT) not in sys.path:
    sys.path.insert(0, str(PI3_ROOT))

if torch is not None:
    from pi3.models.loss import PointLoss  # noqa: E402


class Pi3NormalLossChunkTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_view_chunked_normal_loss_matches_full_loss(self):
        torch.manual_seed(7)
        points = torch.randn(2, 5, 18, 20, 3)
        gt_points = points + 0.01 * torch.randn_like(points)
        gt_points[..., 2] = gt_points[..., 2].abs() + 1.0
        mask = torch.rand(2, 5, 18, 20) > 0.2

        full_loss = PointLoss(normal_loss_view_chunk_size=0).noraml_loss(points, gt_points, mask)
        chunked_loss = PointLoss(normal_loss_view_chunk_size=2).noraml_loss(points, gt_points, mask)

        self.assertTrue(torch.allclose(full_loss, chunked_loss, atol=1e-8, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
