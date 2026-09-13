import torch

from easyvolcap.official_vggt.layers.indexer import LightningIndexer


def test_topk_support_loss_backward():
    torch.manual_seed(7)
    indexer = LightningIndexer(dim=16, n_heads=2, head_dim=4, score_dtype="float32", score_key_chunk_size=3)
    x = torch.randn(2, 7, 16, requires_grad=True)
    p = torch.softmax(torch.randn(2, 7, 7), dim=-1).detach()

    loss = indexer.compute_topk_support_loss(
        p=p,
        support_topk=3,
        x=x,
        support_chunk_size=2,
        query_chunk_size=3,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert indexer.q_proj.weight.grad is not None
    assert torch.isfinite(indexer.q_proj.weight.grad).all()
    assert indexer.q_proj.weight.grad.abs().sum() > 0
