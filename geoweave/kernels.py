"""Compile and numerically check the original GeoWeave CUDA/Triton kernels."""
import argparse
import importlib


def check_extensions(build_all=False):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Kernel verification requires an NVIDIA CUDA GPU')
    importlib.import_module('easyvolcap.utils.custom_flash_attn.sparse_index_flash_attn')
    importlib.import_module('easyvolcap.utils.custom_indexer.sparse_topk_indexer')
    if build_all:
        for name in ('custom_indexer.topk_exact_cuda', 'custom_indexer.merge_two_topk_cuda',
                     'custom_indexer.score_topk_fused_cuda', 'custom_flash_attn.sparse_bwd_grouped_cuda'):
            module = importlib.import_module('easyvolcap.utils.' + name)
            if module._load_extension() is None:
                raise RuntimeError(f'{name} failed to build: {module._EXT_LOAD_ERROR}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-all', action='store_true', help='Also compile the optional C++/CUDA extensions')
    args = parser.parse_args()
    check_extensions(args.build_all)
    import torch
    from easyvolcap.utils.custom_flash_attn.sparse_index_flash_attn import sparse_index_flash_attn_func
    torch.manual_seed(123)
    q, k, v = [torch.randn(1, 128, 2, 64, device='cuda', dtype=torch.float16, requires_grad=True)
               for _ in range(3)]
    positions = torch.rand(1, 128, 128, device='cuda').argsort(-1)[..., :64].to(torch.int32).contiguous()
    result, _ = sparse_index_flash_attn_func(q, k, v, positions)
    q2, k2, v2 = [x.detach().float().requires_grad_() for x in (q, k, v)]
    keys, values = k2[0][positions[0].long()], v2[0][positions[0].long()]
    scores = torch.einsum('qhd,qkhd->qhk', q2[0], keys) / 8
    reference = torch.einsum('qhk,qkhd->qhd', scores.softmax(-1), values)[None]
    torch.testing.assert_close(result.float(), reference, atol=3e-3, rtol=3e-3)
    grad = torch.randn_like(result)
    result.backward(grad)
    reference.backward(grad.float())
    for actual, expected in zip((q, k, v), (q2, k2, v2)):
        torch.testing.assert_close(actual.grad.float(), expected.grad, atol=1e-2, rtol=1e-2)
    print('Sparse attention: CUDA/Triton forward and backward match the PyTorch reference.')
    if args.build_all:
        from easyvolcap.utils.custom_indexer.topk_exact_cuda import topk_exact_select
        scores = torch.randn(1, 4, 2048, device='cuda', dtype=torch.float32)
        actual, indices = topk_exact_select(scores)
        expected = scores.topk(512, dim=-1).values
        torch.testing.assert_close(actual.sort(-1).values, expected.sort(-1).values)
        torch.testing.assert_close(actual, scores.gather(-1, indices))
        print('Optional C++/CUDA extensions compiled; exact top-k matches torch.topk.')


if __name__ == '__main__':
    main()
