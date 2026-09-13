from .indexer_kl_fused import sparse_indexer_kl_loss
from .sparse_topk_indexer import sparse_topk_indexer_func

# Optional inference-only entrypoint. Keep the symbol defined so
# `from easyvolcap.utils.custom_indexer import ..., sparse_topk_indexer_inference_func`
# does not fail and mask the regular sparse top-k kernel.
sparse_topk_indexer_inference_func = None

__all__ = [
    "sparse_topk_indexer_func",
    "sparse_topk_indexer_inference_func",
    "sparse_indexer_kl_loss",
]
