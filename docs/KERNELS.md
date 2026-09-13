# GeoWeave acceleration kernels

The public inference entry point runs the same native model classes used for
training. On CUDA, their indexers call the custom implementations below.
`--backend torch` keeps top-k sparse attention using a PyTorch reference path;
it does not measure the CUDA acceleration and is not bitwise equivalent across
precisions/backends.

| Source | Role |
| --- | --- |
| `easyvolcap/utils/custom_indexer/sparse_topk_indexer.py` | Streaming index scores and top-k selection (Triton) |
| `easyvolcap/utils/custom_indexer/indexer_kl_fused.py` | Fused indexer KL training kernels (Triton) |
| `easyvolcap/utils/custom_indexer/streaming_kl_autograd.py` | Streaming KL autograd integration |
| `easyvolcap/utils/custom_indexer/topk_support_autograd.py` | Training support over selected dependencies |
| `easyvolcap/utils/custom_indexer/csrc/topk_exact*` | Optional exact selection of 512 candidates |
| `easyvolcap/utils/custom_indexer/csrc/score_topk_fused*` | Optional fused score and top-k selection |
| `easyvolcap/utils/custom_indexer/csrc/merge_two_topk*` | Merge partial top-k lists |
| `easyvolcap/utils/custom_flash_attn/sparse_index_flash_attn.py` | Sparse attention forward and backward (Triton) |
| `easyvolcap/utils/custom_flash_attn/csrc/sparse_bwd_grouped*` | Optional grouped probability backward (CUDA) |

## Installation and checks

Install the inference and kernel requirements from the root README. CUDA extension
compilation needs a CUDA toolkit and a host C++ compiler; a driver alone is not
enough. Set `CUDA_HOME` if the toolkit is installed outside the default location.
Ninja and a writable PyTorch extension cache are required. The C++/CUDA sources
are included as package data as well as in the source checkout.

Run from the repository root on a CUDA GPU:

```bash
python -m geoweave.kernels
python -m geoweave.kernels --build-all
```

The first command executes the sparse attention forward and backward kernels on
real tensors and compares them against a gathered PyTorch attention reference.
It exits with an error on missing CUDA/Triton or a numerical mismatch. The second
also compiles all four optional C++/CUDA extensions and compares the 512-way exact
top-k extension against `torch.topk`. Compilation is not a numerical verification
of every optional kernel; additional historical tests are retained under `tests/`.

## Preserve the paper configuration

Both published presets use **top-k 1024**. The optional fixed-512 fused kernels
must not be forced into these runs: changing top-k changes the method. Leave the
`VGGT_*` / `PI3_*` environment overrides unset for the default path. These variables
are recorded in inference metadata when present.

Historical tuning flags and measurements are documented in
`GEOWEAVE_CUDA_OPERATOR_OPT_20260704.md`. They are not a promise of the same speed
on a different GPU, PyTorch/Triton version, precision, or input shape. CPU inference
checks do not validate CUDA kernel execution.
