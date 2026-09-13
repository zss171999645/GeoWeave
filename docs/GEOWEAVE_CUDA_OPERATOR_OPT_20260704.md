# GeoWeave CUDA 算子优化记录 2026-07-04

## 范围

这次优化不改变 GeoWeave 的权重、`topk=1024`、稀疏层位置、head 设置、输入张量、输出 head 或评估逻辑。所有可接受结果都必须满足：同一个 synthetic input 下，baseline env 和 optimized env 输出的 `points`、`local_points`、`camera_poses` 完全一致，`max_abs=0.0`。

本轮只接受两类优化：

1. 现有 CUDA/Triton 算子的 block、merge、kernel 调度参数。
2. 固定 shape 推理下的 fullchain fastpath / CUDA graph replay。这个路径也不改计算结果，但它更像执行图调度优化，因此和纯 operator-only 参数单独列出。

## 代码改动

1. `easyvolcap/utils/custom_indexer/{topk_exact_cuda.py,merge_two_topk_cuda.py,score_topk_fused_cuda.py}` 会在当前 Python env 的 `bin/` 目录存在 `ninja` 时，把该目录补到 `PATH` 前面。
   - 原因：`torch.utils.cpp_extension.load` 需要 `ninja` 可执行文件。`/mnt/cfs/zhoufeng/env/geoweave-rebuttal/bin/ninja` 已存在，但非交互 SSH 命令的 `PATH` 没带上这个目录。
   - 这只修复 extension loading，不改 CUDA kernel 代码，不改模型数学。

2. 新增 `tools/compare_geoweave_operator_flags.py`，用于同输入比较 GeoWeave baseline env 和 optimized env：
   - 记录 CUDA event timing。
   - 比较 `points`、`local_points`、`camera_poses` 的 `max_abs` / `mean_abs`。
   - 如果候选优化产生非零差异，就不作为“保持计算逻辑不变”的优化结果。
   - 后续已把 fullchain / no-contig / fused-proj 相关 env 也加入 baseline 清理列表，避免外部环境变量污染 baseline。

3. `tools/synthetic_pi3_runtime_benchmark.py` 新增 `--skip-empty-cache-after-case`。
   - 默认行为不变。
   - 只在 CUDA graph benchmark 时使用，避免 PyTorch allocator 在 graph replay 后执行 `torch.cuda.empty_cache()` 触发 `captures_underway.empty()` 内部错误。
   - 这个开关只影响 benchmark 清理阶段，不影响 forward 计算。

## 被拒绝的候选

投稿前配置里的完整 env 更快，但不是 bitwise 一致，因此没有采用：

| Candidate | 10-frame optimized | Speedup vs GeoWeave baseline | 是否采用 | 原因 |
|---|---:|---:|---|---|
| old_submission_env | 678.411 ms | 1.197x | 否 | 输出非零差异 |
| old_plus_prev_topk | 512.275 ms | 1.586x | 否 | 输出非零差异，`points` max_abs 约 `0.0035` |
| old_plus_topk_fwd256 | 550.890 ms | 1.475x | 否 | 输出非零差异 |

主要风险来自 `VGGT_INDEXER_TOPK_FWD_MODE=triton`、tensor-core score policy、以及 sparse flash forward block/warps 这类会改变浮点累积路径的设置。它们数学目标相同，但不是 bitwise identical，因此不符合本轮“不改变计算逻辑”的口径。

## 推荐配置

### 严格 operator-only exact 配置

```bash
export VGGT_INDEXER_TOPK_BLOCK=512
export VGGT_INDEXER_TOPK_MERGE_BLOCKS=96
export VGGT_INDEXER_TOPK_MERGE_INDEX_BLOCK_K=128
export VGGT_SPARSE_FLASH_ATTN_BLOCK_N=128
export VGGT_SPARSE_FLASH_ATTN_NUM_WARPS=8
export VGGT_SPARSE_FLASH_ATTN_NUM_STAGES=2
export VGGT_SPARSE_FLASH_DISABLE_ATTN_SUM=1
```

说明：

- `VGGT_SPARSE_FLASH_DISABLE_ATTN_SUM=1` 只在 inference 且 `compute_loss=False` 时使用。已检查输出 tensor 完全一致。
- 当前 checkpoint 是 `topk=1024`，所以 512-only 的 custom fused topk CUDA kernel 不能强行用于正式结果。

### 最快 exact 配置

最终采用的 fastest exact 配置如下：

```bash
export VGGT_INDEXER_TOPK_BLOCK=256
export VGGT_INDEXER_TOPK_MERGE_BLOCKS=128
export VGGT_INDEXER_TOPK_MERGE_INDEX_BLOCK_K=64
export VGGT_SPARSE_FLASH_ATTN_BLOCK_N=128
export VGGT_SPARSE_FLASH_ATTN_NUM_WARPS=8
export VGGT_SPARSE_FLASH_ATTN_NUM_STAGES=2
export VGGT_SPARSE_FLASH_DISABLE_ATTN_SUM=1
export VGGT_DSA_FULLCHAIN_FASTPATH=1
export VGGT_DSA_FULLCHAIN_CUDAGRAPH=1
export VGGT_DSA_FULLCHAIN_GRAPH_WARMUP=1
export VGGT_SPARSE_FLASH_NO_QKV_CONTIG=1
export VGGT_SPARSE_FLASH_NO_KV_CONTIG=1
```

这组适合固定 shape warmed inference benchmark。它仍然通过 `max_abs=0.0` 检查，但应说明包含 CUDA graph / fastpath 调度，不只是单个 kernel 参数调优。

补充测试里，`VGGT_DSA_FUSE_QKV_INDEXER_PROJ=1` 是 exact，但没有带来进一步加速；100-frame optimized 为 `15.419s`，慢于最终采用的 `15.342s`。

## 验证协议

- 机器：单张 NVIDIA A800-SXM4-80GB。
- 输入：synthetic tensor `(1, N, 3, 392, 518)`。
- 精度：CUDA autocast bfloat16。
- Timing：CUDA event 包住直接 `model(imgs)` forward。
- 不包含：图片 I/O、resize、point interpolation、文件写出。
- 输出一致性：baseline env 和 optimized env 比较 `points`、`local_points`、`camera_poses`。

## GeoWeave 自身加速

### 严格 operator-only exact

| Frames | Baseline | Optimized | Speedup | Output max_abs |
|---:|---:|---:|---:|---:|
| 10 | 812.471 ms | 563.444 ms | 1.442x | 0.0 |
| 100 | 33.633 s | 15.506 s | 2.169x | 0.0 |

### 最快 exact fullchain / CUDA graph

| Frames | Baseline | Optimized | Speedup | Output max_abs |
|---:|---:|---:|---:|---:|
| 10 | 810.974 ms | 563.154 ms | 1.440x | 0.0 |
| 100 | 33.625 s | 15.344 s | 2.191x | 0.0 |

## 和 Pi3 base 的直接比较

使用最快 exact 配置。10 帧为 5 repeats，100 帧为 1 repeat；两者均 `warmup=1`。

| Frames | Pi3 base | GeoWeave optimized | GeoWeave / Pi3 latency |
|---:|---:|---:|---:|
| 10 | 305.675 ms | 563.991 ms | 1.845x |
| 100 | 6.514 s | 15.342 s | 2.355x |

对应 FPS：

| Frames | Pi3 base FPS | GeoWeave optimized FPS |
|---:|---:|---:|
| 10 | 32.714 | 17.731 |
| 100 | 15.353 | 6.518 |

注意：GeoWeave optimized 相比原始 GeoWeave baseline 明显变快，但仍然慢于 Pi3 base。因此 rebuttal 里不能写“GeoWeave 比 Pi3 快”，只能写“我们进一步优化了实现，显著降低 GeoWeave overhead；当前实现仍有 runtime overhead，会如实报告”。

## 原始输出

严格 output-equivalence / GeoWeave 自身加速：

- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/geoweave_cuda_ops_compare_20260704_formal2/strict_operator_only_10f.json`
- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/geoweave_cuda_ops_compare_20260704_formal2/strict_operator_only_100f.json`
- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/geoweave_cuda_ops_compare_20260704_formal2/fastest_exact_fullchain_10f.json`
- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/geoweave_cuda_ops_compare_20260704_formal2/fastest_exact_fullchain_100f.json`
- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/geoweave_cuda_ops_compare_20260704_formal3_best100/best100_exact_10f.json`
- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/geoweave_cuda_ops_compare_20260704_formal3_best100/best100_exact_100f.json`

Pi3 base 对比：

- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/geoweave_cuda_ops_compare_20260704_pi3_compare/fastest_exact_10f/synthetic_runtime_rows.json`
- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/geoweave_cuda_ops_compare_20260704_pi3_compare/fastest_exact_100f_skip_empty_cache_v2/synthetic_runtime_rows.json`
- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/geoweave_cuda_ops_compare_20260704_pi3_compare_best100/best100_pi3_10f/synthetic_runtime_rows.json`
- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/geoweave_cuda_ops_compare_20260704_pi3_compare_best100/best100_pi3_100f/synthetic_runtime_rows.json`

候选 sweep：

- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/geoweave_cuda_ops_compare_20260704_sweep2/`
- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/geoweave_cuda_ops_compare_20260704_sweep3_components/`
- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/geoweave_cuda_ops_compare_20260704_sweep4_topk/`
- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/geoweave_cuda_ops_compare_20260704_sweep5_topk_neighbors/`
- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/geoweave_cuda_ops_compare_20260704_sweep6_fullchain/`
- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/geoweave_cuda_ops_compare_20260704_sweep7_100f/`
- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/geoweave_cuda_ops_compare_20260704_sweep8_100f_lowblock/`
- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/geoweave_cuda_ops_compare_20260704_sweep9_100f_final_neighbors/`
