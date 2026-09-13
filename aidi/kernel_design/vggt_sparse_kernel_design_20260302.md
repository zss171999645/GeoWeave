# VGGT Sparse Kernel 设计记录（2026-03-02）

本文记录当前仓库中 DSA/indexer 的 kernel 设计，聚焦推理与训练的核心前后向路径。

## 1. 目标与范围

- 目标：在不改变 DSA 计算语义的前提下，将 `topk` 选点与稀疏 attention 的算子路径尽量 kernel 化。
- 范围：
  - indexer topk：`easyvolcap/utils/custom_indexer/sparse_topk_indexer.py`
  - sparse flash attention：`easyvolcap/utils/custom_flash_attn/sparse_index_flash_attn.py`
  - 接入层：`easyvolcap/official_vggt/layers/indexer.py`、`easyvolcap/official_vggt/layers/dsa_attention.py`

## 2. 顶层执行路径

### 2.1 Indexer 选点（topk）

1) `LightningIndexer.forward`
- 输入 `x -> (q,k,w)` 投影，得到 `[B,T,H,D]` 与 `[B,T,H]`。
- 当 `use_topk_kernel=True` 时调用 `sparse_topk_indexer_func(...)`。

2) `SparseTopkIndexerFunc.forward`
- 分块打分（einsum/headwise/triton）得到 block scores。
- 分阶段 topk 合并（支持 runtime bucket、query 外分块、workspace 复用）。
- 输出 `topk_indices, topk_scores`。

3) `SparseTopkIndexerFunc.backward`（Triton）
- 核心 kernel：`_indexer_topk_bwd_kernel`
- 输入：`q,k,w,topk_idx,grad(topk_scores)`。
- 输出：`dq,dk,dw`。

### 2.2 Sparse Attention

1) `DSAAttention.forward`
- 由 `topk_indices` gather 稀疏 K/V 候选后进入 sparse flash 路径。
- 推理禁用 `attn_sum` 时可走 inference-only 函数。

2) `SparseIndexFlashAttnFunc.forward`
- 核心 forward kernel：`_fwd_kernel`
- 可选 `attn_sum` kernel：`_attn_sum_kernel`
- 输出：`out, attn_sum`

3) `SparseIndexFlashAttnFunc.backward`
- 预处理：`_bwd_preprocess_do_o_dot`
- 主 backward kernel：`_bwd_sparse_index_kernel`
- 输出：`dq,dk,dv`

## 3. 新增：BHTD backward kernel 路径

为减少训练路径中不必要的布局转换，新增了 BHTD autograd 入口：

- 新函数：
  - `SparseIndexFlashAttnBhtdFunc`（forward+backward）
  - `sparse_index_flash_attn_bhtd_func`
- 新能力：
  - `_sparse_index_flash_attn_backward(..., q_layout=\"bhtd\")`
  - backward 内部按布局解析 `stride_h/stride_m`，复用同一组 Triton backward kernel。
- 接入开关（默认关闭，不影响现有管线）：
  - `VGGT_SPARSE_FLASH_BHTD_AUTOGRAD=1`

默认路径仍是原有 `bthd` autograd，不会改变当前训练/推理行为。

## 4. 当前主瓶颈（profile 结论）

在 `view=200, topk=512` 的当前最佳配置中，主要耗时仍集中在：

1) `aten::topk`
2) `_score_block_kernel`
3) `_fwd_kernel`（sparse flash）

说明优化优先级应继续放在“选点打分 + topk 合并”的进一步融合上。

## 5. 数值一致性口径

数值对比统一按以下口径执行：

- 对比对象：
  - kernel 路径：`use_topk_kernel=True, use_sparse_flash_attn=True`
  - 非 kernel 路径：`use_topk_kernel=False, use_sparse_flash_attn=False`
- 同一模型权重、同一输入、同一随机种子。
- 统计：
  - forward：`out`（及可选 `indexer_loss`）`max_abs/mean_abs`
  - backward：`x.grad` 与关键参数梯度 `max_abs/mean_abs`

## 6. 本次实测结果（forward + backward）

测试环境：
- 机器：`gpu-4090-dev014`
- 容器：`vggt-zf-1201`
- GPU：空闲卡 `5`
- 命令：
  - `CUDA_VISIBLE_DEVICES=5 python aidi/scripts/vggt/check_kernel_forward_backward_diff.py --views 1 2 3 4 --dtype bf16 --include-bhtd-autograd 1 --out tmp/kernel_forward_backward_diff_gpu5_v1_v4.json`

关键结果（kernel vs 非kernel）：

| view | seq_len | forward out max_abs | backward x.grad max_abs | indexer_loss abs diff |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1374 | 4.8828125e-04 | 1.1324883e-06 | 0.0 |
| 2 | 2748 | 4.8828125e-04 | 1.2423843e-06 | 0.0 |
| 3 | 4122 | 4.8828125e-04 | 1.3113022e-06 | 0.0 |
| 4 | 5496 | 9.7656250e-04 | 1.2442470e-06 | 0.0 |

参数梯度差异（view=4，max_abs）：
- `qkv.weight`: `1.6689300537109375e-04`
- `proj.weight`: `2.384185791015625e-07`
- `indexer.q_proj.weight`: `4.76837158203125e-07`
- `indexer.k_proj.weight`: `9.5367431640625e-07`
- `indexer.w_proj.weight`: `3.0517578125e-05`

补充：
- `BHTD autograd kernel` 与 `BTHD autograd kernel` 在上述测试中误差量级一致。
- 在 `tokens_per_view=1374` 时，非kernel参考路径从 `view>=5` 开始在 24GB 卡上 OOM，因此大 view 的“全量非kernel前后向对照”无法直接跑通。
