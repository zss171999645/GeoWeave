# VGGT 项目综合进展文档（基于最近 200 commits）

更新日期：2026-03-06  
分支：`feat-vggt-eval-largescene`  
审计窗口：`2026-02-18` ~ `2026-03-06`（`git log -n 200`）

---

## 0. 这份文档回答什么

本文档聚焦四件事：
1. 做了哪些实验（训练/模型）
2. 做了哪些测试（评测与稳定性验证）
3. 做了哪些数据准备（数据处理/可读性修复）
4. 当前整体进度到哪里（含百分比与未完成项）

---

## 1. 当前进度总览

| 维度 | 当前阶段 | 进度 | 当前判断 |
|---|---|---:|---|
| 训练 | 工程化收敛 | 80% | 训练链路稳定可复用，重点从“跑通”转为“性能/精度收敛” |
| 测试 | 论文口径评测 + 集群加固 | 75% | 主评测可持续跑，容错增强明显，少数协议未闭环 |
| 数据准备 | 可读性修复 + 协议统一 | 70% | RE10K/CO3Dv2 主链路可读性问题基本解决，覆盖仍需补齐 |
| 模型改动 | 深改完成，持续调优 | 65% | 稀疏化核心已接入并可训练，仍在性能与稳定性并行优化 |

> 注：这里的进度是“面向论文可交付训练/评测链路”的工程进度，不是论文最终完稿进度。

---

## 2. 训练：已做实验与进展

### 2.1 训练链路建设（已完成）

1) 5090 sparse 主线脚本落地（topk2048, layers9-19, pointhead）
- commit：`0d23bf03`
- 文件：`aidi/scripts/vggt/submit_official_vggt_5090_sparse_fp16_layers9_19_topk2048_pointhead.sh`
- 结果：形成可直接提交的稀疏训练主入口。

2) Pi3 训练链路接入（warmup + sparse）
- commits：`07d8aec9`, `90e694bb`, `773f3594`
- 文件：`aidi/scripts/pi3/*`, `aidi/third_party/pi3_training/*`
- 结果：Pi3 训练与 meshx 主流程打通，可走两阶段训练。

3) 训练提交流程稳定性修复
- commits：`e39ac0a1`, `622e9d83`, `3f63a49f`, `aa460ce5`
- 文件：`aidi/scripts/vggt/train_official_vggt.sh` 等
- 结果：
  - 默认 args-file，避免远程长命令截断。
  - 覆盖参数解析更稳（`@args` 回退能力增强）。
  - 支持显式断点续训与历史 exp 复用。
  - kernel 环境变量可稳定透传到训练任务。

### 2.2 训练性能实验（已完成 + 进行中）

核心实验记录：`aidi/docs/records/vggt_dsa_train_fullstep_opt_20260302.md`

1) DSA 全流程 A/B/C benchmark 体系建立
- commit：`dd4c33f6`
- 脚本：`aidi/scripts/vggt/bench_dsa_train_fullstep_a_vs_c.py`
- 关键结论：
  - 在 `views=1..24, topk=512` 口径下，优化后 C 相比 B（原 sparse）平均提速约 `2.65x`。
  - C 仍慢于 A（dense 参考）约 `3.44x`，说明性能在收敛但尚未追平 dense。

2) 训练非法访存修复与回归验证
- commit：`02cee0d3`
- 文件：`easyvolcap/official_vggt/layers/dsa_attention.py`、`easyvolcap/utils/custom_*`
- 结果：修复 `no_qkv` 训练非法访存，更新训练记录与回归口径。

### 2.3 训练当前状态

- 已完成：训练基础设施可复用、可续训、可批量提交。
- 进行中：稀疏训练性能进一步逼近 dense 参考，同时保持数值稳定。
- 未完成：训练效率与精度的论文口径最终定版。

---

## 3. 测试：已做测试与进展

### 3.1 论文评测链路（已完成主线）

1) 论文口径评测脚本与提交流水
- commits：`13ee5427`, `453a664e`, `44e26471`, `f66f9312`
- 文件：`aidi/scripts/vggt/eval_official_vggt_sparse_datasets_plus.sh` + submit 脚本
- 结果：`official_paper` 等预设可执行，支持批量评测。

2) 结果导出与对比报表
- commits：`457998b3`, `5a2b1384`, `a1952bad`, `01182d03`
- 文件：`aidi/scripts/vggt/export_paper_eval_results_xlsx.py`, `aidi/scripts/vggt/format_eval_plus_results_xlsx.py`
- 结果：支持 checkpoint 批量汇总、对照行高亮、Excel 输出。

3) 近期评测结果样例（来自 `aidi/docs/vggt_paper_eval_results.md`）
- Pose（AUC@30, %）：
  - OFFICIAL_BASELINE：RE10K `85.1990`, CO3Dv2 `86.3180`
  - topk2048 ckpt29：RE10K `86.5459`, CO3Dv2 `86.7298`
- DTU（Overall, m）：
  - OFFICIAL_BASELINE：`0.004924`
  - topk2048 ckpt24：`0.004982`
- ETH3D（Overall, m）：
  - OFFICIAL_BASELINE：`0.027478`
  - topk512_ge34 ckpt34：`0.024020`

### 3.2 稳定性测试与容错修复（已完成多轮）

- commits：`0078b861`, `1aa36aab`, `422e6e2e`, `835382ac`, `212864e3`, `66515313`, `fbb0d705`
- 结果：
  - 修复评测任务秒退、job_type 构造、依赖阻塞问题。
  - open3d 依赖改为可选。
  - CO3Dv2 根路径健康探测与自动回退增强。
  - 指标计算的退化场景保护增强（如 Sim3 尺度问题）。

### 3.3 测试当前状态

- 已完成：主评测脚本 + 结果导出 + 集群运行稳定性。
- 进行中：协议一致性收敛（不同子集/采样口径统一）。
- 未完成：ScanNet-1500 matching 完整 pipeline；IMC/GSO/TAP-Vid 等附录/下游任务。

---

## 4. 数据准备：已做工作与进展

### 4.1 RE10K 数据准备与修复（已完成主链路）

1) 集群可读子集构建
- commit：`b1b29efe`
- 文件：`aidi/scripts/vggt/build_re10k_processed_clusterfix.py`

2) 软链目标归一化修复
- commit：`901b8827`
- 文件：`aidi/scripts/vggt/fix_re10k_processed_symlink_targets.py`

3) paper10 抽帧与时序采样修复
- commits：`a120f31f`, `cf510b7e`
- 文件：`aidi/scripts/vggt/repair_re10k_paper10_frames.py`, `easyvolcap/dataloaders/datasets/generalizable_dataset.py`

结果：RE10K 在集群的“可读 + 可评测 + 可复现实验抽样”闭环基本成立。

### 4.2 CO3Dv2 数据可读性修复（已完成主链路）

- commits：`93e5fd9f`, `670e9d40`, `5360bcad`, `835382ac`
- 文件：`aidi/scripts/vggt/prepare_co3dv2_eval_mirror.sh`, `aidi/scripts/vggt/eval_official_vggt_sparse_datasets_plus.sh`
- 结果：
  - 增加镜像同步与可读性探测。
  - 减少 bucket 权限与 rename 导致的评测中断。

### 4.3 Loader/预处理对齐（已完成）

- commits：`39e8c43c`, `1ba70136`
- 文件：`easyvolcap/dataloaders/datasets/multiview_point_dataset.py`
- 结果：支持 VGGT 官方 crop 预处理，修复 mask 缩放和 meta 缺失引发的崩溃。

### 4.4 数据准备当前状态

- 已完成：RE10K/CO3Dv2 主链路可读性问题大头已清理。
- 进行中：数据覆盖度与论文协议一致性继续对齐。
- 未完成：论文声明训练数据全集仍未全覆盖（如 Kubric/Habitat/PointOdyssey/Objaverse-like）。

---

## 5. 模型改动：已做改动与进展

### 5.1 核心改动范围

高频核心文件（最近 200 commits）：
- `easyvolcap/utils/custom_indexer/sparse_topk_indexer.py`（37 次）
- `easyvolcap/official_vggt/layers/dsa_attention.py`（19 次）
- `easyvolcap/utils/custom_flash_attn/sparse_index_flash_attn.py`（18 次）

### 5.2 关键改动与实验

1) 稀疏路径接入与 warmup kernel 试验
- commit：`71e3d3dc`
- 结果：增加 warmup dense flash kernel 路径与对照基准。

2) topk/indexer 前后向优化
- commits：`ca07dba5`, `086bf979` 等
- 结果：优化前后向路径，修复大序列地址溢出。

3) 训练全流程可选策略
- commits：`dd4c33f6`, `02cee0d3`, `c9ad584e`, `cf14800e`
- 结果：接入 BHTD、概率缓存、query 分块等策略，并保留回退。

4) 回退机制与风险控制
- 最近 200 commits 中 `revert` 共 9 条，主要集中在 indexer/triton 实验。
- 结论：模型改动采用“快迭代 + 可回退”的工程策略，风险可控但仍在收敛阶段。

### 5.3 模型当前状态

- 已完成：稀疏化核心模块可训练、可评测、可开关控制。
- 进行中：继续压缩训练全流程时延，稳定默认配置。
- 未完成：稀疏路径相对 dense 的最终效率/精度闭环定版。

---

## 6. 里程碑时间线（窗口内）

- `2026-02-24`：训练与 Pi3 链路集中完善。
- `2026-02-25` ~ `2026-02-27`：模型 kernel/indexer 高强度优化。
- `2026-03-02` ~ `2026-03-03`：训练全流程 benchmark + 结构修复。
- `2026-03-05` ~ `2026-03-06`：评测稳定性与数据可读性修复高峰（RE10K/CO3Dv2/集群入口）。

---

## 7. 下一步建议（按收益优先级）

1. 固化“训练效率 + 精度”双指标看板（避免只追速度）。
2. 冻结论文主评测协议口径（特别是 ETH3D 子集与 matching）。
3. 将 RE10K/CO3Dv2 clusterfix 与探测流程统一成单入口，减少脚本分叉。
4. 将模型侧大量实验开关收敛为“推荐默认 profile + 实验 profile”。

