# Pi3 Native 与 EVC/VGGT 训练框架边界

本文档只固化当前清理重构的边界，不替代实验记录。后续拆分目录、抽公共工具或改提交脚本时，必须先保证这里列出的 Pi3 native 成功启动链路语义不变。

## Pi3 Native 成功启动链路

以 2026-04-30 最近成功提交/启动的 Pi3 native 训练为基准，当前不能移动 Pi3 native launcher，也不能改变它们生成的关键环境变量和 Hydra override：

- full17 highres finetune：
  - launcher: `aidi/scripts/pi3/submit_pi3_5090_training.sh`
  - shared launcher: `aidi/scripts/pi3/train_pi3_official.sh`
  - data/config: `meshx_pi3_vggt17`, `STAGE=highres`, `LOAD_VGGT=0`
  - successful-submit invariants: `TRAIN_ITERS_PER_EPOCH=800`, `SAVE_TO_AIDI=1`, `CORE4_VAL=1`, `CORE4_FIRST_EVAL=1`, `[2,24]`, bf16, low LR and 3% OneCycle warm-up
  - memory controls: `PI3_DECODER_ATTN_BACKEND=flash`, `PI3_QK_NORM_CHUNK_SIZE=2048`, `PI3_HEAD_USE_CHECKPOINT=1`, `PI3_HEAD_VIEW_CHUNK_SIZE=0`
  - submit-safe optional controls: `PI3_TRAIN_FIXED_RES`, `PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE`, `PI3_TEST_NORMAL_LOSS_VIEW_CHUNK_SIZE`; full finetune 固定分辨率/normal loss chunk 应优先走这些 env，避免在外层提交命令直接传带分号的 `EXTRA_OVERRIDES`
  - index/cache controls: `PI3_LAZY_SEQUENCE_INDEX=1`, `PI3_USE_INDEX_CACHE=1`, `PI3_INDEX_CACHE_WAIT_SEC=0`

- init-old indexer warm-up：
  - launcher: `aidi/scripts/pi3/submit_pi3_5090_warmup.sh`
  - shared launcher: `aidi/scripts/pi3/train_pi3_official.sh`
  - data/config: `TRAIN_CFG=train_pi3_lowres_indexer_warmup`, `meshx_pi3_vggt17`, `LOAD_VGGT=0`
  - successful-submit invariants: `WARMUP_ITERS_PER_EPOCH=800`, `TOPK=512`, `STREAMING_KL_LOSS=1`, `STREAMING_KL_AUTOGRAD=1`, `WARMUP_ONLY_INDEXER_TRAIN=True`, `PI3_STATIC_GRAPH=1`
  - indexer init: `INDEXER_INIT_CKPT` points to the old Pi3/EVC warm-up checkpoint and is passed through only when non-empty.

- scratch all-layer dynamic-resolution warm-up：
  - launcher: `aidi/scripts/pi3/submit_pi3_5090_warmup.sh`
  - source workspace pattern: `/home/users/feng01.zhou/workspace/meshx_submit_pi3_warmup_alllayer_cf6a8df0`
  - successful-submit invariants: `INDEXER_LAYERS=all`, no `INDEXER_INIT_CKPT`, `TRAIN_DYNAMIC_RES=1`, `WARMUP_ITERS_PER_EPOCH=800`, `TOPK=512`, streaming KL, frozen Pi3 backbone.

## Ownership Boundary

- Pi3 native owns `aidi/scripts/pi3/**`, `aidi/third_party/pi3_training/**`, Pi3 Hydra configs, Accelerate trainer, native Pi3 model wrappers, Pi3 dataset adapters and Pi3 TensorBoard/AIDI bridge behavior.
- EVC/VGGT owns `easyvolcap/**`, `configs/exps/vggt/**`, `configs/models/vggt_*.yaml`, `aidi/scripts/vggt/**`, `OfficialVGGTModel`, EVC runner behavior and legacy EVC training/eval entrypoints.
- Pi3 native 允许复用 `easyvolcap` 的低层 utility / DSA-indexer kernel 代码，但禁止依赖 EVC runner、`OfficialVGGTModel` 或 `evc-train/evc-test` 训练入口；这两类共享必须区分。
- Shared neutral bridge is allowed under `aidi/utils/` only when it does not name one framework as the owner of the other. Current shared bridge files are:
  - `aidi/utils/core4_main_val.py`
  - `aidi/utils/core4_model_adapter.py`
- Backward-compatibility wrappers may remain where old imports already exist. `aidi/utils/vggt_core4_main_val.py` is now a compatibility path and should not be imported by new Pi3 native code.

## Refactor Rules

- Do not move Pi3 native launcher files until a dry-run/job-package check proves the generated command still contains the same success-chain overrides.
- 中文口径：不移动 Pi3 native launcher，直到上述 dry-run/job-package 证据补齐。
- Do not make Pi3 native import EVC runner classes or `easyvolcap/runners/volumetric_video_runner.py`; the only allowed shared validation bridge is the neutral `aidi.utils.core4_*` layer.
- Do not make EVC/VGGT training import Pi3 trainer code from `aidi/third_party/pi3_training/**`.
- Keep `aidi/scripts/pi3/train_pi3_official.sh` as the single Pi3 native shared launcher. Thin submit wrappers may set defaults, but training logic stays in the shared launcher.
- Treat `tests/pi3_recent_success_launch_tests.py` as the guardrail for this split. Any cleanup that changes these tests must be justified against the recorded 2026-04-30 Pi3 successful submissions.
