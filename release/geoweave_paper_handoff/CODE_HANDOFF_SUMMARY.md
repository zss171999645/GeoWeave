# GeoWeave Code Handoff Summary

这份代码包是基于 meshx 结构整理的 GeoWeave 论文交接代码。接手人优先看
`release/geoweave_paper_handoff/`，再按需要进入 `easyvolcap/`、
`aidi/third_party/pi3_training/` 和 `configs/`。

## 1. 核心结构

- `release/geoweave_paper_handoff/`：交接入口、最终 eval config、权重索引和检查脚本。
- `easyvolcap/official_vggt/`：VGGT 主体代码，GeoWeave sparse/indexer 在这里接入。
- `easyvolcap/models/official_vggt_model.py`：meshx 内部调用 VGGT / GeoWeave 的封装。
- `easyvolcap/utils/custom_indexer/`：topk、streaming KL、indexer 相关 CUDA/Python 实现。
- `easyvolcap/utils/custom_flash_attn/`：sparse attention 相关 CUDA/Python 实现。
- `aidi/third_party/pi3_training/`：Pi3 训练代码，包含 GeoWeave warm-up 和 sparse stage2。
- `configs/`、`aidi/configs/vggt/`：训练和评测配置；交接优先看 handoff 目录下的 config。

## 2. 主入口

优先使用这些 wrapper，不建议从历史脚本里挑入口：

```bash
bash release/geoweave_paper_handoff/scripts/check_handoff.sh

bash release/geoweave_paper_handoff/scripts/train_vggt.sh --warmup-smoke
bash release/geoweave_paper_handoff/scripts/train_vggt.sh --smoke
bash release/geoweave_paper_handoff/scripts/train_vggt.sh --paper --dry-run

bash release/geoweave_paper_handoff/scripts/train_pi3.sh --warmup-smoke
bash release/geoweave_paper_handoff/scripts/train_pi3.sh --smoke
bash release/geoweave_paper_handoff/scripts/train_pi3.sh --paper --dry-run

bash release/geoweave_paper_handoff/scripts/eval_vggt.sh --paper --dry-run
bash release/geoweave_paper_handoff/scripts/eval_pi3.sh --paper --dry-run
```

正式启动时去掉 `--dry-run`。`--smoke` 只用于启动验证，不能和论文表格指标比较。

## 3. 权重与配置

权重固定在：

`/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/trained_model/geoweave_paper_handoff_20260602_final`

索引见：`release/geoweave_paper_handoff/manifests/WEIGHTS.tsv`。

| 用途 | 权重 | 配置/入口 |
| --- | --- | --- |
| VGGT final eval | `vggt_geoweave_20260503_pt79/79.pt` | `configs/eval/vggt_geoweave_final_trusted.yaml` |
| VGGT train init | `vggt_official_VGGT-1B/*.pt` | `train_vggt.sh --warmup-*` / `train_vggt.sh --paper` |
| Pi3 final eval | `pi3_geoweave_native_sparse_20260505_checkpoint_79/checkpoint_79/pytorch_model.bin` | `configs/eval/pi3_geoweave_final_all_tasks.yaml` |
| Pi3 final resume | `pi3_geoweave_native_sparse_20260505_checkpoint_79/checkpoint_79/` | `train_pi3.sh --resume-final --dry-run` |
| Pi3 sparse init | `pi3_geoweave_stage2_init_checkpoint_49/pytorch_model.bin` | `train_pi3.sh --smoke` / `train_pi3.sh --paper` |
| Pi3 warm-up init | `pi3_base_yyfz233/Pi3_model.safetensors` | `train_pi3.sh --warmup-*` |

## 4. 当前验证状态

- `STRICT_IMPORT=1 check_handoff.sh` 已在 `vggt-zf-1201` 容器通过。
- VGGT warm-up / sparse 2-GPU smoke 已通过。
- Pi3 warm-up / sparse 2-GPU smoke 已通过。
- VGGT / Pi3 paper eval 入口各自跑满 15 分钟，无错误日志。
- 没有跑正式长训，也没有跑完整 paper benchmark；交接阶段只确认入口、配置、权重、训练 smoke 和评测启动可用。

## 5. 不建议随意改动

- `easyvolcap/official_vggt/layers/indexer.py`
- `easyvolcap/official_vggt/layers/dsa_attention.py`
- `easyvolcap/official_vggt/models/aggregator.py`
- `easyvolcap/models/official_vggt_model.py`
- `easyvolcap/utils/custom_indexer/`
- `easyvolcap/utils/custom_flash_attn/`
- `aidi/third_party/pi3_training/pi3/models/pi3_training.py`
- `aidi/third_party/pi3_training/trainers/pi3_trainer.py`
- `aidi/scripts/vggt/run_unified_eval.py`
- `release/geoweave_paper_handoff/configs/eval/*.yaml`

## 6. 结论

这版代码包适合公司内部交接：结构仍然是 meshx，入口集中，历史 wrapper 已收敛，
VGGT/Pi3 两条 GeoWeave 训练和评测路径都保留。后续不建议再为“看起来更干净”
大幅重构；如需继续瘦身，应先确认不会影响上述入口。
