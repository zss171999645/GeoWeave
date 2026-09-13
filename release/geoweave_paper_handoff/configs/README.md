# Config Pointers

This directory contains handoff-facing config entry points and points to the
canonical repository configs. Full training configs remain in their original
locations because they use repository-relative include chains and dataset roots.

## VGGT / GeoWeave

| Role | Canonical path |
| --- | --- |
| Model config | `configs/models/vggt_official_finetune.yaml` |
| Paper training config family | `configs/exps/vggt/vggt_official_finetune_5090_paper.yaml` |
| 5090 memory spec | `configs/specs/vggt/official/memory_5090.yaml` |
| Unified local eval config | `aidi/configs/vggt/unified_eval_local.yaml` |
| Trusted paper-era eval config | `aidi/configs/vggt/unified_eval_vggt_indexer_20260503_pt79_trusted.yaml` |
| Handoff final eval config | `release/geoweave_paper_handoff/configs/eval/vggt_geoweave_final_trusted.yaml` |
| Handoff smoke eval config | `release/geoweave_paper_handoff/configs/eval/vggt_geoweave_smoke_eth3d_pose.yaml` |

## Pi3 / GeoWeave

| Role | Canonical path |
| --- | --- |
| Model config | `aidi/third_party/pi3_training/configs/model/pi3.yaml` |
| Warm-up training config | `aidi/third_party/pi3_training/configs/train/train_pi3_lowres_indexer_warmup.yaml` |
| Sparse training config | `aidi/third_party/pi3_training/configs/train/train_pi3_lowres_indexer_sparse.yaml` |
| Final sparse wrapper family | `aidi/scripts/pi3/submit_pi3_5090_sparse_klwarm_then_task.sh` |
| Paper sparse stage2 wrapper | `aidi/scripts/pi3/submit_pi3_5090_sparse_fp16_layers9_21.sh` |
| Handoff final eval config | `release/geoweave_paper_handoff/configs/eval/pi3_geoweave_final_all_tasks.yaml` |
| Handoff smoke eval config | `release/geoweave_paper_handoff/configs/eval/pi3_geoweave_smoke_eth3d_pose.yaml` |

## Handoff Eval Configs

The handoff eval configs intentionally point to the fixed bucket weights under:

`/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/trained_model/geoweave_paper_handoff_20260602_final`

Use these configs for new handoff runs instead of the older experiment-local
checkpoint paths in the original records.

## Evaluation Records

Use these files as first-level facts before changing any config:

| Role | Canonical path |
| --- | --- |
| Protocol overview | `aidi/docs/vggt_train_test_protocol_overview.md` |
| Paper result summary | `aidi/docs/vggt_paper_eval_results.md` |
| Experiment records | `aidi/docs/records/experiment.md` |
| Submit records | `aidi/docs/submit_records.md` |
