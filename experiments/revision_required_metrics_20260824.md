# Revision Required Metrics 2026-08-24

## Purpose

Complete the SIGGRAPH Asia 2026 required normal-level, depth-level, and training-time overhead evidence using the archived rebuttal protocols and cached model outputs.

## Status

- State: complete
- Branch: `revision-required-metrics-20260824`
- Base Git SHA: `0a664d7`
- Implementation Git SHA: `0c84180`
- Worktree: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/worktrees/revision-required-metrics-20260824`

## Configuration

- Host for GPU benchmark: `common-dev`
- Accelerator: one NVIDIA A800-SXM4-80GB
- Maximum GPU use: one GPU
- Python: `/mnt/cfs/zhoufeng/env/geoweave-rebuttal/bin/python`
- Base model parameters: 958,696,738
- GeoWeave scorer/indexer parameters: 4,761,252
- Cached inference root: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/geoweave_repro_outputs/rebuttal_gt_point_accuracy_inputs_20260702`
- GT root: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/data/rebuttal_bucket_gt_required_frames_20260702`
- Result root: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/revision_required_metrics_20260824`
- Persistent log: `/mnt/cfs/train_logs/zhoufeng/geoweave_revision_required_metrics_2026-08-24.log`

## Datasets

- ScanNet++ weak-overlap: 12 tuples, all 10 views.
- Waymo weak-overlap: 90 tuples, all 10 views.
- Waymo plausible distractor: 20 tuples, evaluated on the fixed 6-view prefix for clean and distractor variants.

## Planned Outputs

- Normal consistency paired with GT-backed point Acc/Comp.
- Depth AbsRel, RMSE, normalized RMSE, and delta1 after one tuple-level Sim(3).
- Forward/backward step latency and peak GPU memory for Pi3 and GeoWeave-Pi3.

## Commands and Results

- Metric flags: `--normal-metrics --depth-metrics`.
- ScanNet++: `24/24` rows successful.
- Waymo weak: `180/180` rows successful.
- Waymo distractor: `80/80` rows successful.
- Training benchmark: one A800, one warmup, three repeats, both models successful at 8-view 518x518 and 10-view 392x518.
- Full metrics finished at `2026-08-23T12:51:50-07:00` on the remote host.
- Training logs:
  - `/mnt/cfs/train_logs/zhoufeng/geoweave_revision_training_time_fair_8f518_2026-08-24.log`
  - `/mnt/cfs/train_logs/zhoufeng/geoweave_revision_training_time_fair_2026-08-24.log`
- Complete results and interpretation: `experiments/required_metrics_report_20260824.md`.
