# Distractor Ratio Sweep 2026-08-24

## Purpose

Evaluate Pi3 and GeoWeave-Pi3 as a controlled function of context distractor ratio while keeping a fixed six-view evaluation prefix and ten total input views.

## Status

- State: complete
- Branch: `revision-required-metrics-20260824`
- Design commit: `76462bb`
- Implementation Git SHA: `40251ab`
- Execution base Git SHA: `a8ddc49`
- Worktree: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/worktrees/revision-required-metrics-20260824`

## Configuration

- Dataset: existing 20 paired Waymo clean/distractor tuples.
- Variants: `noise0`, `noise1`, `noise2`, `noise3`, `noise4`.
- Context distractor ratios: 0%, 25%, 50%, 75%, 100%.
- Total views: 10; evaluated views: `[0,1,2,3,4,5]`.
- Models: Pi3 base and GeoWeave-Pi3 paper checkpoints.
- GPU allocation: at most two A800 GPUs on common-dev, one process/model/GPU.
- Output root: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/revision_distractor_ratio_20260824`.
- Persistent log: `/mnt/cfs/train_logs/zhoufeng/geoweave_distractor_ratio_2026-08-24.log`.

## Success criteria

- 100 validated fixed-ten-view input variants.
- 100 finite inference outputs per model.
- Complete pose and point/normal/depth rows for all 200 model/sample/level combinations.
- Aggregate absolute and clean-relative metrics for all five levels.
- Paper-ready Markdown/CSV/JSON summaries and a trend plot.

## Execution record

Input construction command:

```bash
python aidi/scripts/baselines/build_variable_distractor_from_pairs.py \
  --input-root /mnt/cfs/zhoufeng/rebuttal_received_20260701/extracted/waymo_simple_plausible_wrong_context_1c64b06c_20260512_2248 \
  --output-root /mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/revision_distractor_ratio_20260824/inputs \
  --datasets waymo --variants 0,1,2,3,4 --total-views 10 --eval-views 6 \
  --output-total-mode fixed --clean-fill-mode tail_clean
```

Pi3 and GeoWeave inference use `tools/reprodata_pi3_protocol_infer.py`, variants `noise0 noise1 noise2 noise3 noise4`, width 518, native point source, and one GPU per model. Pose evaluation uses `tools/eval_reprodata_precomputed_pi3_relpose.py`; structure evaluation uses `tools/reprodata_gt_point_accuracy.py --normal-metrics --depth-metrics`.

## Completed execution

- Input construction and validation: 20 samples x 5 variants = 100 fixed-ten-view inputs; all fixed prefix hashes matched.
- Endpoint equivalence: `noise0` and `noise4` inputs matched the original clean/distractor endpoints byte-for-byte for all 20 samples and 10 views.
- Full inference: 2026-08-24 15:11:36 to 15:48:07 (Asia/Shanghai).
- Inference outputs: 100/100 finite Pi3 outputs and 100/100 finite GeoWeave-Pi3 outputs. Every archive contains points, camera poses, colors, and image paths.
- Pose evaluation: 100/100 successful rows per model.
- Point/normal/depth evaluation: 200/200 successful rows; 0 missing and 0 failed.
- GPU smoke: 10/10 jobs passed before the full run.
- Final outputs: `report/distractor_ratio_summary.{md,json}`, `report/distractor_ratio_{absolute,degradation}.csv`, and `report/distractor_ratio_trends.png` under the output root.

## Result summary

For context distractor ratios from 0% through 75%, the metrics are mostly flat and non-monotonic. At 100% context distractors, pose accuracy degrades sharply while the fixed-prefix point, normal, and depth metrics change only slightly.

- Pi3 ATE: 0.084127 -> 0.177483; RPE-t: 0.144310 -> 0.272633.
- GeoWeave-Pi3 ATE: 0.095843 -> 0.220332; RPE-t: 0.153437 -> 0.269468.
- GeoWeave-Pi3 is not consistently more robust than Pi3 in this controlled sweep: its 100% clean-relative ATE degradation is larger, while its RPE-t and depth-delta1 degradations are slightly smaller. Structure-level trends are mixed.

This sweep therefore supports a bounded conclusion: replacing all four context views exposes a pose-level underconstrained failure regime, but the current 20-pair study does not support a broad claim that GeoWeave dominates Pi3 across distractor ratios.

## Verification note

The reconstructed `noise0`/`noise4` endpoint inputs and cached point outputs were checked against the original artifacts and were identical. A direct one-sample run of the repository's pose evaluator also matched the cached-pose evaluator exactly. However, the current endpoint aggregate ordering differs from the older Table 4 product, indicating an old evaluator/code-state difference. Do not overwrite Table 4 with these pose numbers until that historical evaluation environment is reconciled.

The first construction attempt used the five-pair visualization export instead of the formal twenty-pair reproduction root. It stopped on an incomplete fifth noise tuple; that partial output was preserved as `revision_distractor_ratio_20260824_partial_wrong_source` and is excluded from all results. Subsequent pipelines use `set -o pipefail`.
