# Final Expanded Waymo Distractor-Count Sweep

Date: 2026-08-26

## Final protocol

- Historical set: 20 frozen Waymo plausible-wrong examples.
- New set: all 37 unique clean anchors eligible under the frozen high-margin Waymo2 testing protocol.
- Final size: 57 examples.
- New-anchor coverage: 24 camera sequences, 10 driving segments, and all five cameras.
- Each example has six fixed evaluated clean views and four context slots.
- Distractor count: zero through four nested replacements.
- Hard feature condition: `distance(prefix, distractor) < distance(prefix, clean context)`.
- Margin floor: `0.00022410058591049165`, the minimum historical margin.
- Selection ranking: historical composite feature/structure score without model outputs.
- Inference/evaluation: paper Pi3 and GeoWeave-Pi3 checkpoints, 512-pixel loading width, cached camera poses, evaluation indices `[0,1,2,3,4,5]`.

The top 20 new anchors have mean margin 0.000272. The remaining 17 have mean margin 0.000252 and minimum margin 0.000227. All 185 new inputs per model and their pose rows passed completeness and finite-value checks.

## New-37 result

| Distractors | Pi3 ATE | GeoWeave ATE | ATE degradation advantage | Pi3 RPE-t | GeoWeave RPE-t | RPE-t degradation advantage |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.060305 | 0.056210 | 0.000000 | 0.121146 | 0.099768 | 0.000000 |
| 1 | 0.062497 | 0.053392 | +0.005011 | 0.118072 | 0.096669 | +0.000026 |
| 2 | 0.065513 | 0.054629 | +0.006788 | 0.122658 | 0.098111 | +0.003169 |
| 3 | 0.067463 | 0.056323 | +0.007045 | 0.127490 | 0.096327 | +0.009785 |
| 4 | 0.091548 | 0.075602 | +0.011851 | 0.160963 | 0.126854 | +0.012731 |

GeoWeave has lower absolute ATE and RPE-t at every level and smaller clean-relative degradation at all four nonzero levels. The New-37 degradation-area advantages are `+0.007674` ATE and `+0.006428` RPE-t; the full-sweep gate passes. Paired-bootstrap confidence intervals for ATE advantage are positive at one and two distractors and cross zero at three and four; the endpoint mean remains positive but should not be described as uniform per-example superiority.

## Combined-57 result

| Distractors | Pi3 ATE | GeoWeave ATE | ATE degradation advantage | Pi3 RPE-t | GeoWeave RPE-t | RPE-t degradation advantage |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.073744 | 0.072653 | 0.000000 | 0.135606 | 0.121243 | 0.000000 |
| 1 | 0.073039 | 0.071051 | +0.000897 | 0.131628 | 0.119198 | -0.001933 |
| 2 | 0.074630 | 0.065889 | +0.007649 | 0.133600 | 0.113181 | +0.006056 |
| 3 | 0.074907 | 0.065206 | +0.008610 | 0.136879 | 0.110340 | +0.012176 |
| 4 | 0.139881 | 0.110637 | +0.028153 | 0.217832 | 0.165047 | +0.038421 |

GeoWeave has lower absolute ATE and RPE-t at all five levels. The Combined-57 degradation-area advantages are `+0.011327` ATE and `+0.013680` RPE-t; the full-sweep gate passes. At four distractors, the paired-bootstrap 95% intervals are `[0.008398, 0.053442]` for ATE advantage and `[0.010444, 0.073272]` for RPE-t advantage. At one distractor, the RPE-t degradation advantage is slightly negative and statistically mixed, so the result supports increasing robustness under stronger distractor context rather than uniform superiority at every count.

## Outputs

- Top-20 root: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/expanded_waymo_distractor_highmargin_w12_20260826`
- Remaining-17 root: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/expanded_waymo_distractor_highmargin_remaining17_20260826`
- Final summaries: `expanded_waymo_distractor_highmargin_w12_20260826/analysis57`
- New-37 gate: `analysis57/new37_full_gate.json`
- Combined-57 gate: `analysis57/combined57_full_gate.json`
- Per-level CSV/JSON: `analysis57/level_summary.csv` and `analysis57/level_summary.json`

The manuscript and PDFs were not modified.
