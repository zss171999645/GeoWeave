# High-Margin Expanded Waymo Distractor Sweep

Date: 2026-08-26

## Protocol

This is the valid expansion of the historical 20-example Waymo distractor stress test. It uses only image statistics for selection and does not use model outputs.

- Source: `/mnt/cfs/datasets/Waymo2/testing`.
- Temporal candidates: 12 evenly spaced stride-four ten-view anchors per camera sequence, 960 anchors total.
- Hard condition: `distance(prefix, distractor) < distance(prefix, clean context)`.
- Margin floor: `0.00022410058591049165`, equal to the minimum margin of the historical 20 examples.
- Ranking: historical composite feature/structure score.
- Eligible pairs: 396 from 37 unique clean anchors.
- Selected expansion: top 20 unique clean anchors.
- Selected mean margin: 0.000272; range 0.000225--0.000355.
- Selected mean clean-context distance: 0.000419.
- Selected mean distractor distance: 0.000147.
- Inputs per example: six fixed evaluated clean views and four context slots with zero through four nested distractor replacements.
- Inference/evaluation: paper Pi3 and GeoWeave-Pi3 checkpoints, 512-pixel loading width, camera-only cache, evaluation indices `[0,1,2,3,4,5]`.

All 100 tuples and all 100 prediction caches per model passed completeness and finite-value checks.

## New-20 full sweep

| Distractors | Pi3 ATE | GeoWeave ATE | ATE degradation advantage | Pi3 RPE-t | GeoWeave RPE-t | RPE-t degradation advantage |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.054900 | 0.052878 | 0.000000 | 0.108048 | 0.091673 | 0.000000 |
| 1 | 0.060226 | 0.054187 | +0.004017 | 0.119592 | 0.097646 | +0.005571 |
| 2 | 0.061090 | 0.055579 | +0.003489 | 0.119543 | 0.099943 | +0.003226 |
| 3 | 0.062512 | 0.054271 | +0.006218 | 0.124340 | 0.094876 | +0.013089 |
| 4 | 0.072882 | 0.061772 | +0.009088 | 0.129686 | 0.100414 | +0.012897 |

Positive degradation advantage means Pi3 degrades more than GeoWeave relative to each model's own zero-distractor baseline. GeoWeave has lower absolute ATE and RPE-t at every level, and lower degradation at all four nonzero levels. The endpoint advantages are `+0.009088` ATE and `+0.012897` RPE-t. The degradation-area advantages are `+0.005703` and `+0.008696`, so the New-20 full-sweep gate passes.

## Combined-40 full sweep

| Distractors | Pi3 ATE | GeoWeave ATE | ATE degradation advantage | Pi3 RPE-t | GeoWeave RPE-t | RPE-t degradation advantage |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.076753 | 0.077976 | 0.000000 | 0.135203 | 0.126322 | 0.000000 |
| 1 | 0.076384 | 0.078954 | -0.001348 | 0.138149 | 0.129261 | +0.000007 |
| 2 | 0.076294 | 0.071150 | +0.006366 | 0.136693 | 0.120501 | +0.007311 |
| 3 | 0.075595 | 0.067955 | +0.008862 | 0.139294 | 0.115569 | +0.014844 |
| 4 | 0.151090 | 0.118612 | +0.033700 | 0.226362 | 0.168059 | +0.049422 |

The Combined-40 endpoint paired-bootstrap 95% intervals are `[0.008342, 0.067441]` for ATE advantage and `[0.015277, 0.097120]` for RPE-t advantage. The full-sweep degradation-area advantages are `+0.011895` ATE and `+0.017896` RPE-t, so the Combined-40 gate passes. The one-distractor ATE level is mixed and should not be presented as uniform per-level superiority.

## Expansion decision

At the fixed historical margin floor, the testing split yields only 37 unique eligible clean anchors. Therefore the same simple protocol can expand the paper from 20 to 40 examples, but cannot reach 100 examples without changing the candidate source or margin rule. The experiment stops at Combined-40 rather than changing the frozen protocol.

## Outputs

- Root: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/expanded_waymo_distractor_highmargin_w12_20260826`
- Selection manifest: `tuples/selection_manifest.json`
- Protocol: `protocol_selected20.json`
- Predictions: `predictions_selected20`
- New-20 full pose rows: `pose_selected20_full`
- New-20 gate: `analysis/new20_full_gate.json`
- Combined-40 gate: `analysis/combined40_full_gate.json`
- Per-level summary: `analysis/level_summary.csv` and `analysis/level_summary.json`

The manuscript and PDFs were not modified.
