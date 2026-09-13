# Controlled Distractor-Ratio Sweep

## Protocol

- Dataset: 20 paired Waymo examples.
- Input length: 10 views.
- Evaluation prefix: the same six clean views at indices 0--5 for every condition.
- Context views: four slots at indices 6--9, with 0/1/2/3/4 slots replaced by plausible distractors.
- Context distractor ratios: 0%, 25%, 50%, 75%, and 100% (equivalently 0%, 10%, 20%, 30%, and 40% of all input views).
- Models: Pi3 and GeoWeave-Pi3.
- Metrics: pose (ATE, RPE translation/rotation), point accuracy/completeness, normal consistency, and depth AbsRel/RMSE/delta1.

All 100 input variants passed byte-level fixed-prefix validation. Both models produced 100/100 finite predictions, and all 200 model/sample/ratio combinations completed pose and structure evaluation.

## Main result

| Model | Context distractors | ATE down | RPE-t down | Point Acc down | Point Comp down | Normal Consistency up | Depth AbsRel down | Depth delta1 up |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pi3 | 0% | 0.084127 | 0.144310 | 0.030547 | 0.069366 | 0.678188 | 0.241666 | 0.776648 |
| Pi3 | 100% | 0.177483 | 0.272633 | 0.030877 | 0.070024 | 0.676698 | 0.241658 | 0.769527 |
| GeoWeave-Pi3 | 0% | 0.095843 | 0.153437 | 0.031418 | 0.076775 | 0.677296 | 0.245047 | 0.764432 |
| GeoWeave-Pi3 | 100% | 0.220332 | 0.269468 | 0.031516 | 0.076522 | 0.674312 | 0.247003 | 0.758745 |

The complete five-level table and clean-relative deltas are in `distractor_ratio_summary.md` and the companion CSV/JSON files.

## Interpretation

From 0% to 75% context distractors, both models fluctuate within a relatively narrow, non-monotonic range. Replacing all four context views produces a clear pose failure: relative to 0%, Pi3 ATE rises by 0.093356 (+111.0%) and GeoWeave-Pi3 ATE rises by 0.124489 (+129.9%). The RPE-t increases are 0.128323 (+88.9%) and 0.116032 (+75.6%), respectively.

The fixed-prefix point, normal, and depth metrics are substantially more stable. At 100%, Pi3 point accuracy changes by +0.000331 and depth delta1 falls by 0.007121; GeoWeave-Pi3 point accuracy changes by +0.000098 and depth delta1 falls by 0.005687. This suggests that the distractors primarily destabilize global pose consistency before strongly damaging local structure on the evaluated clean views.

GeoWeave-Pi3 is not uniformly more robust than Pi3 in this sweep. It has a smaller 100% RPE-t degradation and slightly smaller depth-delta1 degradation, but a larger ATE degradation; other structure metrics are mixed. The defensible paper use is therefore as a controlled stress test and an underconstrained-case limitation, not as evidence of across-the-board superiority.

## Compatibility caveat

The 0% and 100% endpoint inputs match the original clean/distractor artifacts byte-for-byte, and newly generated point maps match the old cached point maps exactly. Direct and cached pose evaluation agree on the checked endpoint sample. Nevertheless, the current aggregate pose ordering differs from the older Table 4 result, which points to a historical evaluator/code-state difference. These pose values should not replace Table 4 until the exact historical evaluation environment is reconciled.
