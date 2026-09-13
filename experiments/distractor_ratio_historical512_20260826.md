# Controlled Distractor-Ratio Sweep at Historical 512px Setting

Date: 2026-08-26

## Purpose

Re-evaluate the existing 0/25/50/75/100% context-distractor sweep using the 512-pixel input size recorded by the historical Table 4 evaluation script. This resolves the endpoint-order discrepancy in the earlier 518-pixel sweep before manuscript integration.

## Execution

- Machine: `common-dev`, direct SSH execution; no AIHC job.
- Repository branch: `revision-required-metrics-20260824`.
- Protocol: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/revision_distractor_ratio_20260824/protocol.json`.
- Output: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/revision_distractor_ratio_historical512_20260826`.
- Models: Pi3 paper checkpoint and GeoWeave-Pi3 checkpoint 79.
- Views: ten input images; only fixed prefix indices 0--5 evaluated.
- Completeness: 100/100 finite camera caches and 100/100 finite pose rows per model.
- Evaluator dependencies: repository cached-pose evaluator with isolated `evo==1.31.1` runtime.

## Result

| Context distractors | Pi3 ATE | GeoWeave ATE | Pi3 RPE-t | GeoWeave RPE-t |
|---:|---:|---:|---:|---:|
| 0% | 0.098607 | 0.103073 | 0.162358 | 0.160971 |
| 25% | 0.092542 | 0.103721 | 0.156706 | 0.160876 |
| 50% | 0.091497 | 0.086720 | 0.153843 | 0.141060 |
| 75% | 0.088678 | 0.081639 | 0.154248 | 0.136262 |
| 100% | 0.229298 | 0.175453 | 0.323039 | 0.235704 |

The 0--75% trend is non-monotonic. GeoWeave is not uniformly better at every ratio, but is lower on both pose metrics from 50% onward. At 100%, clean-relative ATE degradation is 0.130691 for Pi3 and 0.072379 for GeoWeave; RPE-t degradation is 0.160681 and 0.074733. These endpoints closely reproduce the main-paper evaluation, with remaining unified-rerun differences of at most about 0.004 recorded explicitly.

## Invalid diagnostic excluded

A separate end-to-end endpoint diagnostic was initially launched without the historical `--eval-frame-indices 0,1,2,3,4,5` argument. It evaluated all ten poses and produced incomparable values, including very large errors for cross-scene tuples. Those outputs under `revision_distractor_ratio_historical_direct_20260826` are excluded from all summaries and manuscript tables.
