# Original-Group-A Anchored ScanNet++ Overlap Sweep

Date: 2026-08-26

## Execution record

- Branch: `revision-required-metrics-20260824`
- Mapping implementation: commits `2136c0a` and `30ce63f`
- Fixed-anchor/quantile implementation: commits `b266ae1`, `d7d18f5`, and `db4f216`
- Aggregation implementation: commit `ba2b63c`
- Direct runtime: common-dev; no AIHC jobs.
- Legacy source: `/mnt/cfs/zhoufeng/rebuttal_received_20260701/extracted/scannetpp_slight_overlap5x2_smoke_538c3f3a_20260513_0441`
- Official source: `/mnt/cfs/datasets/scannetpp/scannetpplus/Scannetpp/data`
- Final output: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/original_group_a_overlap_sweep_20260826`

## Validation

- Mapping: 60/60 images, 12/12 five-view anchors, six original scenes.
- Pose-guided mappings: 20 with near-zero scene transform residual.
- SIFT mappings: 40 with at least 23 good local matches; selected candidates rank at most four after one-to-one assignment.
- Initial absolute-band preflight: 8/12 anchors; retained as a failed diagnostic.
- Final quantile preflight: 12/12 anchors, 3,248 candidates, 48 selected rows.
- Materialization: 12 anchors, 48 tuples, fixed Group A hashes across all four levels.
- Inference: 48/48 finite Pi3 and 48/48 finite GeoWeave camera caches at 512px.
- Evaluation: 48/48 finite pose rows per model.
- Aggregation: four ordered measured-overlap levels, 10,000 paired-bootstrap resamples, seed `20260826`.

## Result

Mean measured overlap is `0.4406/0.3351/0.2132/0.0444`. Pi3/GeoWeave mean ATE is `0.1859/0.1409`, `0.1776/0.1461`, `0.2459/0.2092`, and `0.3593/0.1693`. At the lowest level, maximum ATE is `2.531/0.366`, GeoWeave wins 7/12 anchors, mean delta ATE is `+0.1900`, and the paired interval is `[-0.0067, 0.5642]`. The manuscript reports the strong failure-tail reduction together with the crossing confidence interval and mixed medians.
