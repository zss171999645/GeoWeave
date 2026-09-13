# Expanded Waymo Distractor Sweep

Date: 2026-08-26

## Purpose

Expand the historical 20-example Waymo distractor-context stress test with model-independent examples from `Waymo2/testing`. The manuscript was not modified.

## Construction

- Source: `/mnt/cfs/datasets/Waymo2/testing`.
- Candidate anchors: one centered stride-four ten-view window from each of 80 per-camera testing sequences.
- Distractor selection: closest low-resolution RGB tail from a different driving segment and the same camera, subject to the frozen entropy and edge-density filters.
- Model outputs were not used for selection.
- Selected expansion: top 20 candidate pairs.
- Endpoint inputs: `noise0` and `noise4`, with the first six clean views fixed and evaluated.
- Inference: Pi3 and GeoWeave-Pi3 paper checkpoints, 512-pixel loading width, camera-only caches.
- Evaluation: cached-pose evaluator with `evo==1.31.1`, using tuple metadata indices `[0,1,2,3,4,5]`.

All 100 materialized five-level tuples passed image-count, pose-count, source-label, and fixed-prefix hash validation. Both endpoint models produced 40/40 finite camera caches and 40/40 finite pose rows.

## Endpoint result

| Set | Model | ATE 0 | ATE 4 | ATE degradation | RPE-t 0 | RPE-t 4 | RPE-t degradation |
|---|---|---:|---:|---:|---:|---:|---:|
| Original-20 | Pi3 | 0.098607 | 0.229298 | 0.130691 | 0.162358 | 0.323039 | 0.160681 |
| Original-20 | GeoWeave | 0.103073 | 0.175453 | 0.072379 | 0.160971 | 0.235704 | 0.074733 |
| New-20 | Pi3 | 0.050200 | 0.096285 | 0.046085 | 0.077924 | 0.147163 | 0.069239 |
| New-20 | GeoWeave | 0.044511 | 0.091651 | 0.047141 | 0.073633 | 0.146323 | 0.072690 |
| Combined-40 | Pi3 | 0.074403 | 0.162791 | 0.088388 | 0.120141 | 0.235101 | 0.114960 |
| Combined-40 | GeoWeave | 0.073792 | 0.133552 | 0.059760 | 0.117302 | 0.191013 | 0.073712 |

The New-20 degradation advantages, defined as Pi3 degradation minus GeoWeave degradation, are `-0.001056` ATE and `-0.003451` RPE-t. Both are non-positive, so the predeclared endpoint gate returns `stop`. Although Combined-40 still favors GeoWeave (`+0.028628` ATE degradation advantage and `+0.041248` RPE-t degradation advantage), the intermediate levels and expansion to 100 were not launched because the new set alone failed both metrics.

## Outputs

- Root: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/expanded_waymo_distractor_sweep_20260826`
- Selection: `tuples/selection_manifest.json`
- Protocol: `protocol_selected20.json`
- Predictions: `predictions_selected20`
- Endpoint pose rows: `pose_selected20_endpoints`
- Gate decision: `gate40_endpoints.json`
