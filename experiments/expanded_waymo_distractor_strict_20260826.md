# Strict Expanded Waymo Distractor Endpoint

Date: 2026-08-26

## Correction

The first Waymo2 expansion selected the nearest cross-scene same-camera tail but did not enforce the historical relative-similarity condition. That output remains diagnostic only and is excluded here.

The corrected builder requires every candidate to satisfy

`distance(prefix, distractor) < distance(prefix, clean context)`

and ranks eligible candidates with the historical composite feature/structure score. Cross-camera candidates are allowed when image resolution matches. Selection never uses model outputs.

## Construction

- Source: `/mnt/cfs/datasets/Waymo2/testing`.
- Sequence anchors: 80 centered stride-four ten-view windows.
- Eligible positive-margin pairs: 121 from 31 unique clean anchors.
- Selected pairs: 20, including 6 cross-camera pairs.
- Mean clean-context distance: 0.000289.
- Mean distractor distance: 0.000214.
- Mean positive margin: 0.000075.
- Materialized inputs: 20 examples x 5 levels = 100 tuples.
- Prefix hashes, image counts, pose counts, source labels, and model-independent selection flags all passed validation.

## Endpoint result

| Set | Model | ATE 0 | ATE 4 | ATE degradation | RPE-t 0 | RPE-t 4 | RPE-t degradation |
|---|---|---:|---:|---:|---:|---:|---:|
| Original-20 | Pi3 | 0.098607 | 0.229298 | 0.130691 | 0.162358 | 0.323039 | 0.160681 |
| Original-20 | GeoWeave | 0.103073 | 0.175453 | 0.072379 | 0.160971 | 0.235704 | 0.074733 |
| Strict New-20 | Pi3 | 0.072054 | 0.100775 | 0.028721 | 0.121764 | 0.177789 | 0.056026 |
| Strict New-20 | GeoWeave | 0.074566 | 0.114696 | 0.040130 | 0.122992 | 0.176390 | 0.053397 |
| Combined-40 | Pi3 | 0.085330 | 0.165036 | 0.079706 | 0.142061 | 0.250414 | 0.108353 |
| Combined-40 | GeoWeave | 0.088820 | 0.145075 | 0.056255 | 0.141981 | 0.206047 | 0.064065 |

For Strict New-20, the degradation advantage (Pi3 minus GeoWeave) is `-0.011409` for ATE and `+0.002629` for RPE-t. ATE has 10 positive and 10 negative examples, median advantage approximately zero, and paired-bootstrap 95% CI `[-0.056647, 0.020363]`. RPE-t has 13 positive and 7 negative examples, median advantage `+0.018973`, and 95% CI `[-0.031848, 0.032406]`.

Combined-40 still favors GeoWeave on both degradation metrics, but the corrected new set alone has an unfavorable mean ATE and worse absolute ATE at four distractors. Under the user's stricter instruction to stop when a primary metric contradicts the paper claim, the one-, two-, and three-distractor levels and the expansion to 100 were not launched.

## Outputs

- Root: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/expanded_waymo_distractor_strict_20260826`
- Selection manifest: `tuples/selection_manifest.json`
- Protocol: `protocol_selected20.json`
- Predictions: `predictions_selected20`
- Endpoint pose rows: `pose_selected20_endpoints`
- Combined endpoint gate: `gate40_endpoints.json`
