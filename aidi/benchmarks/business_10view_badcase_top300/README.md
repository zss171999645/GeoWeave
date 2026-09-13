# Business 10-View Badcase Top300 Benchmark

This benchmark extends the Top100 business 10-view badcase benchmark to 300
target images. The inference sample is unchanged: target + 9 same-camera
temporal source frames.

- Source eval run A: `vggt/business_eval/latestbiz_2000hv2_val_10view_seq_sceneidx_badcase_20260526_1851`
- Source metrics A: `/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/result/vggt/business_eval/latestbiz_2000hv2_val_10view_seq_sceneidx_badcase_20260526_1851/-0000001/metrics_latestbiz_10view_seq_sceneidx_badcase.json`
- Source eval run B: `vggt/business_eval/latestbiz_2000hv2_val_10view_seq_sceneidx_badcase300_candidates_20260527_1439`
- Source metrics B: `/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/result/vggt/business_eval/latestbiz_2000hv2_val_10view_seq_sceneidx_badcase300_candidates_20260527_1439/-0000001/metrics_latestbiz_10view_seq_sceneidx_badcase300_candidates.json`
- Candidate pool: 658 raw metric rows, 470 unique target windows after de-dup.
- Benchmark size: 300 target images.
- Previous Top100 retained in Top300: 100 / 100.
- Reproduction config: `configs/exps/vggt/saturnv/evaluation/vggt_business_10view_badcase_top300_lio.yaml`
- Preview artifacts: `/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/badcases/latestbiz_2000hv2_val_10view_seq_sceneidx_badcase_top300_20260527_1439`

Scoring used for mining:

`sum z(dpt:abs_rel, dpt:rmse, dpt:l1, -dpt:d1, xyz:l1, xyz:rmse)`

## Latest Business Model Baseline

- Checkpoint: `/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/vggt/finetune_model/basemodel/datascaleup/8x8_2e-5_resumebalancedatano1_2000hv2_use_dptpose_asxyz_liogt/clean.pt`
- Metric source: the frozen top300 rows from the merged candidate pool.
- Samples: 300

| metric | mean | std |
| --- | ---: | ---: |
| `cam:pose_auc_30` | 0.755677 | 0.331497 |
| `cam:pose_auc_20` | 0.711804 | 0.335077 |
| `cam:pose_auc_10` | 0.596615 | 0.342325 |
| `cam:RPE_trans` | 1.537771 | 3.905349 |
| `cam:RPE_rot` | 3.846690 | 10.382176 |
| `dpt:abs_rel` | 0.411531 | 0.113526 |
| `dpt:rmse` | 0.309754 | 0.077398 |
| `dpt:l1` | 0.156305 | 0.031581 |
| `dpt:d1` | 0.386352 | 0.081284 |
| `xyz:l1` | 0.081555 | 0.020106 |
| `xyz:rmse` | 0.172376 | 0.042480 |

## Exported Files

- `target_img_list.txt`: frozen benchmark target list in repo.
- `top300_10view_badcases.csv`: ranked badcase table in bucket preview dir.
- `top300_10view_badcases.json`: ranked records with score parts and sample frames.
- `top300_10view_badcases.md`: contact-sheet markdown preview.
- `top300_10view_badcases.pdf`: 300-page contact-sheet PDF.
