# Business 10-View Badcase Top100 Benchmark

This benchmark fixes the top 100 depth/point-cloud bad cases mined from the
latest registered business VGGT model on the 2000h v2 business validation data.

- Source eval run: `vggt/business_eval/latestbiz_2000hv2_val_10view_seq_sceneidx_badcase_20260526_1851`
- Source metrics: `/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/result/vggt/business_eval/latestbiz_2000hv2_val_10view_seq_sceneidx_badcase_20260526_1851/-0000001/metrics_latestbiz_10view_seq_sceneidx_badcase.json`
- Source pool: 188 deterministic windows from 47 business eval scenes.
- Benchmark size: 100 target images.
- Inference sample: target + 9 same-camera temporal source frames.
- Reproduction config: `configs/exps/vggt/saturnv/evaluation/vggt_business_10view_badcase_top100_lio.yaml`

Scoring used for mining:

`sum z(dpt:abs_rel, dpt:rmse, dpt:l1, -dpt:d1, xyz:l1, xyz:rmse)`

## Latest Business Model Baseline

- Checkpoint: `/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/vggt/finetune_model/basemodel/datascaleup/8x8_2e-5_resumebalancedatano1_2000hv2_use_dptpose_asxyz_liogt/clean.pt`
- Metric source: the frozen top100 rows from the source eval above. The standalone benchmark canary submissions on 2026-05-26 did not write metrics, so the baseline below is computed from the completed mining run with the same checkpoint and samples.
- Samples: 100

| metric | mean | std |
| --- | ---: | ---: |
| `cam:pose_auc_30` | 0.787385 | 0.307045 |
| `cam:pose_auc_20` | 0.744656 | 0.310496 |
| `cam:pose_auc_10` | 0.629267 | 0.322742 |
| `cam:pose_auc_05` | 0.472178 | 0.352500 |
| `cam:pose_auc_03` | 0.377556 | 0.359972 |
| `cam:pose_auc_01` | 0.254000 | 0.358698 |
| `cam:RPE_trans` | 1.115188 | 3.164818 |
| `cam:RPE_rot` | 2.955826 | 7.832369 |
| `dpt:abs_rel` | 0.425203 | 0.119001 |
| `dpt:rmse` | 0.325563 | 0.080939 |
| `dpt:l1` | 0.164896 | 0.032407 |
| `dpt:d1` | 0.374789 | 0.087007 |
| `xyz:l1` | 0.085781 | 0.021941 |
| `xyz:rmse` | 0.181409 | 0.044610 |

## VGGT-Omega 1B

- Checkpoint: `/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/checkpoints/vggt_omega_1b_512.pt`
- Runner: `aidi/scripts/vggt/eval_vggt_omega_business_10view_badcase.py`
- Result: `/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/result/vggt_omega/business_10view_badcase_top100/vggt_omega_business_10view_badcase_top100_full_20260527_0049/run_summary.json`
- Samples: 100
- Omega settings: `resolution=512`, `mode=balanced`, `global_attention_mode=original`.

| metric | mean | std |
| --- | ---: | ---: |
| `cam:pose_auc_30` | 0.643881 | 0.317894 |
| `cam:pose_auc_20` | 0.532267 | 0.370623 |
| `cam:pose_auc_10` | 0.381267 | 0.417174 |
| `cam:pose_auc_05` | 0.317422 | 0.419521 |
| `cam:pose_auc_03` | 0.293630 | 0.420332 |
| `cam:pose_auc_01` | 0.264667 | 0.419313 |
| `cam:translation_scale` | 1.816729 | 1.425751 |
| `dpt:abs_rel` | 0.426674 | 0.128777 |
| `dpt:rmse` | 0.323656 | 0.082936 |
| `dpt:l1` | 0.164511 | 0.033942 |
| `dpt:d1` | 0.370227 | 0.085549 |
| `xyz:l1` | 0.087452 | 0.020433 |
| `xyz:rmse` | 0.181421 | 0.043703 |

Against the latest business baseline on this hard set, VGGT-Omega is similar on
depth and xyz but materially worse on pose AUC.

## VGGT Official 1B

- Checkpoint components: `/horizon-bucket/saturn_v_dev/01_users/tao02.xie/projects/GFM/gfm/data/trained_model/vggt/official/VGGT-1B/{aggregator,camera,point,depth,track}.pt`
- Runner: `aidi/scripts/vggt/eval_official_vggt_sparse_datasets.sh` with `OFFICIAL_BASELINE=1`
- Result: `/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/result/vggt/official/business_10view_badcase_top100_full_20260527_0122_ckptVGGT1B_original/-0000001/metrics_vggt_business_10view_badcase_top100_lio.json`
- Samples: 100
- Eval settings: `VAL_N_SRCS=9`, `VAL_MAX_ITER=100`, `force_xyz_from_depth=True`, camera/depth/xyz metrics only.

| metric | mean | std |
| --- | ---: | ---: |
| `cam:pose_auc_30` | 0.604933 | 0.322787 |
| `cam:pose_auc_20` | 0.510356 | 0.332676 |
| `cam:pose_auc_10` | 0.343022 | 0.331353 |
| `cam:pose_auc_05` | 0.224756 | 0.299710 |
| `cam:pose_auc_03` | 0.165481 | 0.275614 |
| `cam:pose_auc_01` | 0.087111 | 0.216998 |
| `cam:translation_scale` | 8.496890 | 39.065730 |
| `dpt:abs_rel` | 0.443015 | 0.147917 |
| `dpt:rmse` | 0.317126 | 0.082418 |
| `dpt:l1` | 0.162219 | 0.042388 |
| `dpt:d1` | 0.361368 | 0.085471 |
| `xyz:l1` | 0.088151 | 0.022990 |
| `xyz:rmse` | 0.183439 | 0.044257 |

Official VGGT-1B is close on aligned depth/xyz, but pose is clearly below the
business baseline and below VGGT-Omega on this hard set.

## Depth Anything 3 Small

- Model: `/home/feng01.zhou/workspace/meshx_pi3_depthgt79_eval_2cec9d45_20260512_2223/tmp/pretrained_official/depth-anything__DA3-SMALL`
- Repo: `/home/feng01.zhou/workspace/meshx_pi3_depthgt79_eval_2cec9d45_20260512_2223/third_party/Depth-Anything-3`
- Runner: `aidi/scripts/vggt/eval_depthanything3_business_10view_badcase.py`
- Result: `/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/result/depthanything3/business_10view_badcase_top100/depthanything3_business_10view_badcase_top100_full_20260527_0122/run_summary.json`
- Samples: 100
- DA3 settings: `process_res=504`, `process_res_method=upper_bound_resize`, `ref_view_strategy=first`, `use_ray_pose=False`.
- Eval settings: depth uses DA3 prediction; camera uses DA3 extrinsics/intrinsics; xyz is rebuilt from predicted depth + predicted camera.

| metric | mean | std |
| --- | ---: | ---: |
| `cam:pose_auc_30` | 0.542837 | 0.309316 |
| `cam:pose_auc_20` | 0.438844 | 0.309400 |
| `cam:pose_auc_10` | 0.262422 | 0.277263 |
| `cam:pose_auc_05` | 0.137333 | 0.209855 |
| `cam:pose_auc_03` | 0.081407 | 0.160683 |
| `cam:pose_auc_01` | 0.027111 | 0.087409 |
| `cam:translation_scale` | 7.182323 | 18.882503 |
| `dpt:abs_rel` | 0.396135 | 0.076945 |
| `dpt:rmse` | 0.320287 | 0.086228 |
| `dpt:l1` | 0.159809 | 0.036430 |
| `dpt:d1` | 0.387377 | 0.067885 |
| `xyz:l1` | 0.086815 | 0.020386 |
| `xyz:rmse` | 0.181146 | 0.042917 |

DA3-SMALL has the best depth/xyz numbers among the external references on this
aligned hard set, but its pose is the weakest. It is useful as a depth reference,
not as a full 3D reconstruction replacement baseline.
