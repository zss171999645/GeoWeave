# Pi3 / GeoWeave-Pi3 synthetic runtime benchmark 2026-07-04

## 测试口径

- 机器：Baidu AIHC dev machine，单张 NVIDIA A800-SXM4-80GB，`CUDA_VISIBLE_DEVICES=0`。
- 环境：`/mnt/cfs/zhoufeng/env/geoweave-rebuttal/bin/python`。
- 代码：`/mnt/cfs/zhoufeng/workspace/geoweave-rebuttal-repro-20260629`。
- 输入：synthetic tensor，不使用真实图像；张量形状为 `(1, N, 3, 392, 518)`。
- 计时：直接计 `model(imgs)` forward，CUDA event 计时；不包含图片读盘、resize、point interpolation、npz/ply 写盘。
- 精度：A800 上 autocast 使用 bfloat16。
- GeoWeave 加载日志确认 `indexer_state={enabled=True, warmup=False, sparse=True, compute_loss=False, topk=1024}`。

## 结果

| Frames | Model | Warmup | Repeats | Mean seconds | FPS | Peak allocated GiB | Peak reserved GiB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10 | pi3_base | 1 | 5 | 0.305454 | 32.738 | 5.768 | 5.895 |
| 10 | geoweave_pi3 | 1 | 5 | 0.811781 | 12.319 | 5.227 | 5.334 |
| 100 | pi3_base | 1 | 5 | 6.443545 | 15.519 | 9.382 | 11.199 |
| 100 | geoweave_pi3 | 1 | 1 | 33.567129 | 2.979 | 14.530 | 15.109 |

- 10 帧：GeoWeave / Pi3 latency ratio = `2.658x`。
- 100 帧：GeoWeave / Pi3 latency ratio = `5.209x`。这里 Pi3 是 `warmup=1, repeats=5` 均值，GeoWeave 因单次 100 帧已经约 33.6s，只跑了 `warmup=1, repeats=1`。

## 100 帧 cold single-run sanity check

| Frames | Model | Warmup | Repeats | Seconds | FPS | Peak allocated GiB | Peak reserved GiB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 100 | pi3_base | 0 | 1 | 6.997452 | 14.291 | 9.382 | 10.350 |
| 100 | geoweave_pi3 | 0 | 1 | 34.721105 | 2.880 | 14.530 | 15.109 |

Cold single-run 100 帧下，GeoWeave / Pi3 latency ratio = `4.962x`。

## 结论

这个实现和这个 benchmark 口径下，GeoWeave-Pi3 在 10 帧和 100 帧都比 Pi3 base 慢。rebuttal 里不应该声称 GeoWeave 更快；更稳妥的写法是报告 runtime overhead，并明确本文贡献是 selected dependency 对困难 view support 的可靠性，而不是推理加速。

## 原始输出

- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/synthetic_runtime_pi3_geoweave_392x518_summary_20260704/README.md`
- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/synthetic_runtime_pi3_geoweave_392x518_summary_20260704/combined_rows.json`
- `/mnt/cfs/zhoufeng/geoweave_repro_outputs/synthetic_runtime_pi3_geoweave_392x518_summary_20260704/combined_rows.csv`
