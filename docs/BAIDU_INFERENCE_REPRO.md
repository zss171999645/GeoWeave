# Baidu GeoWeave/Pi3 Inference Reproduction

This note records the verified Baidu AIHC smoke inference path for the
GeoWeave rebuttal handoff bundle.

## Paths

- Work copy:
  `/mnt/cfs/zhoufeng/workspace/geoweave-rebuttal-repro-20260629`
- Python:
  `/mnt/cfs/zhoufeng/env/geoweave-rebuttal/bin/python`
- Input images:
  `/mnt/cfs/zhoufeng/geoweave_repro_inputs/scannetpp_firstfig_281ba69af1_a87_b102_20260513`
- Checkpoint:
  `/mnt/cfs/zhoufeng/geoweave_rebuttal_code_weights_20260604/weights/geoweave_paper_handoff_20260602_final/pi3_geoweave_native_sparse_20260505_checkpoint_79/checkpoint_79/pytorch_model.bin`
- Config:
  `/mnt/cfs/zhoufeng/geoweave_rebuttal_code_weights_20260604/weights/geoweave_paper_handoff_20260602_final/pi3_geoweave_native_sparse_20260505_checkpoint_79/config.yaml`

## Command

```bash
cd /mnt/cfs/zhoufeng/workspace/geoweave-rebuttal-repro-20260629
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PWD:$PWD/aidi/third_party/pi3_training" \
/mnt/cfs/zhoufeng/env/geoweave-rebuttal/bin/python \
  tools/run_pi3_geoweave_smoke_infer.py \
  --input-dir /mnt/cfs/zhoufeng/geoweave_repro_inputs/scannetpp_firstfig_281ba69af1_a87_b102_20260513 \
  --output-dir /mnt/cfs/zhoufeng/geoweave_repro_outputs/pi3_geoweave_smoke_scannetpp_20260629_size518 \
  --max-images 10 \
  --load-img-size 518 \
  --device cuda \
  --point-source native \
  --point-stride 8 \
  --verbose
```

## Verified Output

Output directory:

```text
/mnt/cfs/zhoufeng/geoweave_repro_outputs/pi3_geoweave_smoke_scannetpp_20260629_size518
```

Artifacts:

- `points.npz`: full point map and RGB arrays.
- `points_stride8.ply`: sampled preview point cloud.
- `metadata.json`: input, model, CUDA, and output metadata.

Verification:

```text
points_shape (10, 378, 504, 3) float32
colors_shape (10, 378, 504, 3) uint8
finite_ratio 1.0
ply_vertices 30240
load_img_size 518
cuda_device_name NVIDIA A800-SXM4-80GB
```

Model loader log:

```text
indexer_state={enabled=True, warmup=False, sparse=True, compute_loss=False, topk=1024, head_chunk_size=4, score_dtype='float16'}
result=<All keys matched successfully>
```

Notes:

- The `pytorch3d` supervisor import warning is non-fatal for this inference
  path.
- The RoPE2D CUDA extension warning falls back to a slower PyTorch path; the
  inference still completed.
- This smoke run verifies inference on local image assets. It does not reproduce
  paper benchmark metrics because the original benchmark roots under
  `/horizon-bucket` are not mounted on the Baidu development machine.

## DL3DV Sample

Existing DL3DV data from the prior 3DGS-VAE workflow was found at:

```text
/mnt/cfs/liyinglong/data/DL3DV-ALL-960P
```

The 3DGS-VAE prepared manifest records this root at:

```text
/mnt/cfs/zhoufeng/workspace/3DGS-VAE/manifests/dl3dv_summary.json
```

For the sample run, the first 10 frames were extracted from:

```text
/mnt/cfs/liyinglong/data/DL3DV-ALL-960P/10K/9a9d9ad74705265cee75c018beac4ed1c144ae1debf082dc558b3353306eb732.zip
```

Input directory:

```text
/mnt/cfs/zhoufeng/geoweave_repro_inputs/dl3dv_10K_9a9d9ad74705265cee75c018beac4ed1c144ae1debf082dc558b3353306eb732_first10
```

Command:

```bash
cd /mnt/cfs/zhoufeng/workspace/geoweave-rebuttal-repro-20260629
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PWD:$PWD/aidi/third_party/pi3_training" \
/mnt/cfs/zhoufeng/env/geoweave-rebuttal/bin/python \
  tools/run_pi3_geoweave_smoke_infer.py \
  --input-dir /mnt/cfs/zhoufeng/geoweave_repro_inputs/dl3dv_10K_9a9d9ad74705265cee75c018beac4ed1c144ae1debf082dc558b3353306eb732_first10 \
  --output-dir /mnt/cfs/zhoufeng/geoweave_repro_outputs/pi3_geoweave_smoke_dl3dv_10K_9a9d9ad_first10_size518 \
  --max-images 10 \
  --load-img-size 518 \
  --device cuda \
  --point-source native \
  --point-stride 8 \
  --verbose
```

Verified output:

```text
/mnt/cfs/zhoufeng/geoweave_repro_outputs/pi3_geoweave_smoke_dl3dv_10K_9a9d9ad_first10_size518
```

Verification:

```text
points_shape (10, 540, 960, 3) float32
colors_shape (10, 540, 960, 3) uint8
finite_ratio 1.0
ply_vertices 81600
load_img_size 518
cuda_device_name NVIDIA A800-SXM4-80GB
```
