# Training GeoWeave

Both trainers are included. The native Pi3 tree under
`aidi/third_party/pi3_training/` contains the model, training loop, optimizer,
losses, dataset adapters, and Hydra configs. VGGT uses
`easyvolcap/models/official_vggt_model.py`, `easyvolcap/runners/`,
`easyvolcap/official_vggt/training/`, and `configs/`.

## Environment

Start with the inference environment, then install:

```bash
python -m pip install -r requirements-training.txt -r requirements-kernels.txt
```

Run training on Linux with CUDA GPUs. The launch commands below execute locally;
they do not submit a job to a company service. On a managed cluster, put the same
command in your scheduler job. Set `CUDA_VISIBLE_DEVICES` to select GPUs.

## Pi3: warm-up and sparse training

The provided local example uses **processed TarTanAir**, with this layout:

```text
/path/to/tartanair/<scene>/<Easy-or-Hard>/<track>/
  intri.yml
  extri.yml
  images/<frame_id>/<cam_id>.png
  depths/<frame_id>/<cam_id>.exr
```

The depth maps and camera calibration must match the images. The exact adapter is
`datasets/meshx_tartanair_dataset.py` in the native Pi3 tree. Raw dataset downloads
are not interchangeable with this processed format.

```bash
# Indexer warm-up: initialize the backbone from the original Pi3 checkpoint.
python train.py --model pi3 --stage warmup --gpus 2 \
  --checkpoint /path/to/Pi3_model.safetensors \
  --data-root /path/to/tartanair --output runs/pi3-warmup

# Sparse stage: initialize from YOUR warm-up output.
python train.py --model pi3 --stage sparse --gpus 2 \
  --checkpoint /path/to/warmup/checkpoint_N/pytorch_model.bin \
  --data-root /path/to/tartanair --output runs/pi3-sparse
```

Arguments after `--` are passed directly as Hydra overrides. For example:

```bash
python train.py --model pi3 --stage sparse --gpus 1 \
  --checkpoint /path/to/warmup/pytorch_model.bin \
  --data-root /path/to/tartanair --output runs/pi3-small \
  -- train.num_epoch=1 train.iters_per_epoch=2 train.num_workers=0 \
     model.indexer_cfg.indexer_layers=9-17 model.indexer_cfg.topk=1024
```

This short run is a startup check, not reproduction of the final paper training
schedule. The default native stage configs are low-resolution recipes. Exact
paper recipe parameters and mixed-dataset settings are retained in
`release/geoweave_paper_handoff/scripts/train_pi3.sh` and the underlying
`aidi/scripts/pi3/` scripts. Those historical wrappers contain internal default
paths; use them as recipe references, not as the public local launcher.

For the full mixed-dataset recipe, choose the native `data=meshx_pi3_vggt17`
config and provide every dataset root used by that config. `--data-root` sets
`TARTANAIR_ROOT`; it does not relocate all other datasets. Original warm-up and
stage-two initializations are distinct. Keeping the same architecture/config
with its weights is necessary when training a variant.

## VGGT

Pass a **complete** training YAML with your dataset paths. The original full
record is
`aidi/configs/vggt/finetune_5090_sparse_topk2048_l9_19_p34_record.yaml`.
Copy it to your own config, set `model_cfg.vggt_cfg.indexer_cfg.topk: 1024`
for the final GeoWeave variant, and replace the dataset paths. Inspect:

- `dataloader_cfg.dataset_cfg.meta_roots` and corresponding `metaset_cfgs`;
- `val_dataloader_cfg.dataset_cfg` for validation data;
- per-module initialization paths (`agg_ckpt`, `cam_ckpt`, `xyz_ckpt`,
  `dpt_ckpt`, `tra_ckpt`) in `model_cfg`: clear these when initializing from a
  complete training checkpoint through `--checkpoint`;
- indexer warm-up/sparse settings, optimizer, and scheduler for your stage.

```bash
python train.py --model vggt --gpus 2 \
  --config /path/to/local-training.yaml \
  --checkpoint /path/to/full-training-checkpoint.pt --output runs/vggt
```

The checkpoint must contain the training wrapper's `model` state dictionary
(as `geoweave_vggt.pth` does). This launcher initializes model weights and starts a fresh
training schedule; it does not silently restore the old optimizer or epoch.
The launcher sets native distributed mode from `--gpus` and disables the separate
Accelerate mode, so multiple workers participate in one training run.
VGGT overrides after `--` use the original config syntax, e.g.
`runner_cfg.epochs=1 runner_cfg.ep_iter=2`. Additional dataset-specific optional
dependencies may be needed for adapters beyond the paper's processed RGB-D data.

The earlier paper YAML under `configs/exps/` includes experimental variants with
different indexer head counts. Do not pair a different architecture with the final
checkpoint. The inference presets are extracted from the final checkpoint's
recorded model configuration, not guessed from an experiment filename.

## Inspect before launching

Append `--print-command` to either public training command to see its exact
working directory and launcher command without starting training. For Pi3,
Hydra's `--cfg job --resolve` on the native trainer prints the composed config.
Full training requires real data, initialization weights, and GPU time; an import
check or config expansion is not proof of a successful training run.
