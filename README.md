# GeoWeave

GeoWeave inference, training, and custom CUDA/Triton kernels for the paper's
VGGT and Pi3 models. The inference entry point uses the **original trained
architectures**, with the final model configurations included in this repository.
No company account, bucket mount, or separate base-model download is needed for
inference once you have the corresponding full checkpoint and input images.

## Install

Use Python 3.10 or 3.11 in a fresh environment. Run commands from this repository root.
The reference environment uses PyTorch 2.4.1 / torchvision 0.19.1:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-inference.txt -r requirements-kernels.txt
```

For accelerated inference, use Linux and an NVIDIA GPU supported by this PyTorch
build. PyTorch supplies Triton. Optional C++/CUDA extensions also require a local
CUDA toolkit (`nvcc`, matching PyTorch's CUDA version), a C++ compiler, and Ninja.
CPU reference inference is available with a CPU PyTorch installation and
`--device cpu --backend torch`; it is much slower and does not run the kernels.
Do not install the legacy full `requirements.txt` just to run inference.

## Inference with your weights

**Pretrained GeoWeave weights will be released in a future update. Download links
will be added here once they are available.**

| Model | Supported final checkpoint | Included preset |
| --- | --- | --- |
| Pi3 | `geoweave_pi3.pth` | `geoweave/configs/pi3.json`: top-k 1024, global layers 9–17 |
| VGGT | `geoweave_vggt.pth` | `geoweave/configs/vggt.json`: top-k 1024, global layers 9–19 |

Full matching state dictionaries in `.safetensors` are also accepted. These are
GeoWeave checkpoints; an unmodified upstream VGGT/Pi3 checkpoint lacks trained
indexer weights. Loading is strict: missing, extra, or incompatible parameters
cause an error rather than inference with randomly initialized layers. Only load
PyTorch `.pt`/`.pth`/`.bin` files from a source you trust.

```bash
python infer.py --model pi3 \
  --checkpoint /path/to/geoweave_pi3.pth \
  --images /path/to/images --output outputs/pi3

python infer.py --model vggt \
  --checkpoint /path/to/geoweave_vggt.pth \
  --images /path/to/images --output outputs/vggt
```

Images are naturally sorted by filename (`frame2` before `frame10`). Default
resize width is 518 pixels. Pi3 uses the first image's aspect ratio, as in the
paper evaluator; VGGT uses its original crop preprocessing. Use a new output
directory for each run. `--max-images` limits the number of views when memory is
limited; it changes the input context and therefore the predictions.

Outputs:

- `predictions.npz`: all tensor predictions and the actual preprocessed images.
  Pi3 includes `points`, `local_points`, `camera_poses`; VGGT includes
  `world_points`, `depth`, `pose_enc`, `extrinsics` and `intrinsics`.
- `points.ply`: colored point cloud, sampled every 4 pixels by default
  (`--point-stride 1` exports all points). Colors match the preprocessed images.
- `metadata.json`: ordered inputs, resolved model config, runtime/backend/dtype,
  output shapes, and single-forward elapsed time (not a benchmark result).

Pi3 camera poses are camera-to-world; VGGT extrinsics are world-to-camera.
Outputs retain the model's relative coordinate system and scale. Running on
arbitrary images is inference reproduction; reproducing paper tables additionally
requires the corresponding benchmark data, ground truth, and evaluation protocol.
See the existing evaluation tools under `aidi/scripts/` and `tools/`.

## Training

The repository contains both complete training implementations, including Pi3's
native trainer, datasets, losses, and configs, plus VGGT's EasyVolcap trainer.
See [training instructions](docs/TRAINING.md) for data layout, initialization,
local launch commands, and the original warm-up / sparse-training recipes.
Training requires datasets and initialization weights in addition to this code.

## Our acceleration kernels

Sources are included under:

- `easyvolcap/utils/custom_indexer/`: Triton top-k/indexer and KL kernels,
  CUDA exact top-k, fused score/top-k and merge kernels in `csrc/`.
- `easyvolcap/utils/custom_flash_attn/`: Triton sparse attention forward/backward
  and grouped CUDA backward in `csrc/`.

```bash
python -m geoweave.kernels              # sparse attention forward/backward check
python -m geoweave.kernels --build-all  # also compile optional CUDA extensions
```

See [kernel details](docs/KERNELS.md) for supported paths and verification coverage.
The [release validation record](docs/VALIDATION.md) lists the real-checkpoint
inference checks and the GPU checks that have not yet been run.

## Attribution

This repository retains the original upstream copyright and license notices.
The code builds on VGGT, Pi3, DINOv2, and EasyVolcap. See `license` and the licenses
and source headers in the respective directories; the repository is not being
relicensed by this release.
