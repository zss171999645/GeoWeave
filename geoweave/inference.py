"""Weight-only inference using the original paper architectures."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def collect_images(directory: Path) -> list[Path]:
    paths = [p for p in directory.iterdir()
             if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]
    def natural_key(path):
        return [int(s) if s.isdigit() else s.lower()
                for s in re.split(r'(\d+)', path.name)]
    paths.sort(key=lambda p: (natural_key(p), p.name))
    if not paths:
        raise FileNotFoundError(f'No images found in {directory}')
    return paths


def normalize_state_dict(state: dict, model: str) -> dict:
    state = dict(state)
    for prefix in ('module.', '_orig_mod.', 'vggt.' if model == 'vggt' else ''):
        if prefix and any(k.startswith(prefix) for k in state):
            if not all(k.startswith(prefix) for k in state):
                raise ValueError(f'Mixed checkpoint prefix {prefix!r}; refusing to discard weights')
            state = {k[len(prefix):]: v for k, v in state.items()}
    if model != 'vggt':
        return state
    suffixes = ('camera_token', 'register_token', 'patch_embed.cls_token',
                'patch_embed.pos_embed', 'patch_embed.register_tokens', 'patch_embed.mask_token')
    for old in suffixes:
        head, _, tail = old.rpartition('.')
        new = (head + '.' if head else '') + 'special_tokens.' + tail
        for key in list(state):
            if key.endswith(old) and not key.endswith(new):
                target = key[:-len(old)] + new
                if target in state:
                    raise ValueError(f'Checkpoint key collision: {key} -> {target}')
                state[target] = state.pop(key)
    return state


def load_model(model_name: str, checkpoint: Path, device, backend: str):
    import torch
    config_path = Path(__file__).with_name('configs') / f'{model_name}.json'
    config = json.loads(config_path.read_text())
    # A reference backend keeps the same top-k sparse method, without CUDA kernels.
    if backend == 'torch':
        config['indexer_cfg'].update(use_topk_kernel=False, use_sparse_flash_attn=False)
        config['indexer_cfg']['score_dtype'] = 'float32'
        if model_name == 'pi3':
            config.update(encoder_attn_backend='sdpa', decoder_attn_backend='sdpa')
    if model_name == 'pi3':
        native = ROOT / 'aidi/third_party/pi3_training'
        sys.path.insert(0, str(native))
        from pi3.models.pi3_training import Pi3
        model = Pi3(**config)
    else:
        from easyvolcap.official_vggt.models.vggt import VGGT
        from easyvolcap.official_vggt.layers.attention import set_xformers_enabled
        set_xformers_enabled(False)
        model = VGGT(**config)
    if checkpoint.suffix == '.safetensors':
        from safetensors.torch import load_file
        state = load_file(str(checkpoint))
    else:
        # Paper .pt files include optimizer metadata: load only checkpoints you trust.
        state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        for key in ('state_dict', 'model', 'module'):
            if isinstance(state.get(key), dict):
                state = state[key]
                break
    state = normalize_state_dict(state, model_name)
    model.load_state_dict(state, strict=True)
    del state
    model = model.to(device).eval()
    indexer_state = dict(config['indexer_cfg'], enabled=True, warmup=False,
                         sparse=True, compute_loss=False)
    if model_name == 'pi3':
        model.set_indexer_state(indexer_state)
    else:
        model.aggregator.set_indexer_state(indexer_state)
    return model, config


def preprocess(paths: list[Path], model: str, size: int):
    import numpy as np
    import torch
    from PIL import Image
    if model == 'vggt':
        from easyvolcap.official_vggt.utils.load_fn import load_and_preprocess_images
        return load_and_preprocess_images([str(p) for p in paths], mode='crop', target_size=size)[None]
    # Identical to the Pi3 paper evaluator: resize all views to first-view aspect.
    with Image.open(paths[0]) as first:
        height = max(14, round(first.height * size / first.width / 14) * 14)
    arrays = []
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert('RGB').resize((size, height), Image.Resampling.LANCZOS)
            arrays.append(np.asarray(rgb).copy())
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float()[None] / 255


def save_predictions(output: Path, predictions: dict, images, model: str, stride: int):
    import numpy as np
    import torch
    arrays = {k: v.detach().float().cpu().numpy() for k, v in predictions.items()
              if torch.is_tensor(v)}
    if not arrays or any(not np.isfinite(v).all() for v in arrays.values()):
        raise RuntimeError('Model returned empty or non-finite predictions')
    arrays['images'] = images.detach().float().cpu().numpy()
    np.savez_compressed(output / 'predictions.npz', **arrays)
    key = 'points' if model == 'pi3' else 'world_points'
    xyz = arrays[key][0, :, ::stride, ::stride].reshape(-1, 3)
    rgb = (arrays['images'][0].transpose(0, 2, 3, 1)[:, ::stride, ::stride] * 255)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8).reshape(-1, 3)
    records = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                      ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for i, name in enumerate(('x', 'y', 'z')):
        records[name] = xyz[:, i]
    for i, name in enumerate(('red', 'green', 'blue')):
        records[name] = rgb[:, i]
    with (output / 'points.ply').open('wb') as handle:
        header = ('ply\nformat binary_little_endian 1.0\n' + f'element vertex {len(xyz)}\n'
                  'property float x\nproperty float y\nproperty float z\n'
                  'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        handle.write(header.encode())
        handle.write(records.tobytes())
    return {k: list(v.shape) for k, v in arrays.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['pi3', 'vggt'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-size', type=int, default=518, help='Resize width; multiple of 14')
    parser.add_argument('--max-images', type=int, default=0, help='0 uses all images')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--backend', choices=['auto', 'cuda', 'torch'], default='auto')
    parser.add_argument('--dtype', choices=['auto', 'float32', 'float16', 'bfloat16'], default='auto')
    parser.add_argument('--point-stride', type=int, default=4)
    parser.add_argument('--threads', type=int, default=8, help='CPU threads')
    args = parser.parse_args()
    if args.image_size < 28 or args.image_size % 14:
        parser.error('--image-size must be a multiple of 14 and at least 28')
    if args.max_images < 0 or args.point_stride < 1 or args.threads < 1:
        parser.error('Invalid image limit, point stride, or CPU thread count')
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint does not exist: {args.checkpoint}')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty; use a new directory for each run')
    paths = collect_images(args.images)
    if args.max_images:
        paths = paths[:args.max_images]
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = torch.device(args.device)
    if device.type not in ('cuda', 'cpu'):
        parser.error('Supported devices are cpu and cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu --backend torch for reference inference')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    backend = ('cuda' if device.type == 'cuda' else 'torch') if args.backend == 'auto' else args.backend
    if backend == 'cuda':
        if device.type != 'cuda':
            parser.error('--backend cuda requires a CUDA device')
        import triton  # Fail explicitly instead of silently losing acceleration.
        from geoweave.kernels import check_extensions
        check_extensions()
    dtype_name = ('bfloat16' if device.type == 'cuda' and torch.cuda.is_bf16_supported()
                  else 'float16' if device.type == 'cuda' else 'float32') if args.dtype == 'auto' else args.dtype
    if device.type == 'cpu' and dtype_name != 'float32':
        parser.error('CPU reference inference requires float32')
    dtype = getattr(torch, dtype_name)
    if backend == 'cuda' and dtype == torch.float32:
        parser.error('Custom sparse attention requires float16/bfloat16; use --backend torch for float32')
    print(f'Loading {args.model} checkpoint (strict) on {device}, backend={backend}', flush=True)
    model, config = load_model(args.model, args.checkpoint, device, backend)
    images = preprocess(paths, args.model, args.image_size).to(device)
    autocast = torch.autocast(device_type='cuda', dtype=dtype) if device.type == 'cuda' and dtype != torch.float32 else nullcontext()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode(), autocast:
        predictions = model(images)
        if args.model == 'vggt':
            from easyvolcap.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
            extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], images.shape[-2:])
            predictions.update(extrinsics=extrinsics, intrinsics=intrinsics)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    shapes = save_predictions(args.output, predictions, images, args.model, args.point_stride)
    metadata = dict(model=args.model, checkpoint=str(args.checkpoint.resolve()),
                    images=[str(p.resolve()) for p in paths], model_config=config,
                    device=str(device), backend=backend, dtype=dtype_name,
                    torch_version=torch.__version__, inference_seconds=elapsed, outputs=shapes,
                    image_size=args.image_size, point_stride=args.point_stride,
                    checkpoint_bytes=args.checkpoint.stat().st_size,
                    kernel_environment={k: v for k, v in os.environ.items()
                                        if k.startswith(('VGGT_', 'PI3_'))})
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved predictions.npz, points.ply, metadata.json to {args.output} ({elapsed:.2f}s)', flush=True)


if __name__ == '__main__':
    main()
