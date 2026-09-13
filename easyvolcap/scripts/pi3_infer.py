from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from easyvolcap.engine.file_client import FileClient
from easyvolcap.utils.pi3.models.pi3 import Pi3
from easyvolcap.utils.pi3.utils.basic import load_images_as_tensor, write_ply
from easyvolcap.utils.pi3.utils.geometry import depth_edge

PI3_WEIGHT_URL = "https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_ckpt_path() -> Path:
    return _repo_root() / "weights" / "pi3" / "model.safetensors"


def _ensure_pi3_weights(ckpt_path: Path) -> Path:
    if ckpt_path.is_file():
        return ckpt_path

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".download")

    client = FileClient.infer_client(uri=PI3_WEIGHT_URL)
    with client.get_local_path(PI3_WEIGHT_URL) as src_path:
        shutil.copyfile(src_path, tmp_path)

    os.replace(tmp_path, ckpt_path)
    return ckpt_path


def _load_pi3_model(ckpt_path: Path, device: torch.device) -> Pi3:
    model = Pi3().to(device).eval()
    if ckpt_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("safetensors is required to load .safetensors checkpoints") from exc
        state_dict = load_file(str(ckpt_path))
    else:
        state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    return model


def _save_outputs(save_path: Path, outputs: Dict[str, torch.Tensor]) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path = save_path.with_name(f"{save_path.stem}_pi3_outputs.npz")
    np_payload = {k: v.detach().cpu().numpy() for k, v in outputs.items()}
    np.savez_compressed(npz_path, **np_payload)
    return npz_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="examples/skating.mp4",
        help="Path to the input image directory or a video file.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="examples/result.ply",
        help="Path to save the output .ply file.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=-1,
        help="Interval to sample image. Default: 1 for images dir, 10 for video",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to the model checkpoint file. Default: None",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith(".mp4") else 1
    print(f"Sampling interval: {args.interval}")

    save_path = Path(args.save_path)
    ckpt_path = Path(args.ckpt) if args.ckpt else _default_ckpt_path()
    ckpt_path = _ensure_pi3_weights(ckpt_path)

    device = torch.device(args.device)

    print("Loading model...")
    model = _load_pi3_model(ckpt_path, device)

    imgs = load_images_as_tensor(args.data_path, interval=args.interval).to(device)
    if imgs.numel() == 0:
        raise RuntimeError(f"No images found at: {args.data_path}")

    print("Running model inference...")
    use_amp = device.type == "cuda"
    dtype = torch.bfloat16 if use_amp and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast("cuda", dtype=dtype):
                outputs = model(imgs[None])
        else:
            outputs = model(imgs[None])

    npz_path = _save_outputs(save_path, outputs)

    masks = torch.sigmoid(outputs["conf"][..., 0]) > 0.1
    non_edge = ~depth_edge(outputs["local_points"][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0].cpu()

    print(f"Saving point cloud to: {save_path}")
    points = outputs["points"][0].detach().cpu()
    colors = imgs.detach().cpu().permute(0, 2, 3, 1)
    write_ply(points[masks], colors[masks], str(save_path))
    print(f"Saved raw outputs to: {npz_path}")
    print("Done.")


if __name__ == "__main__":
    main()
