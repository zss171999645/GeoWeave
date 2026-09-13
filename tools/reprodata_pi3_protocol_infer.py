#!/usr/bin/env python3
"""Run Pi3 or GeoWeave-Pi3 over a reproduced-data protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from PIL import Image


DEFAULT_WEIGHTS_ROOT = Path(
    "/mnt/cfs/zhoufeng/geoweave_rebuttal_code_weights_20260604/"
    "weights/geoweave_paper_handoff_20260602_final"
)
DEFAULT_BASE_CKPT = DEFAULT_WEIGHTS_ROOT / "pi3_base_yyfz233" / "Pi3_model.safetensors"
DEFAULT_GEOWEAVE_ROOT = DEFAULT_WEIGHTS_ROOT / "pi3_geoweave_native_sparse_20260505_checkpoint_79"
DEFAULT_GEOWEAVE_CKPT = DEFAULT_GEOWEAVE_ROOT / "checkpoint_79" / "pytorch_model.bin"
DEFAULT_GEOWEAVE_CONFIG = DEFAULT_GEOWEAVE_ROOT / "config.yaml"
DEFAULT_VARIANTS = ("clean_tail", "plausible_noise_tail")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=["base", "geoweave"], required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--variants", nargs="*", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--load-img-size", type=int, default=518)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--point-source", choices=["native", "depth_pose"], default="native")
    parser.add_argument("--point-stride", type=int, default=16)
    parser.add_argument("--camera-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def default_model_name(mode: str) -> str:
    return "pi3_base" if mode == "base" else "geoweave_pi3"


def default_checkpoint(mode: str) -> Path:
    return DEFAULT_BASE_CKPT if mode == "base" else DEFAULT_GEOWEAVE_CKPT


def iter_protocol_jobs(
    protocol: dict[str, Any],
    *,
    output_root: Path,
    model_name: str,
    variants: Sequence[str],
    sample_limit: int,
) -> Iterator[tuple[str, str, Path, Path]]:
    samples = list(protocol.get("samples", []))
    if sample_limit > 0:
        samples = samples[:sample_limit]
    for sample in samples:
        sample_id = str(sample["sample_id"])
        variant_payloads = sample.get("variants", {})
        for variant in variants:
            payload = variant_payloads.get(variant)
            if payload is None:
                continue
            input_dir = Path(payload["input_dir"])
            output_dir = output_root / model_name / sample_id / variant
            yield sample_id, str(variant), input_dir, output_dir


def add_repo_imports() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    pi3_native_root = repo_root / "aidi" / "third_party" / "pi3_training"
    for path in (repo_root, pi3_native_root):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return repo_root


def setup_environment(mode: str, config: Path | None) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if mode == "base":
        for key in ("PI3_MODEL_IMPL", "PI3_CONFIG", "PI3_NATIVE_ROOT", "PI3_INDEXER_EVAL_MODE"):
            os.environ.pop(key, None)
        return
    repo_root = Path(__file__).resolve().parents[1]
    pi3_native_root = repo_root / "aidi" / "third_party" / "pi3_training"
    if config is None:
        raise ValueError("GeoWeave mode requires a config path.")
    os.environ["PI3_MODEL_IMPL"] = "native_sparse"
    os.environ["PI3_CONFIG"] = str(config)
    os.environ["PI3_NATIVE_ROOT"] = str(pi3_native_root)
    os.environ.setdefault("PI3_INDEXER_EVAL_MODE", "auto")


def load_model(mode: str, checkpoint: Path, config: Path | None, device: Any):
    repo_root = add_repo_imports()
    setup_environment(mode, config)
    from aidi.scripts.baselines.eval_pi3_mv_recon_core import load_pi3_model

    model, loaded_checkpoint = load_pi3_model(None, str(checkpoint), device=device)
    return repo_root, model, loaded_checkpoint


def prediction_cache_payload(
    *,
    predictions: dict[str, np.ndarray],
    image_paths: Sequence[Path],
    camera_only: bool,
) -> dict[str, np.ndarray]:
    payload = {
        "camera_poses": predictions["camera_poses"].astype(np.float32, copy=False),
        "image_paths": np.asarray([str(path) for path in image_paths]),
    }
    if not bool(camera_only):
        payload["points"] = predictions["points"].astype(np.float32, copy=False)
    return payload


def prediction_cache_is_valid(path: Path) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with np.load(path) as data:
            if "camera_poses" not in data or "image_paths" not in data:
                return False
            poses = np.asarray(data["camera_poses"])
            image_paths = np.asarray(data["image_paths"])
            return bool(
                poses.shape == (10, 4, 4)
                and np.isfinite(poses).all()
                and image_paths.shape == (10,)
            )
    except (EOFError, OSError, KeyError, ValueError, zipfile.BadZipFile):
        return False


def run_single_job(
    *,
    model: Any,
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    metadata_base: dict[str, Any],
) -> None:
    from tools.run_pi3_geoweave_smoke_infer import collect_images, read_rgb_arrays, write_ascii_ply
    from aidi.scripts.baselines.eval_pi3_mv_recon_core import infer_pi3_mv_predictions

    image_paths = collect_images(input_dir.expanduser().resolve(), int(args.max_images))
    first_image = Image.open(image_paths[0]).convert("RGB")
    data_size = (first_image.height, first_image.width)
    started = time.time()
    predictions = infer_pi3_mv_predictions(
        filelist=[str(path) for path in image_paths],
        model=model,
        load_img_size=int(args.load_img_size),
        device=str(args.device),
        verbose=bool(args.verbose),
        data_size=data_size,
        point_source=str(args.point_source),
    )
    points = predictions["points"].astype(np.float32, copy=False)
    camera_poses = predictions["camera_poses"].astype(np.float32, copy=False)
    elapsed = time.time() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "points.npz"
    ply_path = output_dir / f"points_stride{max(int(args.point_stride), 1)}.ply"
    metadata_path = output_dir / "metadata.json"
    cache_payload = prediction_cache_payload(
        predictions=predictions,
        image_paths=image_paths,
        camera_only=bool(args.camera_only),
    )
    if args.camera_only:
        np.savez_compressed(npz_path, **cache_payload)
        ply_vertices = 0
    else:
        colors = read_rgb_arrays(image_paths, data_size)
        cache_payload["colors"] = colors
        np.savez_compressed(npz_path, **cache_payload)
        ply_vertices = write_ascii_ply(ply_path, points=points, colors=colors, stride=int(args.point_stride))

    metadata = dict(metadata_base)
    metadata.update(
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "image_paths": [str(path) for path in image_paths],
            "data_size_hw": list(data_size),
            "elapsed_seconds": float(elapsed),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "points_shape": list(points.shape),
            "points_dtype": str(points.dtype),
            "points_finite_ratio": float(np.isfinite(points).all(axis=-1).mean()),
            "camera_poses_shape": list(camera_poses.shape),
            "camera_poses_finite_ratio": float(np.isfinite(camera_poses).mean()),
            "npz_path": str(npz_path),
            "ply_path": str(ply_path),
            "ply_vertices": int(ply_vertices),
            "camera_only": bool(args.camera_only),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_protocol(args: argparse.Namespace) -> None:
    import torch

    protocol = load_protocol(args.protocol)
    output_root = Path(args.output_root).expanduser().resolve()
    model_name = str(args.model_name or default_model_name(str(args.mode)))
    checkpoint = Path(args.checkpoint or default_checkpoint(str(args.mode))).expanduser().resolve()
    config = Path(args.config or DEFAULT_GEOWEAVE_CONFIG).expanduser().resolve() if args.mode == "geoweave" else None
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if config is not None and not config.is_file():
        raise FileNotFoundError(f"Config not found: {config}")

    device = torch.device(str(args.device))
    repo_root, model, loaded_checkpoint = load_model(str(args.mode), checkpoint, config, device)
    metadata_base = {
        "repo_root": str(repo_root),
        "protocol": str(Path(args.protocol).expanduser().resolve()),
        "mode": str(args.mode),
        "model_name": model_name,
        "checkpoint": str(checkpoint),
        "config": str(config) if config is not None else "",
        "loaded_checkpoint": loaded_checkpoint,
        "max_images": int(args.max_images),
        "load_img_size": int(args.load_img_size),
        "device": str(args.device),
        "point_source": str(args.point_source),
        "camera_only": bool(args.camera_only),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    if device.type == "cuda":
        metadata_base["cuda_device_name"] = torch.cuda.get_device_name(device)

    jobs = list(
        iter_protocol_jobs(
            protocol,
            output_root=output_root,
            model_name=model_name,
            variants=list(args.variants),
            sample_limit=int(args.sample_limit),
        )
    )
    print(f"[pi3-protocol] mode={args.mode} model={model_name} jobs={len(jobs)} output_root={output_root}", flush=True)
    for index, (sample_id, variant, input_dir, output_dir) in enumerate(jobs, start=1):
        cache_path = output_dir / "points.npz"
        if prediction_cache_is_valid(cache_path) and not args.overwrite:
            print(f"[pi3-protocol] skip existing {index}/{len(jobs)} {sample_id} {variant}", flush=True)
            continue
        if cache_path.exists() and not args.overwrite:
            print(f"[pi3-protocol] rerun invalid cache {index}/{len(jobs)} {sample_id} {variant}", flush=True)
        print(f"[pi3-protocol] run {index}/{len(jobs)} {sample_id} {variant}", flush=True)
        job_meta = dict(metadata_base)
        job_meta.update({"sample_id": sample_id, "variant": variant})
        run_single_job(model=model, input_dir=input_dir, output_dir=output_dir, args=args, metadata_base=job_meta)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    run_protocol(parse_args())


if __name__ == "__main__":
    main()
