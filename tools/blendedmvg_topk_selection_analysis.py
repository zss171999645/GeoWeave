#!/usr/bin/env python3
"""Capture and summarize GeoWeave Top-K selection roles on BlendedMVG inputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_WEIGHTS_ROOT = Path(
    "/mnt/cfs/zhoufeng/geoweave_rebuttal_code_weights_20260604/"
    "weights/geoweave_paper_handoff_20260602_final"
)
DEFAULT_PI3_ROOT = DEFAULT_WEIGHTS_ROOT / "pi3_geoweave_native_sparse_20260505_checkpoint_79"
DEFAULT_CKPT = DEFAULT_PI3_ROOT / "checkpoint_79" / "pytorch_model.bin"
DEFAULT_CONFIG = DEFAULT_PI3_ROOT / "config.yaml"
DEFAULT_VARIANTS = ("clean", "tail", "xscene")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--variants", nargs="*", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--load-img-size", type=int, default=518)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--query-token-stride", type=int, default=8)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def collect_images(input_dir: Path, max_images: int) -> list[Path]:
    image_paths: list[Path] = []
    for pattern in ("input_v*.png", "input_v*.jpg", "input_v*.jpeg", "*.png", "*.jpg", "*.jpeg"):
        for path in sorted(input_dir.glob(pattern)):
            if path.is_file() and path not in image_paths:
                image_paths.append(path)
    image_paths = [path for path in image_paths if "column" not in path.stem.lower()]
    if max_images > 0:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise FileNotFoundError(f"No input images found under {input_dir}")
    return image_paths


def _query_indices(
    *,
    num_tokens: int,
    tokens_per_view: int,
    patch_start_idx: int,
    prefix_size: int,
    scope: str,
    stride: int,
) -> np.ndarray:
    token_ids = np.arange(num_tokens, dtype=np.int64)
    local_ids = token_ids % int(tokens_per_view)
    view_ids = token_ids // int(tokens_per_view)
    if scope == "all_patch_queries":
        mask = local_ids >= int(patch_start_idx)
    elif scope == "prefix_patch_queries":
        mask = (view_ids < int(prefix_size)) & (local_ids >= int(patch_start_idx))
    else:
        raise ValueError(scope)
    selected = token_ids[mask]
    stride = max(int(stride), 1)
    return selected[::stride]


def summarize_topk_roles(
    *,
    topk_indices: np.ndarray,
    roles: list[str],
    tokens_per_view: int,
    patch_start_idx: int,
    prefix_size: int,
    query_token_stride: int = 1,
) -> list[dict[str, Any]]:
    if topk_indices.ndim != 3 or topk_indices.shape[0] != 1:
        raise ValueError(f"Expected topk_indices shape [1,L,K], got {topk_indices.shape}")
    num_tokens = int(topk_indices.shape[1])
    rows: list[dict[str, Any]] = []
    for scope in ("all_patch_queries", "prefix_patch_queries"):
        q_indices = _query_indices(
            num_tokens=num_tokens,
            tokens_per_view=tokens_per_view,
            patch_start_idx=patch_start_idx,
            prefix_size=prefix_size,
            scope=scope,
            stride=query_token_stride,
        )
        if q_indices.size == 0:
            continue
        selected = topk_indices[0, q_indices].reshape(-1).astype(np.int64, copy=False)
        valid = (selected >= 0) & (selected < num_tokens)
        selected = selected[valid]
        key_views = selected // int(tokens_per_view)
        key_local = selected % int(tokens_per_view)
        total = int(selected.size)
        seen_roles: list[str] = []
        for role in roles:
            if role not in seen_roles:
                seen_roles.append(role)
        for role in seen_roles:
            role_view_indices = [idx for idx, value in enumerate(roles) if value == role]
            count = int(np.sum(np.isin(key_views, role_view_indices)))
            rows.append(
                {
                    "query_scope": scope,
                    "granularity": "role_group",
                    "key_role": role,
                    "key_view": -2,
                    "count": count,
                    "total": total,
                    "share": float(count / total) if total else float("nan"),
                }
            )
        for view_index, role in enumerate(roles):
            count = int(np.sum(key_views == int(view_index)))
            rows.append(
                {
                    "query_scope": scope,
                    "granularity": "view",
                    "key_role": role,
                    "key_view": int(view_index),
                    "count": count,
                    "total": total,
                    "share": float(count / total) if total else float("nan"),
                }
            )
        special_count = int(np.sum(key_local < int(patch_start_idx)))
        rows.append(
            {
                "query_scope": scope,
                "granularity": "role_group",
                "key_role": "__special_tokens__",
                "key_view": -1,
                "count": special_count,
                "total": total,
                "share": float(special_count / total) if total else float("nan"),
            }
        )
    return rows


def setup_model(args: argparse.Namespace):
    import torch

    repo_root = Path(__file__).resolve().parents[1]
    pi3_native_root = repo_root / "aidi" / "third_party" / "pi3_training"
    for path in (repo_root, pi3_native_root):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    os.environ["PI3_MODEL_IMPL"] = "native_sparse"
    os.environ["PI3_CONFIG"] = str(Path(args.config).expanduser().resolve())
    os.environ["PI3_NATIVE_ROOT"] = str(pi3_native_root)
    os.environ["PI3_INDEXER_EVAL_MODE"] = "sparse"
    os.environ["VGGT_DSA_SPARSE_STREAM_RECORD_LAST"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from aidi.scripts.baselines.eval_pi3_mv_recon_core import load_pi3_model

    model, loaded_checkpoint = load_pi3_model(None, str(Path(args.checkpoint).expanduser().resolve()), device=torch.device(args.device))
    return model, loaded_checkpoint


def run_forward(model, image_paths: list[Path], args: argparse.Namespace) -> None:
    from aidi.scripts.baselines.eval_pi3_mv_recon_core import infer_pi3_mv_pointclouds

    first = Image.open(image_paths[0]).convert("RGB")
    points = infer_pi3_mv_pointclouds(
        filelist=[str(path) for path in image_paths],
        model=model,
        load_img_size=int(args.load_img_size),
        device=str(args.device),
        verbose=False,
        data_size=(first.height, first.width),
        point_source="native",
    )
    del points


def capture_sample_variant(
    *,
    model,
    sample: dict[str, Any],
    variant: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    image_paths = collect_images(Path(sample["variants"][variant]["input_dir"]), max_images=int(args.max_images))
    roles = [frame["role"] for frame in sample["variants"][variant]["frames"][: len(image_paths)]]
    run_forward(model, image_paths, args)

    rows: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        topk = getattr(module, "last_topk_indices", None)
        if topk is None:
            continue
        topk_np = topk.detach().cpu().numpy()
        tokens_per_view = int(topk_np.shape[1]) // len(roles)
        layer_rows = summarize_topk_roles(
            topk_indices=topk_np,
            roles=roles,
            tokens_per_view=tokens_per_view,
            patch_start_idx=int(getattr(model, "patch_start_idx", 5)),
            prefix_size=int(sample.get("prefix_size", 6)),
            query_token_stride=int(args.query_token_stride),
        )
        for row in layer_rows:
            row.update(
                {
                    "sample_id": sample["sample_id"],
                    "variant": variant,
                    "layer": name,
                    "tokens_per_view": tokens_per_view,
                    "topk": int(topk_np.shape[-1]),
                    "query_token_stride": int(args.query_token_stride),
                }
            )
        rows.extend(layer_rows)
        module.last_topk_indices = None
    return rows


def main() -> None:
    args = parse_args()
    protocol = json.loads(Path(args.protocol).expanduser().resolve().read_text(encoding="utf-8"))
    samples = list(protocol["samples"])
    if int(args.sample_limit) > 0:
        samples = samples[: int(args.sample_limit)]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, loaded_checkpoint = setup_model(args)
    all_rows: list[dict[str, Any]] = []
    for sample in samples:
        for variant in args.variants:
            all_rows.extend(capture_sample_variant(model=model, sample=sample, variant=variant, args=args))
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    csv_path = output_dir / "topk_role_shares.csv"
    json_path = output_dir / "topk_role_shares.json"
    write_csv(csv_path, all_rows)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": str(Path(args.protocol).expanduser().resolve()),
        "loaded_checkpoint": loaded_checkpoint,
        "load_img_size": int(args.load_img_size),
        "sample_count": len(samples),
        "variants": list(args.variants),
        "rows": all_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[topk-selection] wrote {csv_path}")
    print(f"[topk-selection] wrote {json_path}")


if __name__ == "__main__":
    main()
