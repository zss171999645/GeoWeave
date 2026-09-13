#!/usr/bin/env python3
"""Run a dense-teacher Top-K baseline on the BlendedMVG rebuttal protocol.

This script does not train or change checkpoint weights.  It patches each DSA
attention module at inference time so that the support set is selected by the
module's own dense attention distribution, then the attention output is
recomputed on only those Top-K keys.  This is the same-backbone diagnostic for
"dense teacher Top-K pruning/re-normalization".
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from PIL import Image

from blendedmvg_topk_selection_analysis import (
    DEFAULT_CONFIG,
    DEFAULT_CKPT,
    DEFAULT_VARIANTS,
    collect_images,
    summarize_topk_roles,
    write_csv,
)


MODEL_NAME = "dense_teacher_topk"
DEFAULT_PREFIX_SIZE = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run teacher-TopK sparse inference and role accounting.")
    run.add_argument("--protocol", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument("--variants", nargs="*", default=list(DEFAULT_VARIANTS))
    run.add_argument("--sample-limit", type=int, default=0)
    run.add_argument("--load-img-size", type=int, default=518)
    run.add_argument("--device", default="cuda")
    run.add_argument("--max-images", type=int, default=10)
    run.add_argument("--teacher-topk", type=int, default=0)
    run.add_argument("--query-token-stride", type=int, default=8)
    run.add_argument("--point-source", choices=["native", "depth_pose"], default="native")
    run.add_argument("--point-stride", type=int, default=16)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--roles-only", action="store_true", help="Skip writing point outputs.")

    jobs = sub.add_parser("emit-jobs", help="Print protocol jobs.")
    jobs.add_argument("--protocol", required=True)
    jobs.add_argument("--output-root", required=True)
    jobs.add_argument("--variants", nargs="*", default=list(DEFAULT_VARIANTS))
    jobs.add_argument("--sample-limit", type=int, default=0)

    return parser.parse_args()


def load_protocol(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def iter_protocol_jobs(
    protocol: dict[str, Any],
    *,
    output_root: Path,
    variants: Sequence[str],
    sample_limit: int,
) -> Iterator[tuple[str, str, dict[str, Any], dict[str, Any], Path]]:
    samples = list(protocol.get("samples", []))
    if int(sample_limit) > 0:
        samples = samples[: int(sample_limit)]
    for sample in samples:
        sample_id = str(sample["sample_id"])
        for variant in variants:
            variant_payload = sample.get("variants", {}).get(variant)
            if variant_payload is None:
                continue
            out_dir = output_root / MODEL_NAME / sample_id / variant
            yield sample_id, variant, variant_payload, sample, out_dir


def read_rgb_arrays(image_paths: list[Path], target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    arrays = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        arrays.append(np.asarray(img, dtype=np.uint8))
    return np.stack(arrays, axis=0)


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray, stride: int) -> int:
    stride = max(int(stride), 1)
    sampled_points = points[:, ::stride, ::stride, :].reshape(-1, 3)
    sampled_colors = colors[:, ::stride, ::stride, :].reshape(-1, 3)
    finite = np.isfinite(sampled_points).all(axis=1)
    bounded = np.abs(sampled_points).max(axis=1) < 1.0e6
    valid = finite & bounded
    sampled_points = sampled_points[valid].astype(np.float32, copy=False)
    sampled_colors = sampled_colors[valid].astype(np.uint8, copy=False)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(sampled_points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for xyz, rgb in zip(sampled_points, sampled_colors):
            handle.write(
                f"{xyz[0]:.7g} {xyz[1]:.7g} {xyz[2]:.7g} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )
    return int(len(sampled_points))


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
    os.environ["PI3_INDEXER_EVAL_MODE"] = "dense"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from aidi.scripts.baselines.eval_pi3_mv_recon_core import load_pi3_model

    model, loaded_checkpoint = load_pi3_model(
        None,
        str(Path(args.checkpoint).expanduser().resolve()),
        device=torch.device(args.device),
    )
    install_dense_teacher_topk_forward(model, teacher_topk=int(args.teacher_topk))
    return model, loaded_checkpoint


def _dense_teacher_forward(self, x, pos=None):
    import torch

    bsz, tgt_len, _ = x.shape
    self.last_topk_indices = None
    self.last_topk_scores = None
    self.last_dense_teacher_topk_indices = None
    self.last_dense_teacher_topk_scores = None
    state = self._state()
    mask = self._normalize_mask(state.get("attn_mask", None), bsz)

    qkv = self.qkv(x)
    q, k, v = self._project_attention_qkv_from_flat(qkv, bsz, tgt_len)
    q, k = self.q_norm(q), self.k_norm(k)
    if self.rope is not None and pos is not None:
        q = self.rope(q, pos)
        k = self.rope(k, pos)

    orig_dtype = q.dtype
    score_dtype = self._resolve_score_dtype(state, orig_dtype)
    if score_dtype != orig_dtype:
        q = q.to(score_dtype)
        k = k.to(score_dtype)
        v = v.to(score_dtype)
    if mask is not None and mask.dtype != q.dtype:
        mask = mask.to(q.dtype)

    k_dense = self._expand_attention_kv_heads(k)
    v_dense = self._expand_attention_kv_heads(v)
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k_dense) * self._scale_like(q)
    if mask is not None:
        scores = scores + mask
    dense_attn = scores.softmax(dim=-1)
    dense_summary = dense_attn.mean(dim=1)

    requested_topk = int(state.get("teacher_topk", 0) or getattr(self, "_teacher_topk_override", 0) or state.get("topk", self.indexer_cfg.topk))
    topk = max(1, min(requested_topk, int(tgt_len)))
    topk_scores, topk_indices = torch.topk(dense_summary, k=topk, dim=-1, largest=True, sorted=False)
    self.last_dense_teacher_topk_indices = topk_indices.detach().to(torch.int32).cpu()
    self.last_dense_teacher_topk_scores = topk_scores.detach().float().cpu()
    self.last_topk_indices = self.last_dense_teacher_topk_indices
    self.last_topk_scores = self.last_dense_teacher_topk_scores

    del scores, dense_attn, dense_summary, topk_scores
    sparse_out, _ = self._sparse_attention(q, k_dense, v_dense, topk_indices, mask)
    if sparse_out.dtype != orig_dtype:
        sparse_out = sparse_out.to(orig_dtype)
    out = sparse_out.transpose(1, 2).reshape(bsz, tgt_len, -1)
    out = self.proj(out)
    out = self.proj_drop(out)
    return out, None


def install_dense_teacher_topk_forward(model, *, teacher_topk: int) -> int:
    patched = 0
    for _name, module in model.named_modules():
        if not (
            hasattr(module, "_project_attention_qkv_from_flat")
            and hasattr(module, "_sparse_attention")
            and hasattr(module, "indexer_cfg")
        ):
            continue
        module._teacher_topk_override = int(teacher_topk)
        module.forward = types.MethodType(_dense_teacher_forward, module)
        patched += 1
    if patched <= 0:
        raise RuntimeError("No DSA attention modules were patched for dense-teacher Top-K.")
    return patched


def run_forward(model, image_paths: list[Path], args: argparse.Namespace) -> np.ndarray:
    from aidi.scripts.baselines.eval_pi3_mv_recon_core import infer_pi3_mv_pointclouds

    first = Image.open(image_paths[0]).convert("RGB")
    return infer_pi3_mv_pointclouds(
        filelist=[str(path) for path in image_paths],
        model=model,
        load_img_size=int(args.load_img_size),
        device=str(args.device),
        verbose=False,
        data_size=(first.height, first.width),
        point_source=str(args.point_source),
    )


def collect_dense_teacher_rows(
    *,
    model,
    sample: dict[str, Any],
    variant: str,
    query_token_stride: int,
) -> list[dict[str, Any]]:
    roles = [frame["role"] for frame in sample["variants"][variant]["frames"]]
    rows: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        topk = getattr(module, "last_dense_teacher_topk_indices", None)
        if topk is None:
            continue
        import torch

        if isinstance(topk, torch.Tensor):
            topk_np = topk.detach().cpu().numpy()
        else:
            topk_np = np.asarray(topk)
        tokens_per_view = int(topk_np.shape[1]) // len(roles)
        layer_rows = summarize_topk_roles(
            topk_indices=topk_np,
            roles=roles,
            tokens_per_view=tokens_per_view,
            patch_start_idx=int(getattr(model, "patch_start_idx", 5)),
            prefix_size=int(sample.get("prefix_size", DEFAULT_PREFIX_SIZE)),
            query_token_stride=int(query_token_stride),
        )
        for row in layer_rows:
            row.update(
                {
                    "sample_id": sample["sample_id"],
                    "variant": variant,
                    "layer": name,
                    "tokens_per_view": tokens_per_view,
                    "topk": int(topk_np.shape[-1]),
                    "query_token_stride": int(query_token_stride),
                    "baseline": MODEL_NAME,
                }
            )
        rows.extend(layer_rows)
        module.last_dense_teacher_topk_indices = None
        module.last_dense_teacher_topk_scores = None
        module.last_topk_indices = None
        module.last_topk_scores = None
    return rows


def run_protocol(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model, loaded_checkpoint = setup_model(args)

    all_rows: list[dict[str, Any]] = []
    for sample_id, variant, variant_payload, sample, out_dir in iter_protocol_jobs(
        protocol,
        output_root=output_root,
        variants=tuple(args.variants),
        sample_limit=int(args.sample_limit),
    ):
        if out_dir.exists() and (out_dir / "points.npz").is_file() and not args.overwrite and not args.roles_only:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        image_paths = collect_images(Path(variant_payload["input_dir"]), max_images=int(args.max_images))
        started = time.time()
        points = run_forward(model, image_paths, args)
        elapsed = time.time() - started
        all_rows.extend(
            collect_dense_teacher_rows(
                model=model,
                sample=sample,
                variant=variant,
                query_token_stride=int(args.query_token_stride),
            )
        )
        if not args.roles_only:
            first = Image.open(image_paths[0]).convert("RGB")
            data_size = (first.height, first.width)
            colors = read_rgb_arrays(image_paths, data_size)
            np.savez_compressed(out_dir / "points.npz", points=points.astype(np.float32), colors=colors)
            ply_vertices = write_ascii_ply(
                out_dir / f"points_stride{max(int(args.point_stride), 1)}.ply",
                points,
                colors,
                stride=int(args.point_stride),
            )
            metadata = {
                "baseline": MODEL_NAME,
                "sample_id": sample_id,
                "variant": variant,
                "image_paths": [str(path) for path in image_paths],
                "loaded_checkpoint": loaded_checkpoint,
                "config": str(Path(args.config).expanduser().resolve()),
                "teacher_topk": int(args.teacher_topk),
                "load_img_size": int(args.load_img_size),
                "point_source": str(args.point_source),
                "elapsed_seconds": float(elapsed),
                "points_shape": list(points.shape),
                "points_finite_ratio": float(np.isfinite(points).all(axis=-1).mean()),
                "ply_vertices": int(ply_vertices),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            (out_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    role_csv = output_root / "dense_teacher_topk_role_shares.csv"
    role_json = output_root / "dense_teacher_topk_role_shares.json"
    write_csv(role_csv, all_rows)
    role_json.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "baseline": MODEL_NAME,
                "protocol": str(Path(args.protocol).expanduser().resolve()),
                "output_root": str(output_root),
                "loaded_checkpoint": loaded_checkpoint,
                "teacher_topk": int(args.teacher_topk),
                "load_img_size": int(args.load_img_size),
                "sample_limit": int(args.sample_limit),
                "variants": list(args.variants),
                "rows": all_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[dense-teacher-topk] wrote {role_csv}")
    print(f"[dense-teacher-topk] wrote {role_json}")


def emit_jobs(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    output_root = Path(args.output_root).expanduser().resolve()
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    for sample_id, variant, variant_payload, _sample, out_dir in iter_protocol_jobs(
        protocol,
        output_root=output_root,
        variants=tuple(args.variants),
        sample_limit=int(args.sample_limit),
    ):
        writer.writerow([sample_id, variant, variant_payload["input_dir"], str(out_dir)])


def main() -> None:
    args = parse_args()
    if args.cmd == "run":
        run_protocol(args)
    elif args.cmd == "emit-jobs":
        emit_jobs(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
