#!/usr/bin/env python3
"""Analyze GeoWeave learned Top-K alignment with dense attention Top-K.

The diagnostic keeps the normal GeoWeave sparse inference path intact.  Each
DSA attention module is wrapped to compute dense-attention probabilities for a
deterministic subset of patch queries at the same hidden state, then compares
the learned GeoWeave Top-K keys against the dense-attention Top-K keys.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from blendedmvg_topk_selection_analysis import (
    DEFAULT_CONFIG,
    DEFAULT_CKPT,
    collect_images,
    setup_model,
    write_csv,
)


DEFAULT_VARIANTS = ("clean_tail", "plausible_noise_tail")
DEFAULT_SCOPES = ("all_patch_queries", "prefix_patch_queries")


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
    parser.add_argument("--query-token-stride", type=int, default=32)
    parser.add_argument("--max-query-tokens", type=int, default=128)
    parser.add_argument("--scopes", nargs="*", default=list(DEFAULT_SCOPES))
    return parser.parse_args()


def load_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def finite_mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(vals)) if vals else float("nan")


def query_indices_numpy(
    *,
    num_tokens: int,
    tokens_per_view: int,
    patch_start_idx: int,
    prefix_size: int,
    scope: str,
    stride: int,
    max_queries: int,
) -> np.ndarray:
    token_ids = np.arange(int(num_tokens), dtype=np.int64)
    local_ids = token_ids % int(tokens_per_view)
    view_ids = token_ids // int(tokens_per_view)
    if scope == "all_patch_queries":
        mask = local_ids >= int(patch_start_idx)
    elif scope == "prefix_patch_queries":
        mask = (view_ids < int(prefix_size)) & (local_ids >= int(patch_start_idx))
    else:
        raise ValueError(f"Unknown query scope: {scope}")

    selected = token_ids[mask]
    selected = selected[:: max(int(stride), 1)]
    if int(max_queries) > 0 and selected.size > int(max_queries):
        keep = np.linspace(0, selected.size - 1, int(max_queries)).round().astype(np.int64)
        selected = selected[keep]
    return selected.astype(np.int64, copy=False)


def summarize_alignment_numpy(
    learned_topk: np.ndarray,
    teacher_topk: np.ndarray,
    dense_probs: np.ndarray,
) -> dict[str, Any]:
    learned = np.asarray(learned_topk, dtype=np.int64)
    teacher = np.asarray(teacher_topk, dtype=np.int64)
    probs = np.asarray(dense_probs, dtype=np.float64)
    if learned.ndim != 2 or teacher.ndim != 2 or probs.ndim != 2:
        raise ValueError(
            f"Expected learned [Q,K], teacher [Q,K], probs [Q,L], got "
            f"{learned.shape}, {teacher.shape}, {probs.shape}"
        )
    if learned.shape != teacher.shape or learned.shape[0] != probs.shape[0]:
        raise ValueError(
            f"Mismatched learned/teacher/probability shapes: "
            f"{learned.shape}, {teacher.shape}, {probs.shape}"
        )

    query_count, topk = learned.shape
    key_count = int(probs.shape[1])
    overlap_counts: list[float] = []
    recalls: list[float] = []
    jaccards: list[float] = []
    learned_masses: list[float] = []
    teacher_masses: list[float] = []

    for query_index in range(query_count):
        learned_row = learned[query_index]
        teacher_row = teacher[query_index]
        learned_valid = learned_row[(learned_row >= 0) & (learned_row < key_count)]
        teacher_valid = teacher_row[(teacher_row >= 0) & (teacher_row < key_count)]
        learned_set = set(int(value) for value in learned_valid.tolist())
        teacher_set = set(int(value) for value in teacher_valid.tolist())
        intersection = learned_set & teacher_set
        union = learned_set | teacher_set
        overlap = float(len(intersection))
        overlap_counts.append(overlap)
        recalls.append(overlap / float(max(len(teacher_set), 1)))
        jaccards.append(overlap / float(max(len(union), 1)))
        learned_masses.append(float(probs[query_index, learned_valid].sum()) if learned_valid.size else 0.0)
        teacher_masses.append(float(probs[query_index, teacher_valid].sum()) if teacher_valid.size else 0.0)

    mean_learned_mass = finite_mean(learned_masses)
    mean_teacher_mass = finite_mean(teacher_masses)
    return {
        "queries": int(query_count),
        "topk": int(topk),
        "key_count": int(key_count),
        "mean_overlap_count": finite_mean(overlap_counts),
        "mean_recall_at_k": finite_mean(recalls),
        "mean_jaccard_at_k": finite_mean(jaccards),
        "mean_dense_mass_learned": mean_learned_mass,
        "mean_dense_mass_teacher": mean_teacher_mass,
        "mean_dense_mass_ratio": float(mean_learned_mass / mean_teacher_mass)
        if mean_teacher_mass > 0.0
        else float("nan"),
        "random_mass_at_k": float(topk / key_count) if key_count > 0 else float("nan"),
    }


def _mask_for_query_indices(mask: Any, q_indices: Any) -> Any:
    if mask is None:
        return None
    if int(mask.shape[-2]) == int(q_indices.numel()):
        return mask
    return mask.index_select(-2, q_indices)


def _alignment_capture_forward(self, x, pos=None):
    import torch

    state = self._state()
    should_capture = (
        self.indexer is not None
        and bool(state.get("enabled", False))
        and bool(state.get("sparse", False))
        and not torch.is_grad_enabled()
    )
    captures: dict[str, dict[str, Any]] = {}

    if should_capture:
        bsz, tgt_len, _ = x.shape
        mask = self._normalize_mask(state.get("attn_mask", None), bsz)
        qkv = self.qkv(x)
        q, k, _v = self._project_attention_qkv_from_flat(qkv, bsz, tgt_len)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        score_dtype = self._resolve_score_dtype(state, q.dtype)
        if score_dtype != q.dtype:
            q = q.to(score_dtype)
            k = k.to(score_dtype)
        if mask is not None and mask.dtype != q.dtype:
            mask = mask.to(q.dtype)
        k_dense = self._expand_attention_kv_heads(k)

        topk = min(int(state.get("topk", self.indexer_cfg.topk)), int(tgt_len))
        num_views = int(state.get("num_views", 0) or 0)
        tokens_per_view = int(state.get("tokens_per_view", 0) or 0)
        if tokens_per_view <= 0 and num_views > 0:
            tokens_per_view = int(tgt_len) // int(num_views)
        if tokens_per_view <= 0:
            tokens_per_view = int(tgt_len)
        patch_start_idx = int(state.get("patch_start_idx", 0) or 0)
        prefix_size = int(getattr(self, "_alignment_prefix_size", 6))
        stride = int(getattr(self, "_alignment_query_token_stride", 32))
        max_queries = int(getattr(self, "_alignment_max_query_tokens", 128))
        scopes = tuple(getattr(self, "_alignment_query_scopes", DEFAULT_SCOPES))

        with torch.no_grad():
            for scope in scopes:
                q_indices_np = query_indices_numpy(
                    num_tokens=int(tgt_len),
                    tokens_per_view=int(tokens_per_view),
                    patch_start_idx=int(patch_start_idx),
                    prefix_size=int(prefix_size),
                    scope=str(scope),
                    stride=int(stride),
                    max_queries=int(max_queries),
                )
                if q_indices_np.size == 0:
                    continue
                q_indices = torch.as_tensor(q_indices_np, dtype=torch.long, device=q.device)
                q_sample = q.index_select(2, q_indices)
                scores = torch.einsum("bhqd,bhkd->bhqk", q_sample, k_dense) * self._scale_like(q_sample)
                q_mask = _mask_for_query_indices(mask, q_indices)
                if q_mask is not None:
                    scores = scores + q_mask
                dense_probs = scores.float().softmax(dim=-1).mean(dim=1)
                _teacher_scores, teacher_topk = torch.topk(
                    dense_probs,
                    k=int(topk),
                    dim=-1,
                    largest=True,
                    sorted=False,
                )
                captures[str(scope)] = {
                    "q_indices": q_indices.detach().cpu(),
                    "dense_probs": dense_probs.detach().cpu(),
                    "teacher_topk": teacher_topk.detach().cpu(),
                    "tokens_per_view": int(tokens_per_view),
                    "patch_start_idx": int(patch_start_idx),
                    "topk": int(topk),
                    "num_tokens": int(tgt_len),
                }
                del scores, dense_probs, teacher_topk, q_sample, q_indices

        del qkv, q, k, k_dense

    out = self._alignment_original_forward(x, pos=pos)

    if should_capture and captures:
        learned_topk = getattr(self, "last_topk_indices", None)
        records: list[dict[str, Any]] = []
        if isinstance(learned_topk, torch.Tensor):
            learned_cpu = learned_topk.detach().cpu()
            for scope, capture in captures.items():
                q_indices_cpu = capture["q_indices"]
                dense_probs_cpu = capture["dense_probs"]
                teacher_topk_cpu = capture["teacher_topk"]
                topk = min(int(capture["topk"]), int(learned_cpu.shape[-1]), int(teacher_topk_cpu.shape[-1]))
                if topk <= 0:
                    continue
                for batch_index in range(int(learned_cpu.shape[0])):
                    learned_sample = learned_cpu[batch_index].index_select(0, q_indices_cpu)[..., :topk].numpy()
                    teacher_sample = teacher_topk_cpu[batch_index, :, :topk].numpy()
                    dense_probs_sample = dense_probs_cpu[batch_index].numpy()
                    metrics = summarize_alignment_numpy(learned_sample, teacher_sample, dense_probs_sample)
                    metrics.update(
                        {
                            "query_scope": str(scope),
                            "batch_index": int(batch_index),
                            "tokens_per_view": int(capture["tokens_per_view"]),
                            "patch_start_idx": int(capture["patch_start_idx"]),
                            "num_tokens": int(capture["num_tokens"]),
                        }
                    )
                    records.append(metrics)
        self.last_topk_alignment = records

    return out


def install_alignment_capture(model: Any, *, prefix_size: int, args: argparse.Namespace) -> int:
    patched = 0
    for _name, module in model.named_modules():
        if not (
            hasattr(module, "_project_attention_qkv_from_flat")
            and hasattr(module, "_expand_attention_kv_heads")
            and hasattr(module, "indexer_cfg")
        ):
            continue
        if hasattr(module, "_alignment_original_forward"):
            continue
        module._alignment_original_forward = module.forward
        module._alignment_prefix_size = int(prefix_size)
        module._alignment_query_token_stride = int(args.query_token_stride)
        module._alignment_max_query_tokens = int(args.max_query_tokens)
        module._alignment_query_scopes = tuple(str(scope) for scope in args.scopes)
        module.last_topk_alignment = []
        module.forward = types.MethodType(_alignment_capture_forward, module)
        patched += 1
    if patched <= 0:
        raise RuntimeError("No DSA attention modules were patched for Top-K alignment capture.")
    return patched


def run_forward(model: Any, image_paths: list[Path], args: argparse.Namespace) -> None:
    from aidi.scripts.baselines.eval_pi3_mv_recon_core import infer_pi3_mv_pointclouds
    from PIL import Image

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


def collect_alignment_rows(
    *,
    model: Any,
    sample: dict[str, Any],
    variant: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    image_paths = collect_images(Path(sample["variants"][variant]["input_dir"]), max_images=int(args.max_images))
    run_forward(model, image_paths, args)

    rows: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        records = getattr(module, "last_topk_alignment", None)
        if not records:
            continue
        for record in records:
            row = dict(record)
            row.update(
                {
                    "sample_id": str(sample["sample_id"]),
                    "variant": str(variant),
                    "layer": str(name),
                    "query_token_stride": int(args.query_token_stride),
                    "max_query_tokens": int(args.max_query_tokens),
                }
            )
            rows.append(row)
        module.last_topk_alignment = []
        module.last_topk_indices = None
        module.last_topk_scores = None
    return rows


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["variant"]), str(row["query_scope"])), []).append(row)

    aggregate: list[dict[str, Any]] = []
    metric_keys = (
        "mean_overlap_count",
        "mean_recall_at_k",
        "mean_jaccard_at_k",
        "mean_dense_mass_learned",
        "mean_dense_mass_teacher",
        "mean_dense_mass_ratio",
        "random_mass_at_k",
    )
    for (variant, scope), group_rows in sorted(groups.items()):
        out: dict[str, Any] = {
            "variant": variant,
            "query_scope": scope,
            "rows": len(group_rows),
            "sample_layers": len(group_rows),
            "queries_mean": finite_mean(row["queries"] for row in group_rows),
            "topk_mean": finite_mean(row["topk"] for row in group_rows),
        }
        for key in metric_keys:
            out[key] = finite_mean(row[key] for row in group_rows)
        aggregate.append(out)
    return aggregate


def write_markdown(path: Path, aggregate: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# GeoWeave Top-K vs Dense Attention Alignment",
        "",
        f"- Generated at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"- Protocol: `{Path(args.protocol).expanduser().resolve()}`",
        f"- Variants: {', '.join(args.variants)}",
        f"- Query stride: {int(args.query_token_stride)}",
        f"- Max query tokens per layer/scope: {int(args.max_query_tokens)}",
        f"- Layer rows: {len(rows)}",
        "",
        "| Variant | Query scope | Rows | Recall@K | Jaccard@K | Dense mass on GeoWeave Top-K | Dense mass on dense Top-K | Mass ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {variant} | {scope} | {rows} | {recall:.6f} | {jaccard:.6f} | {mass_l:.6f} | {mass_t:.6f} | {ratio:.6f} |".format(
                variant=row["variant"],
                scope=row["query_scope"],
                rows=int(row["rows"]),
                recall=float(row["mean_recall_at_k"]),
                jaccard=float(row["mean_jaccard_at_k"]),
                mass_l=float(row["mean_dense_mass_learned"]),
                mass_t=float(row["mean_dense_mass_teacher"]),
                ratio=float(row["mean_dense_mass_ratio"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    samples = list(protocol.get("samples", []))
    if int(args.sample_limit) > 0:
        samples = samples[: int(args.sample_limit)]

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VGGT_DSA_SPARSE_STREAM_RECORD_LAST"] = "1"

    model, loaded_checkpoint = setup_model(args)
    patched = install_alignment_capture(
        model,
        prefix_size=int(protocol.get("prefix_size", 6)),
        args=args,
    )

    rows: list[dict[str, Any]] = []
    for sample in samples:
        for variant in args.variants:
            if variant not in sample.get("variants", {}):
                continue
            rows.extend(collect_alignment_rows(model=model, sample=sample, variant=str(variant), args=args))

    aggregate = aggregate_rows(rows)
    csv_path = output_dir / "topk_alignment_rows.csv"
    json_path = output_dir / "topk_alignment_rows.json"
    aggregate_path = output_dir / "topk_alignment_aggregate.json"
    markdown_path = output_dir / "topk_alignment_summary.md"
    write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "protocol": str(Path(args.protocol).expanduser().resolve()),
                "loaded_checkpoint": loaded_checkpoint,
                "patched_attention_modules": int(patched),
                "sample_count": len(samples),
                "variants": list(args.variants),
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    aggregate_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(markdown_path, aggregate, rows, args)
    print(f"[topk-alignment] patched_attention_modules={patched}")
    print(f"[topk-alignment] wrote {csv_path}")
    print(f"[topk-alignment] wrote {aggregate_path}")
    print(f"[topk-alignment] wrote {markdown_path}")


if __name__ == "__main__":
    main()
