#!/usr/bin/env python3
"""Run Pi3/GeoWeave Top-K mechanism experiments on reproduced rebuttal data.

Subcommands:

selection-change
    Compare Pi3 base dense-attention Top-K selections from Pi3's own forward
    pass against GeoWeave scorer-selected Top-K selections from GeoWeave's own
    forward pass, restricted to the layers where GeoWeave is active.

base-topk-eval
    Evaluate Pi3 base with dense-attention Top-K pruning on the same pose
    protocols.  Two modes are supported:
      online: at each patched layer, select Top-K from current dense logits and
              softmax only over those keys in the same forward pass.
      two_pass: first run dense Pi3 to record Top-K masks; run the same input a
                second time using those fixed masks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import types
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
for _path in (REPO_ROOT, TOOLS_ROOT):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from reprodata_pi3_protocol_infer import (
    DEFAULT_BASE_CKPT,
    DEFAULT_GEOWEAVE_CKPT,
    DEFAULT_GEOWEAVE_CONFIG,
    load_model,
)
from reprodata_topk_alignment_analysis import finite_mean, query_indices_numpy


DEFAULT_PROTOCOL_ROOT = Path("/mnt/cfs/zhoufeng/geoweave_repro_inputs/rebuttal_allsettings_20260702")
DEFAULT_LAYER_SPEC = "9-17"
DEFAULT_TOPK = 1024
DEFAULT_SCOPES = ("all_patch_queries", "prefix_patch_queries")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sel = sub.add_parser("selection-change", help="Compare Pi3 dense Top-K and GeoWeave scorer Top-K.")
    sel.add_argument("--protocol", required=True)
    sel.add_argument("--output-dir", required=True)
    sel.add_argument("--setting-name", default="")
    sel.add_argument("--base-checkpoint", default=str(DEFAULT_BASE_CKPT))
    sel.add_argument("--geoweave-checkpoint", default=str(DEFAULT_GEOWEAVE_CKPT))
    sel.add_argument("--geoweave-config", default=str(DEFAULT_GEOWEAVE_CONFIG))
    sel.add_argument("--variants", nargs="*", required=True)
    sel.add_argument("--sample-limit", type=int, default=0)
    sel.add_argument("--sample-shard-index", type=int, default=0)
    sel.add_argument("--sample-num-shards", type=int, default=1)
    sel.add_argument("--max-images", type=int, default=10)
    sel.add_argument("--load-img-size", type=int, default=518)
    sel.add_argument("--device", default="cuda")
    sel.add_argument("--layer-spec", default=DEFAULT_LAYER_SPEC)
    sel.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    sel.add_argument("--query-token-stride", type=int, default=64)
    sel.add_argument("--max-query-tokens", type=int, default=64)
    sel.add_argument("--capture-query-chunk", type=int, default=128)
    sel.add_argument("--role-only", action="store_true")
    sel.add_argument("--save-topk-records", action="store_true")
    sel.add_argument("--topk-record-dir", default="")
    sel.add_argument(
        "--teacher-only",
        action="store_true",
        help="Only run GeoWeave and record dense-teacher/scorer Top-K diagnostics; skip Pi3 base forward.",
    )
    sel.add_argument("--scopes", nargs="*", default=list(DEFAULT_SCOPES))

    pose = sub.add_parser("base-topk-eval", help="Evaluate Pi3 base with dense Top-K masking.")
    pose.add_argument("--setting-name", required=True, choices=("scannetpp_weak", "waymo_weak", "waymo_plausible"))
    pose.add_argument("--protocol", required=True)
    pose.add_argument("--output-dir", required=True)
    pose.add_argument("--mode", required=True, choices=("online", "two_pass"))
    pose.add_argument("--base-checkpoint", default=str(DEFAULT_BASE_CKPT))
    pose.add_argument("--device", default="cuda")
    pose.add_argument("--load-img-size", type=int, default=518)
    pose.add_argument("--layer-spec", default=DEFAULT_LAYER_SPEC)
    pose.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    pose.add_argument("--query-chunk", type=int, default=128)
    pose.add_argument("--limit-seqs", type=int, default=0)
    pose.add_argument("--skip-plot", action="store_true", default=True)
    pose.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def load_protocol(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def layer_names_from_indexer_layers(spec: str) -> list[str]:
    names: list[str] = []
    for part in str(spec).split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            values = range(start, end + 1)
        else:
            values = range(int(token), int(token) + 1)
        for value in values:
            names.append(f"decoder.{2 * int(value) + 1}.attn")
    return names


def iter_protocol_samples(
    protocol: dict[str, Any],
    sample_limit: int,
    sample_shard_index: int = 0,
    sample_num_shards: int = 1,
) -> list[dict[str, Any]]:
    samples = list(protocol.get("samples", []))
    if int(sample_limit) > 0:
        samples = samples[: int(sample_limit)]
    num_shards = int(sample_num_shards)
    shard_index = int(sample_shard_index)
    if num_shards <= 0:
        raise ValueError(f"sample_num_shards must be positive, got {sample_num_shards}")
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"sample_shard_index must be in [0, {num_shards}), got {sample_shard_index}")
    if num_shards > 1:
        samples = [sample for idx, sample in enumerate(samples) if idx % num_shards == shard_index]
    return samples


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_record_slug(text: str, *, max_prefix: int = 96) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")
    digest = hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:10]
    prefix = cleaned[: int(max_prefix)] if cleaned else "record"
    return f"{prefix}_{digest}"


def write_topk_record_npz(
    *,
    record_root: Path,
    metadata: dict[str, Any],
    q_indices: np.ndarray,
    pi3_dense_topk: np.ndarray | None = None,
    geoweave_scorer_topk: np.ndarray | None = None,
    geoweave_dense_teacher_topk: np.ndarray | None = None,
) -> None:
    rel_dir = Path(
        str(metadata["setting"]),
        str(metadata["variant"]),
        str(metadata["layer"]).replace(".", "_"),
        str(metadata["query_scope"]),
    )
    filename = f"{safe_record_slug(str(metadata['sample_id']))}_b{int(metadata['batch_index']):02d}.npz"
    path = record_root / rel_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    q_out = np.asarray(q_indices, dtype=np.int32)
    arrays: dict[str, np.ndarray] = {"q_indices": q_out}
    if pi3_dense_topk is not None:
        arrays["pi3_dense_topk"] = np.asarray(pi3_dense_topk, dtype=np.int32)
    if geoweave_scorer_topk is not None:
        arrays["geoweave_scorer_topk"] = np.asarray(geoweave_scorer_topk, dtype=np.int32)
    if geoweave_dense_teacher_topk is not None:
        arrays["geoweave_dense_teacher_topk"] = np.asarray(geoweave_dense_teacher_topk, dtype=np.int32)
    if len(arrays) <= 1:
        raise ValueError("At least one Top-K array must be provided.")
    np.savez_compressed(path, **arrays)

    manifest_path = record_root / "topk_records_manifest.jsonl"
    row = dict(metadata)
    row.update(
        {
            "file": str(path.relative_to(record_root)),
            "q_indices_shape": list(q_out.shape),
            "dtype": "int32",
            "contains_dense_probabilities": False,
        }
    )
    for key, value in arrays.items():
        if key == "q_indices":
            continue
        row[f"{key}_shape"] = list(value.shape)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_topk_pair_numpy(
    base_topk: np.ndarray,
    geoweave_topk: np.ndarray,
    base_probs: np.ndarray,
) -> dict[str, Any]:
    base = np.asarray(base_topk, dtype=np.int64)
    geo = np.asarray(geoweave_topk, dtype=np.int64)
    probs = np.asarray(base_probs, dtype=np.float64)
    if base.ndim != 2 or geo.ndim != 2 or probs.ndim != 2:
        raise ValueError(f"Expected [Q,K], [Q,K], [Q,N], got {base.shape}, {geo.shape}, {probs.shape}")
    if base.shape != geo.shape or base.shape[0] != probs.shape[0]:
        raise ValueError(f"Mismatched topk/probability shapes: {base.shape}, {geo.shape}, {probs.shape}")

    key_count = int(probs.shape[1])
    overlap_counts: list[float] = []
    recalls: list[float] = []
    jaccards: list[float] = []
    base_masses: list[float] = []
    geoweave_masses: list[float] = []
    for query_idx in range(int(base.shape[0])):
        base_valid = base[query_idx][(base[query_idx] >= 0) & (base[query_idx] < key_count)]
        geo_valid = geo[query_idx][(geo[query_idx] >= 0) & (geo[query_idx] < key_count)]
        base_set = set(int(v) for v in base_valid.tolist())
        geo_set = set(int(v) for v in geo_valid.tolist())
        overlap = len(base_set & geo_set)
        union = len(base_set | geo_set)
        overlap_counts.append(float(overlap))
        recalls.append(float(overlap) / float(max(len(base_set), 1)))
        jaccards.append(float(overlap) / float(max(union, 1)))
        base_masses.append(float(probs[query_idx, base_valid].sum()) if base_valid.size else 0.0)
        geoweave_masses.append(float(probs[query_idx, geo_valid].sum()) if geo_valid.size else 0.0)

    mean_base_mass = finite_mean(base_masses)
    mean_geo_mass = finite_mean(geoweave_masses)
    return {
        "queries": int(base.shape[0]),
        "topk": int(base.shape[1]),
        "key_count": int(key_count),
        "mean_overlap_count": finite_mean(overlap_counts),
        "mean_recall_at_k": finite_mean(recalls),
        "mean_jaccard_at_k": finite_mean(jaccards),
        "mean_base_mass_on_base_topk": mean_base_mass,
        "mean_base_mass_on_geoweave_topk": mean_geo_mass,
        "mean_base_mass_ratio_geoweave_over_base": float(mean_geo_mass / mean_base_mass)
        if mean_base_mass > 0.0
        else float("nan"),
        "random_mass_at_k": float(base.shape[1] / key_count) if key_count > 0 else float("nan"),
    }


def infer_tokens_per_view(*, num_tokens: int, num_views: int) -> int:
    tokens = int(num_tokens)
    views = int(num_views)
    if views <= 0:
        raise ValueError(f"num_views must be positive, got {num_views}")
    if tokens <= 0 or tokens % views != 0:
        raise ValueError(f"Cannot infer tokens_per_view from num_tokens={num_tokens}, num_views={num_views}")
    return tokens // views


def frame_role_labels(frames: Sequence[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for frame in sorted(frames, key=lambda item: int(item.get("index", len(labels)))):
        group = frame.get("group")
        role = str(frame.get("role", "")).strip()
        if group is not None:
            labels.append(str(group).strip().upper())
        elif role == "group_a":
            labels.append("A")
        elif role == "group_b":
            labels.append("B")
        elif role:
            labels.append(role)
        else:
            labels.append("unknown")
    return labels


def _topk_view_labels(topk_row: np.ndarray, *, tokens_per_view: int, view_labels: Sequence[str]) -> list[str]:
    labels: list[str] = []
    for token in np.asarray(topk_row, dtype=np.int64).tolist():
        token_id = int(token)
        if token_id < 0:
            continue
        view_id = token_id // int(tokens_per_view)
        if 0 <= view_id < len(view_labels):
            labels.append(str(view_labels[view_id]))
    return labels


def summarize_same_view_group_ratio_numpy(
    *,
    topk: np.ndarray,
    q_indices: np.ndarray,
    tokens_per_view: int,
    view_labels: Sequence[str],
) -> dict[str, dict[str, Any]]:
    selected = np.asarray(topk, dtype=np.int64)
    queries = np.asarray(q_indices, dtype=np.int64)
    if selected.ndim != 2 or queries.ndim != 1 or selected.shape[0] != queries.shape[0]:
        raise ValueError(f"Expected topk [Q,K] and q_indices [Q], got {selected.shape}, {queries.shape}")

    ratios: dict[str, list[float]] = {}
    for row_idx, query_token in enumerate(queries.tolist()):
        query_view = int(query_token) // int(tokens_per_view)
        if not (0 <= query_view < len(view_labels)):
            continue
        query_group = str(view_labels[query_view])
        key_labels = _topk_view_labels(
            selected[row_idx],
            tokens_per_view=int(tokens_per_view),
            view_labels=view_labels,
        )
        if not key_labels:
            continue
        ratios.setdefault(query_group, []).append(
            float(sum(1 for label in key_labels if label == query_group)) / float(len(key_labels))
        )

    return {
        group: {
            "queries": len(values),
            "mean_same_view_group_ratio": finite_mean(values),
            "mean_cross_view_group_ratio": 1.0 - finite_mean(values),
        }
        for group, values in sorted(ratios.items())
    }


def summarize_role_target_ratio_numpy(
    *,
    topk: np.ndarray,
    q_indices: np.ndarray,
    tokens_per_view: int,
    view_labels: Sequence[str],
    target_roles: set[str],
) -> dict[str, Any]:
    selected = np.asarray(topk, dtype=np.int64)
    queries = np.asarray(q_indices, dtype=np.int64)
    if selected.ndim != 2 or queries.ndim != 1 or selected.shape[0] != queries.shape[0]:
        raise ValueError(f"Expected topk [Q,K] and q_indices [Q], got {selected.shape}, {queries.shape}")

    targets = {str(role) for role in target_roles}
    ratios: list[float] = []
    for row_idx in range(int(selected.shape[0])):
        key_labels = _topk_view_labels(
            selected[row_idx],
            tokens_per_view=int(tokens_per_view),
            view_labels=view_labels,
        )
        if not key_labels:
            continue
        ratios.append(float(sum(1 for label in key_labels if label in targets)) / float(len(key_labels)))

    return {
        "queries": len(ratios),
        "mean_target_role_ratio": finite_mean(ratios),
    }


def aggregate_numeric_rows(
    rows: Sequence[dict[str, Any]],
    *,
    group_keys: Sequence[str],
    metric_keys: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key, "") for key in group_keys), []).append(row)

    out: list[dict[str, Any]] = []
    for key_tuple, group in sorted(groups.items()):
        row = {key: value for key, value in zip(group_keys, key_tuple)}
        row["rows"] = len(group)
        for metric in metric_keys:
            values = [item[metric] for item in group if metric in item]
            if values:
                row[metric] = finite_mean(values)
        out.append(row)
    return out


def aggregate_metric_rows(
    rows: Sequence[dict[str, Any]],
    *,
    group_keys: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key, "") for key in group_keys), []).append(row)
    metric_keys = [
        "queries",
        "topk",
        "mean_overlap_count",
        "mean_recall_at_k",
        "mean_jaccard_at_k",
        "mean_base_mass_on_base_topk",
        "mean_base_mass_on_geoweave_topk",
        "mean_base_mass_ratio_geoweave_over_base",
        "random_mass_at_k",
    ]
    out: list[dict[str, Any]] = []
    for key_tuple, group in sorted(groups.items()):
        row = {key: value for key, value in zip(group_keys, key_tuple)}
        row["rows"] = len(group)
        for metric in metric_keys:
            values = [item[metric] for item in group if metric in item]
            if values:
                row[metric] = finite_mean(values)
        out.append(row)
    return out


def _selected_query_indices(
    *,
    num_tokens: int,
    prefix_size: int,
    scope: str,
    stride: int,
    max_queries: int,
) -> np.ndarray:
    tokens_per_view = num_tokens // 10 if num_tokens % 10 == 0 else max(num_tokens, 1)
    return query_indices_numpy(
        num_tokens=num_tokens,
        tokens_per_view=tokens_per_view,
        patch_start_idx=1,
        prefix_size=prefix_size,
        scope=scope,
        stride=stride,
        max_queries=max_queries,
    )


def dense_topk_for_query_indices(
    *,
    q: Any,
    k: Any,
    q_indices: Any,
    topk: int,
    query_chunk: int,
    include_probs: bool,
    attn_bias: Any | None = None,
) -> tuple[Any, Any | None]:
    import torch

    if int(query_chunk) <= 0:
        raise ValueError(f"query_chunk must be positive, got {query_chunk}")
    if int(topk) <= 0:
        raise ValueError(f"topk must be positive, got {topk}")

    topk_chunks: list[torch.Tensor] = []
    prob_chunks: list[torch.Tensor] = []
    for start in range(0, int(q_indices.numel()), int(query_chunk)):
        chunk_indices = q_indices[start : start + int(query_chunk)]
        if int(chunk_indices.numel()) == 0:
            continue
        q_sample = q.index_select(2, chunk_indices)
        scores = torch.einsum("bhqd,bhkd->bhqk", q_sample, k)
        if attn_bias is not None:
            bias = attn_bias.index_select(2, chunk_indices)
            scores = scores + bias
        dense_probs = scores.float().softmax(dim=-1).mean(dim=1)
        k_value = min(int(topk), int(dense_probs.shape[-1]))
        _scores, topk_indices = torch.topk(dense_probs, k=k_value, dim=-1, largest=True, sorted=False)
        topk_chunks.append(topk_indices.detach().cpu())
        if include_probs:
            prob_chunks.append(dense_probs.detach().cpu())
        del q_sample, scores, dense_probs, topk_indices

    if not topk_chunks:
        empty_topk = torch.empty((int(q.shape[0]), 0, 0), dtype=torch.long)
        empty_probs = torch.empty((int(q.shape[0]), 0, int(k.shape[2])), dtype=torch.float32) if include_probs else None
        return empty_topk, empty_probs

    topk_out = torch.cat(topk_chunks, dim=1)
    probs_out = torch.cat(prob_chunks, dim=1) if include_probs else None
    return topk_out, probs_out


def _base_capture_forward(self, x, attn_bias=None, xpos=None):
    import torch

    bsz, num_tokens, channels = x.shape
    qkv = self.qkv(x).reshape(bsz, num_tokens, 3, self.num_heads, channels // self.num_heads).transpose(1, 3)
    q, k, _v = [qkv[:, :, idx] for idx in range(3)]
    q = self.q_norm(q).to(qkv.dtype)
    k = self.k_norm(k).to(qkv.dtype)
    if self.rope is not None:
        q = self.rope(q, xpos)
        k = self.rope(k, xpos)
    q = q * float(self.scale)

    records: list[dict[str, Any]] = []
    for scope in getattr(self, "_topk_scopes", DEFAULT_SCOPES):
        q_indices_np = _selected_query_indices(
            num_tokens=int(num_tokens),
            prefix_size=int(getattr(self, "_topk_prefix_size", 6)),
            scope=str(scope),
            stride=int(getattr(self, "_topk_query_token_stride", 64)),
            max_queries=int(getattr(self, "_topk_max_query_tokens", 64)),
        )
        if q_indices_np.size == 0:
            continue
        q_indices = torch.as_tensor(q_indices_np, dtype=torch.long, device=x.device)
        topk = min(int(getattr(self, "_topk_k", DEFAULT_TOPK)), int(num_tokens))
        topk_indices, dense_probs = dense_topk_for_query_indices(
            q=q,
            k=k,
            q_indices=q_indices,
            topk=topk,
            query_chunk=int(getattr(self, "_topk_capture_query_chunk", 128)),
            include_probs=not bool(getattr(self, "_topk_role_only", False)),
        )
        records.append(
            {
                "scope": str(scope),
                "q_indices": q_indices.detach().cpu(),
                "base_topk": topk_indices.detach().cpu(),
                "num_tokens": int(num_tokens),
            }
        )
        if dense_probs is not None:
            records[-1]["base_probs"] = dense_probs.detach().cpu()
    self.last_base_dense_topk_records = records
    return self._topk_original_forward(x, attn_bias=attn_bias, xpos=xpos)


def _geoweave_capture_forward(self, x, pos=None):
    import torch

    out = self._topk_original_forward(x, pos=pos)
    learned = getattr(self, "last_topk_indices", None)
    records: list[dict[str, Any]] = []
    if isinstance(learned, torch.Tensor):
        learned_cpu = learned.detach().cpu()
        for scope in getattr(self, "_topk_scopes", DEFAULT_SCOPES):
            q_indices_np = _selected_query_indices(
                num_tokens=int(learned_cpu.shape[1]),
                prefix_size=int(getattr(self, "_topk_prefix_size", 6)),
                scope=str(scope),
                stride=int(getattr(self, "_topk_query_token_stride", 64)),
                max_queries=int(getattr(self, "_topk_max_query_tokens", 64)),
            )
            if q_indices_np.size == 0:
                continue
            q_indices = torch.as_tensor(q_indices_np, dtype=torch.long)
            records.append(
                {
                    "scope": str(scope),
                    "q_indices": q_indices,
                    "geoweave_topk": learned_cpu.index_select(1, q_indices),
                }
            )
    self.last_geoweave_topk_records = records
    self.last_geoweave_dense_teacher_topk_records = geoweave_dense_teacher_topk_records(
        self,
        x,
        pos=pos,
        scopes=getattr(self, "_topk_scopes", DEFAULT_SCOPES),
        prefix_size=int(getattr(self, "_topk_prefix_size", 6)),
        topk=int(getattr(self, "_topk_k", DEFAULT_TOPK)),
        query_token_stride=int(getattr(self, "_topk_query_token_stride", 64)),
        max_query_tokens=int(getattr(self, "_topk_max_query_tokens", 64)),
        capture_query_chunk=int(getattr(self, "_topk_capture_query_chunk", 128)),
    )
    return out


def geoweave_dense_teacher_topk_records(
    module: Any,
    x: Any,
    *,
    pos: Any | None,
    scopes: Sequence[str],
    prefix_size: int,
    topk: int,
    query_token_stride: int,
    max_query_tokens: int,
    capture_query_chunk: int,
) -> list[dict[str, Any]]:
    import torch

    bsz, num_tokens, channels = x.shape
    with torch.no_grad():
        qkv_flat = module.qkv(x)
        if hasattr(module, "_project_attention_qkv_from_flat"):
            q, k, _v = module._project_attention_qkv_from_flat(qkv_flat, int(bsz), int(num_tokens))
        else:
            qkv = qkv_flat.reshape(bsz, num_tokens, 3, module.num_heads, channels // module.num_heads).transpose(1, 3)
            q, k, _v = [qkv[:, :, idx] for idx in range(3)]
        q = module.q_norm(q)
        k = module.k_norm(k)
        if getattr(module, "rope", None) is not None and pos is not None:
            q = module.rope(q, pos)
            k = module.rope(k, pos)

        if hasattr(module, "_state"):
            state = module._state()
            attn_mask = state.get("attn_mask", None)
            if hasattr(module, "_normalize_mask"):
                attn_bias = module._normalize_mask(attn_mask, int(bsz))
            else:
                attn_bias = None
            if hasattr(module, "_resolve_score_dtype"):
                score_dtype = module._resolve_score_dtype(state, q.dtype)
                if score_dtype != q.dtype:
                    q = q.to(score_dtype)
                    k = k.to(score_dtype)
                    if attn_bias is not None and attn_bias.dtype != score_dtype:
                        attn_bias = attn_bias.to(score_dtype)
        else:
            attn_bias = None

        if hasattr(module, "_expand_attention_kv_heads"):
            k = module._expand_attention_kv_heads(k)
        scale = float(getattr(module, "scale", 1.0))
        q = q * scale

        records: list[dict[str, Any]] = []
        for scope in scopes:
            q_indices_np = _selected_query_indices(
                num_tokens=int(num_tokens),
                prefix_size=int(prefix_size),
                scope=str(scope),
                stride=int(query_token_stride),
                max_queries=int(max_query_tokens),
            )
            if q_indices_np.size == 0:
                continue
            q_indices = torch.as_tensor(q_indices_np, dtype=torch.long, device=x.device)
            topk_indices, _dense_probs = dense_topk_for_query_indices(
                q=q,
                k=k,
                q_indices=q_indices,
                topk=min(int(topk), int(num_tokens)),
                query_chunk=int(capture_query_chunk),
                include_probs=False,
                attn_bias=attn_bias,
            )
            records.append(
                {
                    "scope": str(scope),
                    "q_indices": q_indices.detach().cpu(),
                    "teacher_topk": topk_indices.detach().cpu(),
                    "num_tokens": int(num_tokens),
                }
            )
    return records


def install_selection_capture(
    model: Any,
    *,
    layer_names: Sequence[str],
    prefix_size: int,
    topk: int,
    scopes: Sequence[str],
    query_token_stride: int,
    max_query_tokens: int,
    capture_query_chunk: int,
    role_only: bool,
    geoweave: bool,
) -> int:
    modules = dict(model.named_modules())
    patched = 0
    for name in layer_names:
        module = modules.get(name)
        if module is None:
            continue
        if not hasattr(module, "_topk_original_forward"):
            module._topk_original_forward = module.forward
        module._topk_prefix_size = int(prefix_size)
        module._topk_k = int(topk)
        module._topk_scopes = tuple(str(scope) for scope in scopes)
        module._topk_query_token_stride = int(query_token_stride)
        module._topk_max_query_tokens = int(max_query_tokens)
        module._topk_capture_query_chunk = int(capture_query_chunk)
        module._topk_role_only = bool(role_only)
        module.forward = types.MethodType(_geoweave_capture_forward if geoweave else _base_capture_forward, module)
        patched += 1
    return patched


def collect_images(input_dir: Path, max_images: int) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(sorted(input_dir.glob(pattern)))
    paths = sorted(paths)
    if int(max_images) > 0:
        paths = paths[: int(max_images)]
    if not paths:
        raise FileNotFoundError(f"No images found in {input_dir}")
    return paths


def run_pointcloud_forward(model: Any, image_paths: list[Path], *, load_img_size: int, device: str) -> None:
    from PIL import Image
    from aidi.scripts.baselines.eval_pi3_mv_recon_core import infer_pi3_mv_pointclouds

    first = Image.open(image_paths[0]).convert("RGB")
    points = infer_pi3_mv_pointclouds(
        filelist=[str(path) for path in image_paths],
        model=model,
        load_img_size=int(load_img_size),
        device=str(device),
        verbose=False,
        data_size=(first.height, first.width),
        point_source="native",
    )
    del points


def collect_selection_records(model: Any, attr_name: str) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for name, module in model.named_modules():
        captured = getattr(module, attr_name, None)
        if not captured:
            continue
        for item in captured:
            records[(name, str(item["scope"]))] = item
        setattr(module, attr_name, [])
    return records


def append_role_ratio_rows(
    role_rows: list[dict[str, Any]],
    *,
    setting_name: str,
    sample_id: str,
    variant: str,
    layer: str,
    scope: str,
    batch_idx: int,
    model_name: str,
    topk: np.ndarray,
    q_indices: np.ndarray,
    tokens_per_view: int,
    view_labels: Sequence[str],
) -> None:
    label_set = {str(label) for label in view_labels}
    base_meta = {
        "setting": str(setting_name),
        "sample_id": str(sample_id),
        "variant": str(variant),
        "layer": str(layer),
        "query_scope": str(scope),
        "batch_index": int(batch_idx),
        "model": str(model_name),
        "tokens_per_view": int(tokens_per_view),
        "num_views": len(view_labels),
    }

    if {"A", "B"}.issubset(label_set):
        same_group = summarize_same_view_group_ratio_numpy(
            topk=topk,
            q_indices=q_indices,
            tokens_per_view=int(tokens_per_view),
            view_labels=view_labels,
        )
        for group, metrics in same_group.items():
            row = dict(base_meta)
            row.update(
                {
                    "experiment": "weak_same_view_group",
                    "target_name": "same_view_group",
                    "target_definition": f"query_group_{group}",
                    "query_view_group": str(group),
                    "queries": int(metrics["queries"]),
                    "mean_same_view_group_ratio": float(metrics["mean_same_view_group_ratio"]),
                    "mean_cross_view_group_ratio": float(metrics["mean_cross_view_group_ratio"]),
                }
            )
            role_rows.append(row)

    target_specs: list[tuple[str, str, set[str]]] = []
    for label in ("prefix", "clean_context", "distractor_context"):
        if label in label_set:
            target_specs.append((f"role_{label}", label, {label}))
    if "clean_context" in label_set:
        target_specs.append(("plausible_useful_context", "clean_context", {"clean_context"}))
    elif "distractor_context" in label_set and "prefix" in label_set:
        target_specs.append(("plausible_useful_context", "prefix_non_distractor", {"prefix"}))

    for target_name, target_definition, target_roles in target_specs:
        metrics = summarize_role_target_ratio_numpy(
            topk=topk,
            q_indices=q_indices,
            tokens_per_view=int(tokens_per_view),
            view_labels=view_labels,
            target_roles=target_roles,
        )
        row = dict(base_meta)
        row.update(
            {
                "experiment": "context_role_ratio",
                "target_name": str(target_name),
                "target_definition": str(target_definition),
                "queries": int(metrics["queries"]),
                "mean_target_role_ratio": float(metrics["mean_target_role_ratio"]),
            }
        )
        role_rows.append(row)


def run_selection_change(args: argparse.Namespace) -> None:
    import torch

    os.environ["VGGT_DSA_SPARSE_STREAM_RECORD_LAST"] = "1"
    protocol = load_protocol(args.protocol)
    setting_name = str(args.setting_name or Path(args.protocol).stem.replace("_protocol", ""))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    topk_record_root = (
        Path(args.topk_record_dir).expanduser().resolve()
        if str(args.topk_record_dir).strip()
        else output_dir / "topk_records"
    )
    if bool(args.save_topk_records):
        topk_record_root.mkdir(parents=True, exist_ok=True)
        record_manifest = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "protocol": str(Path(args.protocol).expanduser().resolve()),
            "setting": setting_name,
            "variants": list(args.variants),
            "topk": int(args.topk),
            "layer_spec": str(args.layer_spec),
            "query_token_stride": int(args.query_token_stride),
            "max_query_tokens": int(args.max_query_tokens),
            "capture_query_chunk": int(args.capture_query_chunk),
            "sample_limit": int(args.sample_limit),
            "sample_shard_index": int(args.sample_shard_index),
            "sample_num_shards": int(args.sample_num_shards),
            "role_only": bool(args.role_only),
            "teacher_only": bool(args.teacher_only),
            "note": "Records are diagnostics only. Model forward output is returned from the original attention forward.",
        }
        (topk_record_root / "run_manifest.json").write_text(
            json.dumps(record_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    layer_names = layer_names_from_indexer_layers(args.layer_spec)

    base_model = None
    base_ckpt = ""
    patched_base = 0
    if not bool(args.teacher_only):
        _, base_model, base_ckpt = load_model("base", Path(args.base_checkpoint), None, torch.device(args.device))
    _, geoweave_model, geoweave_ckpt = load_model(
        "geoweave",
        Path(args.geoweave_checkpoint),
        Path(args.geoweave_config),
        torch.device(args.device),
    )
    prefix_size = int(protocol.get("prefix_size", 6))
    if base_model is not None:
        patched_base = install_selection_capture(
            base_model,
            layer_names=layer_names,
            prefix_size=prefix_size,
            topk=int(args.topk),
            scopes=args.scopes,
            query_token_stride=int(args.query_token_stride),
            max_query_tokens=int(args.max_query_tokens),
            capture_query_chunk=int(args.capture_query_chunk),
            role_only=bool(args.role_only),
            geoweave=False,
        )
    patched_geoweave = install_selection_capture(
        geoweave_model,
        layer_names=layer_names,
        prefix_size=prefix_size,
        topk=int(args.topk),
        scopes=args.scopes,
        query_token_stride=int(args.query_token_stride),
        max_query_tokens=int(args.max_query_tokens),
        capture_query_chunk=int(args.capture_query_chunk),
        role_only=bool(args.role_only),
        geoweave=True,
    )
    expected_base = 0 if bool(args.teacher_only) else len(layer_names)
    if patched_base != expected_base or patched_geoweave != len(layer_names):
        raise RuntimeError(f"Patched base={patched_base}, geoweave={patched_geoweave}, expected {len(layer_names)}")

    rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    for sample in iter_protocol_samples(
        protocol,
        int(args.sample_limit),
        sample_shard_index=int(args.sample_shard_index),
        sample_num_shards=int(args.sample_num_shards),
    ):
        for variant in args.variants:
            payload = sample.get("variants", {}).get(variant)
            if payload is None:
                continue
            view_labels = frame_role_labels(payload.get("frames", []))
            if not view_labels:
                raise ValueError(f"Protocol sample {sample.get('sample_id')} variant {variant} has no frame labels")
            image_paths = collect_images(Path(payload["input_dir"]), int(args.max_images))
            base_records: dict[tuple[str, str], dict[str, Any]] = {}
            if base_model is not None:
                run_pointcloud_forward(
                    base_model,
                    image_paths,
                    load_img_size=int(args.load_img_size),
                    device=str(args.device),
                )
                base_records = collect_selection_records(base_model, "last_base_dense_topk_records")
            run_pointcloud_forward(
                geoweave_model,
                image_paths,
                load_img_size=int(args.load_img_size),
                device=str(args.device),
            )
            geoweave_records = collect_selection_records(geoweave_model, "last_geoweave_topk_records")
            teacher_records = collect_selection_records(geoweave_model, "last_geoweave_dense_teacher_topk_records")

            record_keys = sorted(geoweave_records.keys() if bool(args.teacher_only) else base_records.keys())
            for key in record_keys:
                base_record = base_records.get(key)
                geoweave_record = geoweave_records.get(key)
                if geoweave_record is None:
                    continue
                teacher_record = teacher_records.get(key)
                if teacher_record is None:
                    raise RuntimeError(f"Missing GeoWeave dense-teacher Top-K for {sample['sample_id']} {variant} {key}")
                layer, scope = key
                geoweave_topk = np.asarray(geoweave_record["geoweave_topk"])
                teacher_topk = np.asarray(teacher_record["teacher_topk"])
                q_indices = np.asarray(teacher_record["q_indices"], dtype=np.int64)
                geo_q_indices = np.asarray(geoweave_record["q_indices"], dtype=np.int64)
                if not np.array_equal(q_indices, geo_q_indices):
                    raise ValueError(
                        f"Mismatched teacher/scorer query indices for {sample['sample_id']} {variant} {layer} {scope}"
                    )
                base_topk = None
                base_q_indices = None
                if base_record is not None:
                    base_topk = np.asarray(base_record["base_topk"])
                    base_q_indices = np.asarray(base_record["q_indices"], dtype=np.int64)
                    if not np.array_equal(q_indices, base_q_indices):
                        raise ValueError(
                            f"Mismatched teacher/base query indices for {sample['sample_id']} {variant} {layer} {scope}"
                        )
                tokens_per_view = infer_tokens_per_view(
                    num_tokens=int(teacher_record["num_tokens"]),
                    num_views=len(view_labels),
                )
                topk_candidates = [int(geoweave_topk.shape[-1]), int(teacher_topk.shape[-1])]
                if base_topk is not None:
                    topk_candidates.append(int(base_topk.shape[-1]))
                topk = min(topk_candidates)
                batch_count = int(geoweave_topk.shape[0])
                if int(teacher_topk.shape[0]) != batch_count:
                    raise ValueError(f"Mismatched teacher/scorer batch count for {sample['sample_id']} {variant}")
                if base_topk is not None and int(base_topk.shape[0]) != batch_count:
                    raise ValueError(f"Mismatched base/scorer batch count for {sample['sample_id']} {variant}")
                for batch_idx in range(batch_count):
                    if bool(args.save_topk_records):
                        write_topk_record_npz(
                            record_root=topk_record_root,
                            metadata={
                                "setting": setting_name,
                                "sample_id": str(sample["sample_id"]),
                                "variant": str(variant),
                                "layer": str(layer),
                                "query_scope": str(scope),
                                "batch_index": int(batch_idx),
                                "tokens_per_view": int(tokens_per_view),
                                "num_views": len(view_labels),
                                "view_labels": list(view_labels),
                                "queries": int(q_indices.shape[0]),
                                "topk": int(topk),
                            },
                            q_indices=q_indices,
                            pi3_dense_topk=None if base_topk is None else base_topk[batch_idx, :, :topk],
                            geoweave_scorer_topk=geoweave_topk[batch_idx, :, :topk],
                            geoweave_dense_teacher_topk=teacher_topk[batch_idx, :, :topk],
                        )
                    if base_topk is not None and not bool(args.role_only):
                        base_probs = np.asarray(base_record["base_probs"])
                        metrics = summarize_topk_pair_numpy(
                            base_topk[batch_idx, :, :topk],
                            geoweave_topk[batch_idx, :, :topk],
                            base_probs[batch_idx],
                        )
                        metrics.update(
                            {
                                "setting": setting_name,
                                "sample_id": str(sample["sample_id"]),
                                "variant": str(variant),
                                "layer": str(layer),
                                "query_scope": str(scope),
                                "batch_index": int(batch_idx),
                            }
                        )
                        rows.append(metrics)
                    if base_topk is not None:
                        append_role_ratio_rows(
                            role_rows,
                            setting_name=setting_name,
                            sample_id=str(sample["sample_id"]),
                            variant=str(variant),
                            layer=str(layer),
                            scope=str(scope),
                            batch_idx=int(batch_idx),
                            model_name="pi3_dense_topk",
                            topk=base_topk[batch_idx, :, :topk],
                            q_indices=q_indices,
                            tokens_per_view=int(tokens_per_view),
                            view_labels=view_labels,
                        )
                    append_role_ratio_rows(
                        role_rows,
                        setting_name=setting_name,
                        sample_id=str(sample["sample_id"]),
                        variant=str(variant),
                        layer=str(layer),
                        scope=str(scope),
                        batch_idx=int(batch_idx),
                        model_name="geoweave_scorer_topk",
                        topk=geoweave_topk[batch_idx, :, :topk],
                        q_indices=q_indices,
                        tokens_per_view=int(tokens_per_view),
                        view_labels=view_labels,
                    )
                    append_role_ratio_rows(
                        role_rows,
                        setting_name=setting_name,
                        sample_id=str(sample["sample_id"]),
                        variant=str(variant),
                        layer=str(layer),
                        scope=str(scope),
                        batch_idx=int(batch_idx),
                        model_name="geoweave_dense_teacher_topk",
                        topk=teacher_topk[batch_idx, :, :topk],
                        q_indices=q_indices,
                        tokens_per_view=int(tokens_per_view),
                        view_labels=view_labels,
                    )
            torch.cuda.empty_cache()

    aggregate_by_setting = aggregate_metric_rows(rows, group_keys=("setting", "variant", "query_scope"))
    aggregate_by_layer = aggregate_metric_rows(rows, group_keys=("setting", "variant", "query_scope", "layer"))
    role_aggregate = aggregate_numeric_rows(
        role_rows,
        group_keys=(
            "experiment",
            "setting",
            "variant",
            "query_scope",
            "model",
            "target_name",
            "target_definition",
            "query_view_group",
        ),
        metric_keys=(
            "queries",
            "mean_same_view_group_ratio",
            "mean_cross_view_group_ratio",
            "mean_target_role_ratio",
        ),
    )
    write_csv(output_dir / "topk_change_rows.csv", rows)
    write_csv(output_dir / "topk_role_rows.csv", role_rows)
    (output_dir / "topk_change_rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "topk_role_rows.json").write_text(json.dumps(role_rows, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "topk_change_aggregate.json").write_text(
        json.dumps({"by_setting": aggregate_by_setting, "by_layer": aggregate_by_layer}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "topk_role_aggregate.json").write_text(
        json.dumps({"by_setting": role_aggregate}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_selection_markdown(
        output_dir / "topk_change_summary.md",
        aggregate_by_setting,
        args=args,
        base_ckpt=base_ckpt,
        geoweave_ckpt=geoweave_ckpt,
        patched_base=patched_base,
        patched_geoweave=patched_geoweave,
        rows=rows,
    )
    write_role_markdown(output_dir / "topk_role_summary.md", role_aggregate, args=args, rows=role_rows)
    print(f"[selection-change] rows={len(rows)} role_rows={len(role_rows)} output={output_dir}")


def write_selection_markdown(
    path: Path,
    aggregate: Sequence[dict[str, Any]],
    *,
    args: argparse.Namespace,
    base_ckpt: str,
    geoweave_ckpt: str,
    patched_base: int,
    patched_geoweave: int,
    rows: Sequence[dict[str, Any]],
) -> None:
    lines = [
        "# Pi3 Dense Top-K vs GeoWeave Top-K Selection Change",
        "",
        f"- Generated at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"- Protocol: `{Path(args.protocol).expanduser().resolve()}`",
        f"- Base checkpoint: `{base_ckpt}`",
        f"- GeoWeave checkpoint: `{geoweave_ckpt}`",
        f"- Patched layers: base={patched_base}, geoweave={patched_geoweave}",
        f"- Top-K: {int(args.topk)}",
        f"- Rows: {len(rows)}",
        "",
        "| Setting | Variant | Query scope | Rows | Recall@K | Jaccard@K | Base mass on base Top-K | Base mass on GeoWeave Top-K | Mass ratio |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {setting} | {variant} | {scope} | {rows} | {recall:.6f} | {jaccard:.6f} | {base_mass:.6f} | {geo_mass:.6f} | {ratio:.6f} |".format(
                setting=row.get("setting", ""),
                variant=row.get("variant", ""),
                scope=row.get("query_scope", ""),
                rows=int(row.get("rows", 0)),
                recall=float(row.get("mean_recall_at_k", float("nan"))),
                jaccard=float(row.get("mean_jaccard_at_k", float("nan"))),
                base_mass=float(row.get("mean_base_mass_on_base_topk", float("nan"))),
                geo_mass=float(row.get("mean_base_mass_on_geoweave_topk", float("nan"))),
                ratio=float(row.get("mean_base_mass_ratio_geoweave_over_base", float("nan"))),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_role_markdown(
    path: Path,
    aggregate: Sequence[dict[str, Any]],
    *,
    args: argparse.Namespace,
    rows: Sequence[dict[str, Any]],
) -> None:
    lines = [
        "# Top-K View-Group / Context Role Ratios",
        "",
        f"- Generated at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"- Protocol: `{Path(args.protocol).expanduser().resolve()}`",
        f"- Top-K: {int(args.topk)}",
        f"- Rows: {len(rows)}",
        "",
        "## Weak Same-View-Group Ratio",
        "",
        "| Setting | Variant | Query scope | Model | Query group | Rows | Same-group ratio | Cross-group ratio |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in aggregate:
        if row.get("experiment") != "weak_same_view_group":
            continue
        lines.append(
            "| {setting} | {variant} | {scope} | {model} | {group} | {rows} | {same:.6f} | {cross:.6f} |".format(
                setting=row.get("setting", ""),
                variant=row.get("variant", ""),
                scope=row.get("query_scope", ""),
                model=row.get("model", ""),
                group=row.get("query_view_group", ""),
                rows=int(row.get("rows", 0)),
                same=float(row.get("mean_same_view_group_ratio", float("nan"))),
                cross=float(row.get("mean_cross_view_group_ratio", float("nan"))),
            )
        )
    lines.extend(
        [
            "",
            "## Context Role Ratio",
            "",
            "| Setting | Variant | Query scope | Model | Target | Definition | Rows | Target ratio |",
            "|---|---|---|---|---|---|---:|---:|",
        ]
    )
    for row in aggregate:
        if row.get("experiment") != "context_role_ratio":
            continue
        lines.append(
            "| {setting} | {variant} | {scope} | {model} | {target} | {definition} | {rows} | {ratio:.6f} |".format(
                setting=row.get("setting", ""),
                variant=row.get("variant", ""),
                scope=row.get("query_scope", ""),
                model=row.get("model", ""),
                target=row.get("target_name", ""),
                definition=row.get("target_definition", ""),
                rows=int(row.get("rows", 0)),
                ratio=float(row.get("mean_target_role_ratio", float("nan"))),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _project_base_qkv(module: Any, x: Any):
    bsz, num_tokens, channels = x.shape
    qkv = module.qkv(x).reshape(bsz, num_tokens, 3, module.num_heads, channels // module.num_heads).transpose(1, 3)
    q, k, v = [qkv[:, :, idx] for idx in range(3)]
    q = module.q_norm(q).to(v.dtype)
    k = module.k_norm(k).to(v.dtype)
    return q, k, v


def _gather_kv(kv: Any, indices: Any) -> Any:
    bsz, heads, num_tokens, dim = kv.shape
    query_count = indices.shape[2]
    return kv.unsqueeze(2).expand(bsz, heads, query_count, num_tokens, dim).gather(
        3,
        indices.unsqueeze(-1).expand(bsz, heads, query_count, indices.shape[-1], dim),
    )


def topk_attention_chunked(
    q: Any,
    k: Any,
    v: Any,
    *,
    topk: int,
    scale: float,
    query_chunk: int,
    fixed_indices: Any | None = None,
) -> tuple[Any, Any]:
    import torch

    bsz, heads, num_queries, _dim = q.shape
    key_count = int(k.shape[2])
    k_value = max(1, min(int(topk), key_count))
    chunk = max(1, int(query_chunk))
    outputs = []
    indices_out = []
    for start in range(0, num_queries, chunk):
        end = min(start + chunk, num_queries)
        q_chunk = q[:, :, start:end, :]
        if fixed_indices is None:
            scores = torch.einsum("bhqd,bhkd->bhqk", q_chunk, k) * float(scale)
            summary = scores.float().mean(dim=1)
            _summary_scores, idx = torch.topk(summary, k=k_value, dim=-1, largest=True, sorted=False)
        else:
            idx = fixed_indices[:, start:end, :k_value].to(device=q.device, dtype=torch.long)
            gathered_k = _gather_kv(k, idx.unsqueeze(1).expand(-1, heads, -1, -1))
            scores = (q_chunk.unsqueeze(3) * gathered_k).sum(dim=-1) * float(scale)
        idx_heads = idx.unsqueeze(1).expand(-1, heads, -1, -1)
        if fixed_indices is None:
            selected_scores = scores.gather(-1, idx_heads)
        else:
            selected_scores = scores
        weights = torch.softmax(selected_scores.float(), dim=-1).to(v.dtype)
        gathered_v = _gather_kv(v, idx_heads)
        outputs.append((weights.unsqueeze(-1) * gathered_v).sum(dim=-2))
        indices_out.append(idx.detach().to(torch.int32).cpu())
    return torch.cat(outputs, dim=2), torch.cat(indices_out, dim=1)


def _base_online_or_capture_forward(self, x, attn_bias=None, xpos=None):
    if attn_bias is not None:
        raise ValueError("base Top-K patch does not support non-null attn_bias")
    q, k, v = _project_base_qkv(self, x)
    if self.rope is not None:
        q = self.rope(q, xpos)
        k = self.rope(k, xpos)

    mode = getattr(self, "_base_topk_mode", "online")
    if mode == "capture":
        out = self._base_topk_original_forward(x, attn_bias=attn_bias, xpos=xpos)
        _topk_out, topk_indices = topk_attention_chunked(
            q,
            k,
            v,
            topk=int(getattr(self, "_base_topk_k", DEFAULT_TOPK)),
            scale=float(getattr(self, "scale", 1.0)),
            query_chunk=int(getattr(self, "_base_topk_query_chunk", 128)),
        )
        self._base_topk_fixed_indices = topk_indices
        del _topk_out
        return out
    if mode == "fixed":
        fixed = getattr(self, "_base_topk_fixed_indices", None)
        if fixed is None:
            raise RuntimeError("two_pass fixed mode reached without captured Top-K indices")
        attn_out, _idx = topk_attention_chunked(
            q,
            k,
            v,
            topk=int(getattr(self, "_base_topk_k", DEFAULT_TOPK)),
            scale=float(getattr(self, "scale", 1.0)),
            query_chunk=int(getattr(self, "_base_topk_query_chunk", 128)),
            fixed_indices=fixed,
        )
    elif mode == "online":
        attn_out, topk_indices = topk_attention_chunked(
            q,
            k,
            v,
            topk=int(getattr(self, "_base_topk_k", DEFAULT_TOPK)),
            scale=float(getattr(self, "scale", 1.0)),
            query_chunk=int(getattr(self, "_base_topk_query_chunk", 128)),
        )
        self._base_topk_last_indices = topk_indices
    else:
        raise ValueError(f"Unknown base Top-K mode: {mode}")
    out = attn_out.transpose(1, 2).reshape(x.shape[0], x.shape[1], x.shape[2])
    out = self.proj(out)
    out = self.proj_drop(out)
    return out


def install_base_topk_patch(
    model: Any,
    *,
    layer_names: Sequence[str],
    mode: str,
    topk: int,
    query_chunk: int,
) -> int:
    modules = dict(model.named_modules())
    patched_modules = []
    for name in layer_names:
        module = modules.get(name)
        if module is None:
            continue
        if not hasattr(module, "_base_topk_original_forward"):
            module._base_topk_original_forward = module.forward
        module._base_topk_mode = str(mode)
        module._base_topk_k = int(topk)
        module._base_topk_query_chunk = int(query_chunk)
        module._base_topk_fixed_indices = None
        module.forward = types.MethodType(_base_online_or_capture_forward, module)
        patched_modules.append(module)
    if mode == "two_pass":
        original_model_forward = model.forward

        def _two_pass_model_forward(self, *forward_args, **forward_kwargs):
            for module in patched_modules:
                module._base_topk_mode = "capture"
                module._base_topk_fixed_indices = None
            with __import__("torch").no_grad():
                _ = original_model_forward(*forward_args, **forward_kwargs)
            for module in patched_modules:
                module._base_topk_mode = "fixed"
            return original_model_forward(*forward_args, **forward_kwargs)

        model.forward = types.MethodType(_two_pass_model_forward, model)
    return len(patched_modules)


def dataset_args_from_setting(setting_name: str, protocol: dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    source_root = str(Path(protocol["source_root"]).expanduser().resolve())
    ns = argparse.Namespace(
        datasets="scannetv2" if setting_name == "scannetpp_weak" else "vkitti2",
        sintel_root="",
        tum_root="",
        scannet_root=source_root if setting_name == "scannetpp_weak" else "",
        vkitti_root=source_root if setting_name != "scannetpp_weak" else "",
        model_path=str(Path(args.base_checkpoint).expanduser().resolve()),
        model_family="pi3",
        pi3_config="",
        pi3_model_impl="",
        pi3_native_root="aidi/third_party/pi3_training",
        device=str(args.device),
        load_img_size=int(args.load_img_size),
        pose_eval_stride=1,
        image_load_retries=8,
        image_load_retry_sleep=0.5,
        limit_seqs=int(args.limit_seqs),
        output_dir=str(Path(args.output_dir).expanduser().resolve()),
        eval_frame_indices="0,1,2,3,4,5" if setting_name == "waymo_plausible" else "",
        model_tag=f"pi3_base_dense_topk_{args.mode}",
        skip_plot=bool(args.skip_plot),
        require_official_layout=False,
        verbose=False,
        vggt_model_tag="official",
        vggt_config="",
        vggt_official_ckpt_root="",
        vggt_topk_override=0,
    )
    return ns


def run_base_topk_eval(args: argparse.Namespace) -> None:
    import torch
    from aidi.scripts.baselines import eval_pi3_relpose_distance_protocol as eval_pose

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(args.protocol)
    eval_args = dataset_args_from_setting(str(args.setting_name), protocol, args)
    device = eval_pose.resolve_device(str(args.device))
    _, model, loaded_ckpt = load_model("base", Path(args.base_checkpoint), None, torch.device(str(device)))
    layer_names = layer_names_from_indexer_layers(str(args.layer_spec))
    patched = install_base_topk_patch(
        model,
        layer_names=layer_names,
        mode=str(args.mode),
        topk=int(args.topk),
        query_chunk=int(args.query_chunk),
    )
    if patched != len(layer_names):
        raise RuntimeError(f"Patched {patched} base attention layers, expected {len(layer_names)}")
    dataset_names = eval_pose.parse_dataset_names(eval_args.datasets)
    results = [
        eval_pose.evaluate_dataset(
            dataset_name=dataset_name,
            args=eval_args,
            model=model,
            loaded_ckpt=f"{loaded_ckpt}::dense_topk_{args.mode}",
            output_root=output_dir,
        )
        for dataset_name in dataset_names
    ]
    summary = eval_pose.build_run_summary(results, dataset_names, f"{loaded_ckpt}::dense_topk_{args.mode}", output_dir)
    summary.update(
        {
            "setting_name": str(args.setting_name),
            "topk_mode": str(args.mode),
            "patched_layers": int(patched),
            "layer_names": layer_names,
            "topk": int(args.topk),
            "query_chunk": int(args.query_chunk),
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.cmd == "selection-change":
        run_selection_change(args)
    elif args.cmd == "base-topk-eval":
        run_base_topk_eval(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
