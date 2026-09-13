#!/usr/bin/env python3
"""Small adapters for evaluating a local VGGT-Omega checkout with meshx scripts."""

from __future__ import annotations

import sys
import os
from types import MethodType
from pathlib import Path
from typing import Any, Sequence

import numpy as np

VGGT_OMEGA_GLOBAL_ATTENTION_MODES = ("original", "camera_register_query", "query_view_image_global")


def _omega_debug(message: str) -> None:
    enabled = os.environ.get("VGGT_OMEGA_EVAL_DEBUG", "").strip().lower()
    if enabled in {"1", "true", "yes", "on"}:
        print(f"[vggt-omega-eval] {message}", flush=True)


def ensure_vggt_omega_repo(repo_path: str | Path) -> Path:
    repo = Path(repo_path).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"VGGT-Omega repo not found: {repo}")
    if not (repo / "vggt_omega").is_dir():
        raise FileNotFoundError(f"VGGT-Omega package not found under repo: {repo / 'vggt_omega'}")
    repo_text = str(repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    return repo


def load_vggt_omega_model(
    *,
    repo_path: str | Path,
    checkpoint_path: str | Path,
    device: Any,
    global_attention_mode: str = "original",
) -> tuple[Any, str]:
    ensure_vggt_omega_repo(repo_path)

    import torch
    from vggt_omega.models import VGGTOmega

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint}")

    _omega_debug("constructing VGGTOmega on meta device")
    try:
        with torch.device("meta"):
            model = VGGTOmega()
        use_assign = True
    except Exception as exc:
        _omega_debug(f"meta construction failed ({exc!r}); falling back to normal construction")
        model = VGGTOmega()
        use_assign = False

    _omega_debug(f"loading checkpoint from {checkpoint}")
    state = torch.load(str(checkpoint), map_location="cpu")
    _omega_debug("loading checkpoint state_dict into model")
    if use_assign:
        try:
            model.load_state_dict(state, strict=True, assign=True)
        except TypeError:
            _omega_debug("load_state_dict(assign=True) unavailable; reconstructing normally")
            model = VGGTOmega()
            model.load_state_dict(state, strict=True)
    else:
        model.load_state_dict(state, strict=True)

    _omega_debug(f"moving model to {device}")
    model = model.to(device).eval()
    configure_vggt_omega_attention_experiment(model, global_attention_mode=global_attention_mode)
    _omega_debug("model ready")
    return model, str(checkpoint)


def camera_register_query_positions(*, num_frames: int, num_tokens: int, patch_token_start: int) -> list[int]:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    if patch_token_start <= 0 or patch_token_start > num_tokens:
        raise ValueError(f"patch_token_start must be in [1, {num_tokens}], got {patch_token_start}")
    return [
        frame_idx * num_tokens + token_idx
        for frame_idx in range(num_frames)
        for token_idx in range(patch_token_start)
    ]


def query_view_image_global_query_positions(
    *,
    num_frames: int,
    num_tokens: int,
    patch_token_start: int,
    query_view_index: int,
) -> list[int]:
    if query_view_index < 0 or query_view_index >= num_frames:
        raise ValueError(f"query_view_index must be in [0, {num_frames - 1}], got {query_view_index}")
    positions = camera_register_query_positions(
        num_frames=num_frames,
        num_tokens=num_tokens,
        patch_token_start=patch_token_start,
    )
    view_start = int(query_view_index) * num_tokens
    positions.extend(range(view_start + patch_token_start, view_start + num_tokens))
    return sorted(set(positions))


def set_vggt_omega_query_view_index(model: Any, query_view_index: int) -> None:
    aggregator = getattr(model, "aggregator", None)
    if aggregator is None:
        raise AttributeError("VGGT-Omega model has no aggregator attribute")
    setattr(aggregator, "_meshx_query_view_index", int(query_view_index))


def configure_vggt_omega_attention_experiment(model: Any, *, global_attention_mode: str = "original") -> str:
    mode = str(global_attention_mode or "original")
    if mode not in VGGT_OMEGA_GLOBAL_ATTENTION_MODES:
        raise ValueError(
            f"Unsupported VGGT-Omega global attention mode: {mode}. "
            f"Expected one of {VGGT_OMEGA_GLOBAL_ATTENTION_MODES}."
        )

    aggregator = getattr(model, "aggregator", None)
    if aggregator is None:
        raise AttributeError("VGGT-Omega model has no aggregator attribute")

    original_name = "_meshx_original_run_inter_frame_attention_block"
    if not hasattr(aggregator, original_name):
        setattr(aggregator, original_name, aggregator._run_inter_frame_attention_block)

    if mode == "original":
        aggregator._run_inter_frame_attention_block = getattr(aggregator, original_name)
    elif mode == "camera_register_query":
        aggregator._run_inter_frame_attention_block = MethodType(
            _run_inter_frame_attention_camera_register_query,
            aggregator,
        )
    elif mode == "query_view_image_global":
        if not hasattr(aggregator, "_meshx_query_view_index"):
            setattr(aggregator, "_meshx_query_view_index", 0)
        aggregator._run_inter_frame_attention_block = MethodType(
            _run_inter_frame_attention_query_view_image_global,
            aggregator,
        )

    aggregator._meshx_global_attention_mode = mode
    return mode


def _run_inter_frame_attention_camera_register_query(
    self,
    tokens,
    batch_size: int,
    num_frames: int,
    num_tokens: int,
    embed_dim: int,
    block_idx: int,
    attention_type: str,
):
    if attention_type != "global":
        return self._meshx_original_run_inter_frame_attention_block(
            tokens,
            batch_size,
            num_frames,
            num_tokens,
            embed_dim,
            block_idx,
            attention_type,
        )

    import torch

    patch_token_start = int(self.patch_token_start)
    tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)
    flat_tokens = tokens.reshape(batch_size, num_frames * num_tokens, embed_dim)
    query_positions = torch.as_tensor(
        camera_register_query_positions(
            num_frames=num_frames,
            num_tokens=num_tokens,
            patch_token_start=patch_token_start,
        ),
        device=flat_tokens.device,
        dtype=torch.long,
    )
    query_tokens = flat_tokens.index_select(dim=1, index=query_positions)
    updated_query_tokens = _run_query_subset_self_attention_block(
        block=self.inter_frame_blocks[block_idx],
        query_tokens=query_tokens,
        context_tokens=flat_tokens,
    )
    output = flat_tokens.clone()
    output[:, query_positions, :] = updated_query_tokens
    return output.view(batch_size, num_frames, num_tokens, embed_dim)


def _run_inter_frame_attention_query_view_image_global(
    self,
    tokens,
    batch_size: int,
    num_frames: int,
    num_tokens: int,
    embed_dim: int,
    block_idx: int,
    attention_type: str,
):
    if attention_type != "global":
        return self._meshx_original_run_inter_frame_attention_block(
            tokens,
            batch_size,
            num_frames,
            num_tokens,
            embed_dim,
            block_idx,
            attention_type,
        )

    import torch

    patch_token_start = int(self.patch_token_start)
    query_view_index = int(getattr(self, "_meshx_query_view_index", 0))
    tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)
    flat_tokens = tokens.reshape(batch_size, num_frames * num_tokens, embed_dim)
    query_positions = torch.as_tensor(
        query_view_image_global_query_positions(
            num_frames=num_frames,
            num_tokens=num_tokens,
            patch_token_start=patch_token_start,
            query_view_index=query_view_index,
        ),
        device=flat_tokens.device,
        dtype=torch.long,
    )
    query_tokens = flat_tokens.index_select(dim=1, index=query_positions)
    updated_query_tokens = _run_query_subset_self_attention_block(
        block=self.inter_frame_blocks[block_idx],
        query_tokens=query_tokens,
        context_tokens=flat_tokens,
    )
    output = flat_tokens.clone()
    output[:, query_positions, :] = updated_query_tokens
    return output.view(batch_size, num_frames, num_tokens, embed_dim)


def _run_query_subset_self_attention_block(*, block: Any, query_tokens: Any, context_tokens: Any) -> Any:
    query_norm = block.norm1(query_tokens)
    context_norm = block.norm1(context_tokens)
    residual = _run_query_subset_attention(attn=block.attn, query_tokens=query_norm, context_tokens=context_norm)
    query_attn = query_tokens + block.ls1(residual)
    return query_attn + block.ls2(block.mlp(block.norm2(query_attn)))


def _run_query_subset_attention(*, attn: Any, query_tokens: Any, context_tokens: Any) -> Any:
    import torch

    batch_size, query_count, _ = query_tokens.shape
    context_count = context_tokens.shape[1]
    channels = attn.qkv.in_features
    num_heads = attn.num_heads
    head_dim = channels // num_heads

    query_qkv = attn.qkv(query_tokens).reshape(batch_size, query_count, 3, num_heads, head_dim)
    context_qkv = attn.qkv(context_tokens).reshape(batch_size, context_count, 3, num_heads, head_dim)
    q = query_qkv[:, :, 0].transpose(1, 2)
    k = context_qkv[:, :, 1].transpose(1, 2)
    v = context_qkv[:, :, 2].transpose(1, 2)
    if getattr(attn, "use_qk_norm", False):
        q = attn.q_norm(q)
        k = attn.k_norm(k)
    output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    output = output.transpose(1, 2).reshape(batch_size, query_count, channels)
    output = attn.proj(output)
    if hasattr(attn, "proj_drop"):
        output = attn.proj_drop(output)
    return output


def predict_vggt_omega(
    *,
    image_files: Sequence[str],
    model: Any,
    repo_path: str | Path,
    image_resolution: int,
    mode: str,
    device: Any,
) -> tuple[dict[str, Any], Any, Any]:
    ensure_vggt_omega_repo(repo_path)

    import torch
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    images = load_and_preprocess_images(
        [str(path) for path in image_files],
        mode=mode,
        image_resolution=int(image_resolution),
    ).to(device)
    with torch.inference_mode():
        predictions = model(images)
    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    return predictions, extrinsics, intrinsics


def as_homogeneous_w2c(extrinsics: Any) -> np.ndarray:
    if hasattr(extrinsics, "detach"):
        extrinsics = extrinsics.detach().float().cpu().numpy()
    arr = np.asarray(extrinsics, dtype=np.float32)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for VGGT-Omega extrinsics, got {arr.shape}")
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[1:] != (3, 4):
        raise ValueError(f"Expected VGGT-Omega extrinsics shape (N,3,4), got {arr.shape}")
    hom = np.tile(np.eye(4, dtype=np.float32), (arr.shape[0], 1, 1))
    hom[:, :3, :4] = arr
    return hom


def depth_to_world_points(depth: Any, extrinsics: Any, intrinsics: Any) -> np.ndarray:
    if hasattr(depth, "detach"):
        depth = depth.detach().float().cpu().numpy()
    if hasattr(extrinsics, "detach"):
        extrinsics = extrinsics.detach().float().cpu().numpy()
    if hasattr(intrinsics, "detach"):
        intrinsics = intrinsics.detach().float().cpu().numpy()

    depth_arr = np.asarray(depth, dtype=np.float32)
    extri_arr = np.asarray(extrinsics, dtype=np.float32)
    intri_arr = np.asarray(intrinsics, dtype=np.float32)
    if depth_arr.ndim == 5:
        if depth_arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for VGGT-Omega depth, got {depth_arr.shape}")
        depth_arr = depth_arr[0]
    if extri_arr.ndim == 4:
        if extri_arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for VGGT-Omega extrinsics, got {extri_arr.shape}")
        extri_arr = extri_arr[0]
    if intri_arr.ndim == 4:
        if intri_arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for VGGT-Omega intrinsics, got {intri_arr.shape}")
        intri_arr = intri_arr[0]
    if depth_arr.ndim != 4 or depth_arr.shape[-1] != 1:
        raise ValueError(f"Expected depth shape (N,H,W,1), got {depth_arr.shape}")
    if extri_arr.ndim != 3 or extri_arr.shape[1:] != (3, 4):
        raise ValueError(f"Expected extrinsics shape (N,3,4), got {extri_arr.shape}")
    if intri_arr.ndim != 3 or intri_arr.shape[1:] != (3, 3):
        raise ValueError(f"Expected intrinsics shape (N,3,3), got {intri_arr.shape}")

    depth_hw = depth_arr[..., 0]
    num_frames, height, width = depth_hw.shape
    if extri_arr.shape[0] != num_frames or intri_arr.shape[0] != num_frames:
        raise ValueError(
            f"Frame count mismatch: depth={depth_hw.shape}, extrinsics={extri_arr.shape}, intrinsics={intri_arr.shape}"
        )

    y, x = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))
    fx = intri_arr[:, 0, 0][:, None, None]
    fy = intri_arr[:, 1, 1][:, None, None]
    cx = intri_arr[:, 0, 2][:, None, None]
    cy = intri_arr[:, 1, 2][:, None, None]
    camera_points = np.stack(
        [
            (x - cx) / fx * depth_hw,
            (y - cy) / fy * depth_hw,
            depth_hw,
        ],
        axis=-1,
    )
    rotation = extri_arr[:, :3, :3]
    translation = extri_arr[:, :3, 3]
    world = np.einsum(
        "nij,nhwj->nhwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )
    return world.astype(np.float32, copy=False)
