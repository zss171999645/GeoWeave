#!/usr/bin/env python3
"""Shared Pi3 checkpoint loading helpers for official and native-sparse eval."""

from __future__ import annotations

import os
import sys
import inspect
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NATIVE_PI3_ROOT = REPO_ROOT / "aidi" / "third_party" / "pi3_training"


def _as_path(text: str | os.PathLike[str] | None) -> Path | None:
    if text is None:
        return None
    value = str(text).strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def find_hydra_config_for_checkpoint(ckpt: str | os.PathLike[str]) -> Path | None:
    path = Path(ckpt).expanduser()
    if path.is_file():
        path = path.parent
    for parent in (path, *path.parents):
        candidate = parent / ".hydra" / "config.yaml"
        if candidate.is_file():
            return candidate
    return None


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _is_native_config(config_path: Path) -> bool:
    try:
        payload = load_yaml(config_path)
    except OSError:
        return False
    model_cfg = payload.get("model", {})
    if not isinstance(model_cfg, Mapping):
        return False
    target = str(model_cfg.get("_target_", ""))
    if "pi3_training.Pi3" in target:
        return True
    indexer_cfg = model_cfg.get("indexer_cfg", {})
    return isinstance(indexer_cfg, Mapping) and bool(indexer_cfg.get("enabled", False))


def resolve_pi3_eval_options(
    ckpt: str,
    *,
    model_impl: str | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> Tuple[str, Path | None]:
    impl = (model_impl or os.environ.get("PI3_MODEL_IMPL", "")).strip().lower()
    env_config = _as_path(config_path) or _as_path(os.environ.get("PI3_CONFIG"))
    inferred_config = env_config or find_hydra_config_for_checkpoint(ckpt)

    if impl in {"native", "native_sparse", "pi3_training", "training"}:
        return "native_sparse", inferred_config
    if impl in {"official", "standard", "hf"}:
        return "official", env_config
    if inferred_config is not None and _is_native_config(inferred_config):
        return "native_sparse", inferred_config
    return "official", env_config


def is_native_pi3_requested(ckpt: str, *, model_impl: str | None = None, config_path: str | None = None) -> bool:
    impl, _ = resolve_pi3_eval_options(ckpt, model_impl=model_impl, config_path=config_path)
    return impl == "native_sparse"


def _load_checkpoint_state_dict(ckpt_path: Path, map_location: Any) -> Dict[str, Any]:
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(ckpt_path))
    else:
        import torch

        state = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
    if isinstance(state, Mapping):
        for key in ("state_dict", "model", "module"):
            value = state.get(key)
            if isinstance(value, Mapping):
                state = value
                break
    if not isinstance(state, Mapping):
        raise ValueError(f"Unsupported checkpoint payload type for {ckpt_path}: {type(state)!r}")
    return {str(k): v for k, v in state.items()}


def _strip_module_prefix(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    if not state_dict:
        return {}
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return dict(state_dict)


def build_native_pi3_kwargs(config_path: Path) -> Dict[str, Any]:
    payload = load_yaml(config_path)
    model_cfg = payload.get("model", {})
    if not isinstance(model_cfg, Mapping):
        raise ValueError(f"Missing model mapping in {config_path}")

    allowed_keys = {
        "pos_type",
        "decoder_size",
        "load_vggt",
        "freeze_encoder",
        "enable_point",
        "enable_camera",
        "use_global_points",
        "train_conf",
        "num_dec_blk_not_to_checkpoint",
        "qk_norm_chunk_size",
        "head_use_checkpoint",
        "head_view_chunk_size",
        "indexer_cfg",
        "encoder_attn_backend",
        "decoder_attn_backend",
    }
    kwargs = {key: value for key, value in dict(model_cfg).items() if key in allowed_keys}
    kwargs.setdefault("load_vggt", False)
    return kwargs


def filter_native_pi3_kwargs_for_class(kwargs: Mapping[str, Any], pi3_cls: Any) -> Dict[str, Any]:
    """Keep eval loading compatible with native Pi3 variants from different commits."""
    try:
        signature = inspect.signature(pi3_cls.__init__)
    except (TypeError, ValueError):
        return dict(kwargs)
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if accepts_var_kwargs:
        return dict(kwargs)
    allowed = {name for name in signature.parameters if name != "self"}
    return {key: value for key, value in dict(kwargs).items() if key in allowed}


def _plain_state_dict(state: Any) -> Dict[str, Any]:
    if state is None:
        return {}
    if isinstance(state, Mapping):
        return {str(key): value for key, value in state.items()}
    return {
        key: getattr(state, key)
        for key in dir(state)
        if not key.startswith("_") and not callable(getattr(state, key))
    }


def _native_pi3_eval_step(indexer_cfg: Mapping[str, Any]) -> int:
    env_step = os.environ.get("PI3_INDEXER_EVAL_STEP", "").strip()
    if env_step:
        return int(env_step)
    warmup_steps = int(indexer_cfg.get("warmup_steps", 0) or 0)
    sparse_start = int(indexer_cfg.get("sparse_start_step", warmup_steps) or 0)
    return max(warmup_steps, sparse_start)


def configure_native_pi3_eval_indexer(model: Any, indexer_cfg: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Synchronize native Pi3 sparse eval state with the training config."""

    if not isinstance(indexer_cfg, Mapping) or not bool(indexer_cfg.get("enabled", False)):
        return {}
    if not hasattr(model, "set_indexer_state"):
        return {}

    mode = os.environ.get("PI3_INDEXER_EVAL_MODE", "auto").strip().lower() or "auto"
    if mode in {"off", "disable", "disabled", "none"}:
        state = {"enabled": False, "warmup": False, "sparse": False, "compute_loss": False}
        model.set_indexer_state(state)
        return dict(state)

    if not hasattr(model, "set_indexer_state_by_step"):
        return {}

    state = _plain_state_dict(model.set_indexer_state_by_step(_native_pi3_eval_step(indexer_cfg), training=False))
    state["compute_loss"] = False

    if mode in {"dense", "force_dense"}:
        state["enabled"] = True
        state["warmup"] = False
        state["sparse"] = False
        model.set_indexer_state(state)
    elif mode in {"sparse", "force_sparse"}:
        state["enabled"] = True
        state["warmup"] = False
        state["sparse"] = True
        model.set_indexer_state(state)
    elif mode not in {"auto", "default"}:
        raise ValueError(
            "Unsupported PI3_INDEXER_EVAL_MODE="
            f"{mode!r}; choose from auto, sparse, dense, off."
        )

    return dict(state)


def _format_indexer_state_for_log(state: Mapping[str, Any]) -> str:
    keys = ("enabled", "warmup", "sparse", "compute_loss", "topk", "head_chunk_size", "score_dtype")
    parts = [f"{key}={state[key]!r}" for key in keys if key in state]
    return "{" + ", ".join(parts) + "}"


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _evict_foreign_pi3_modules(native_root: Path) -> None:
    native_root = native_root.resolve()
    for name, module in list(sys.modules.items()):
        if name != "pi3" and not name.startswith("pi3."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            module_path = Path(module_file).resolve()
        except OSError:
            continue
        if not _path_is_relative_to(module_path, native_root):
            del sys.modules[name]


def import_native_pi3_class(native_root: str | os.PathLike[str] | None = None):
    root = _as_path(native_root) or _as_path(os.environ.get("PI3_NATIVE_ROOT")) or DEFAULT_NATIVE_PI3_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"Native Pi3 root not found: {root}")
    _evict_foreign_pi3_modules(root)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from pi3.models.pi3_training import Pi3

    return Pi3, root


def load_pi3_model_for_eval(
    ckpt: str,
    device,
    *,
    official_pi3_cls=None,
    native_root: str | os.PathLike[str] | None = None,
    model_impl: str | None = None,
    config_path: str | os.PathLike[str] | None = None,
):
    impl, resolved_config = resolve_pi3_eval_options(ckpt, model_impl=model_impl, config_path=config_path)
    ckpt_path = Path(ckpt).expanduser()

    if impl == "native_sparse":
        if resolved_config is None:
            raise ValueError(
                "Native sparse Pi3 eval requires PI3_CONFIG/model.pi3_config or a checkpoint under outputs/*/.hydra/config.yaml"
            )
        Pi3, _ = import_native_pi3_class(native_root)
        kwargs = filter_native_pi3_kwargs_for_class(build_native_pi3_kwargs(resolved_config), Pi3)
        model = Pi3(**kwargs).to(device).eval()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Native Pi3 checkpoint not found: {ckpt_path}")
        state_dict = _strip_module_prefix(_load_checkpoint_state_dict(ckpt_path, map_location="cpu"))
        load_result = model.load_state_dict(state_dict, strict=False)
        loaded_indexer = any(".attn.indexer." in key for key in state_dict)
        if kwargs.get("indexer_cfg", {}).get("enabled", False) and not loaded_indexer:
            print(
                f"[pi3-loader] warning: native config enables indexer but no indexer weights were found in {ckpt_path}",
                file=sys.stderr,
                flush=True,
            )
        eval_indexer_state = configure_native_pi3_eval_indexer(model, kwargs.get("indexer_cfg", {}))
        print(
            f"[pi3-loader] loaded native_sparse ckpt={ckpt_path} config={resolved_config} "
            f"indexer_state={_format_indexer_state_for_log(eval_indexer_state)} result={load_result}",
            flush=True,
        )
        return model, str(ckpt_path)

    if official_pi3_cls is None:
        from easyvolcap.utils.pi3.models.pi3 import Pi3 as official_pi3_cls

    if ckpt_path.is_file():
        model = official_pi3_cls().to(device).eval()
        state_dict = _strip_module_prefix(_load_checkpoint_state_dict(ckpt_path, map_location=device))
        model.load_state_dict(state_dict)
        return model, str(ckpt_path)

    if not ckpt:
        ckpt = "yyfz233/Pi3"
    return official_pi3_cls.from_pretrained(ckpt).to(device).eval(), ckpt
