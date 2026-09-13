#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

from easyvolcap.engine import Config
from easyvolcap.engine import io


REPO_ROOT = Path(__file__).resolve().parents[3]
EVC_ROOT = REPO_ROOT / "easyvolcap"
LEGACY_BASE_KEY = "_base_"


def _resolve_missing_absolute_repo_base(base_path: Path) -> Path | None:
    parts = list(base_path.parts)
    for idx, part in enumerate(parts):
        if part == "meshx" or part.startswith("meshx_"):
            candidate = REPO_ROOT.joinpath(*parts[idx + 1 :]).resolve()
            if candidate.exists():
                return candidate
    return None


def _resolve_base_config_path(cfg_path: Path, base_ref: str) -> Path:
    base_path = Path(base_ref)
    if base_path.is_absolute():
        if not base_path.exists():
            repo_relative_path = _resolve_missing_absolute_repo_base(base_path)
            if repo_relative_path is None:
                raise FileNotFoundError(f"Base config not found: {base_path}")
            return repo_relative_path
        return base_path

    candidates = [
        (cfg_path.parent / base_ref),
        (EVC_ROOT / base_ref),
        (REPO_ROOT / base_ref),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Base config not found for {cfg_path}: {base_ref}")


def _load_legacy_yaml_config(cfg_path: Path) -> Dict[str, Any]:
    cfg_dict = io.load(str(cfg_path))
    base_refs = cfg_dict.pop(LEGACY_BASE_KEY, None)
    if not base_refs:
        return cfg_dict

    base_refs = base_refs if isinstance(base_refs, list) else [base_refs]
    merged_base: Dict[str, Any] = {}
    for base_ref in base_refs:
        base_path = _resolve_base_config_path(cfg_path, base_ref)
        base_cfg = load_resolved_config(base_path)
        merged_base = Config._merge_a_into_b(dict(base_cfg), merged_base)
    return Config._merge_a_into_b(cfg_dict, merged_base)


def load_resolved_config(cfg_path: str | Path) -> Config:
    cfg_path = Path(cfg_path).resolve()
    cfg = Config.fromfile(str(cfg_path))
    if LEGACY_BASE_KEY not in cfg:
        return cfg

    if cfg_path.suffix not in {".yml", ".yaml", ".json"}:
        raise ValueError(f"Legacy _base_ expansion only supports yaml/json configs: {cfg_path}")

    merged_cfg = _load_legacy_yaml_config(cfg_path)
    return Config(merged_cfg, filename=str(cfg_path))
