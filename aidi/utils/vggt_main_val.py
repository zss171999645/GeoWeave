from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class EvalCfgEntry:
    name: str
    cfg: str
    max_iter: Optional[int] = None


def _derive_entry_name(cfg_path: str) -> str:
    return Path(cfg_path).stem


def normalize_eval_cfg_entries(entries: Optional[Iterable]) -> List[EvalCfgEntry]:
    normalized: List[EvalCfgEntry] = []
    for entry in entries or []:
        if isinstance(entry, str):
            normalized.append(EvalCfgEntry(name=_derive_entry_name(entry), cfg=entry))
            continue

        if not isinstance(entry, dict):
            raise TypeError(f"Unsupported eval cfg entry type: {type(entry)!r}")

        cfg_path = entry.get("cfg") or entry.get("cfg_path")
        if not cfg_path:
            raise ValueError(f"Missing cfg/cfg_path in eval cfg entry: {entry}")

        name = entry.get("name") or _derive_entry_name(cfg_path)
        max_iter = entry.get("max_iter")
        normalized.append(EvalCfgEntry(name=name, cfg=cfg_path, max_iter=max_iter))

    return normalized
