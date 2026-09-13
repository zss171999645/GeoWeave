# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

try:
    from wcmatch import fnmatch
    _HAS_WCMATCH = True
except ImportError:
    import fnmatch as _fnmatch
    _HAS_WCMATCH = False
from functools import wraps
from typing import List

import torch.nn as nn

if _HAS_WCMATCH:
    GLOB_FLAGS = (
        fnmatch.CASE
        | fnmatch.DOTMATCH
        | fnmatch.EXTMATCH
        | fnmatch.SPLIT
    )
    def _match(name: str, pattern: str) -> bool:
        return fnmatch.fnmatch(name, pattern, flags=GLOB_FLAGS)
else:
    GLOB_FLAGS = None
    def _match(name: str, pattern: str) -> bool:
        return _fnmatch.fnmatchcase(name, pattern)


def freeze_modules(
    model: nn.Module,
    patterns: List[str],
    recursive: bool = True,
    lock_train: bool = True,
) -> nn.Module:
    """Freeze (stop training) parts of *model* whose *name* matches *patterns*.

    Parameters
    ----------
    model : nn.Module
        The complete model you are working with.
    patterns : list[str]
        Glob patterns to match sub-module names.  Example: ``["encoder.*", "cls_head"]``
    recursive : bool, default = True
        - ``True``  -> also freeze every child of a matched module.
        - ``False`` -> freeze only the matched module itself.
    lock_train : bool, default = True
        - ``True``  -> lock matched modules to eval mode via train() override.
        - ``False`` -> only toggle requires_grad, keep train/eval behavior.

    Returns
    -------
    nn.Module
        The same model object, now with some parts frozen.

    Example
    -------
    >>> freeze_modules(model, ["encoder.*", "decoder.layer1"], recursive=True)
    """
    matched: set[str] = set()

    for name, mod in model.named_modules():
        # does *name* match ANY user pattern?
        if any(_match(name, p) for p in patterns):
            matched.add(name)
            _freeze(mod, recursive, lock_train)

    _check_every_pattern_used(matched, patterns)
    return model


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def _freeze(mod: nn.Module, recursive: bool, lock_train: bool) -> None:
    """Optionally lock *mod* to eval mode and freeze its parameters."""

    if lock_train:
        if recursive:
            mod.eval()            # affects the whole subtree
        else:
            mod.training = False  # only this exact module

        original_train = mod.train

        @wraps(original_train)
        def locked_train(mode: bool = True):
            if recursive:
                return original_train(False)  # ignore user's *mode*
            out = original_train(mode)        # children follow user's choice
            out.training = False              # but this module stays frozen
            return out

        mod.train = locked_train  # type: ignore[attr-defined]

    param_iter = (
        mod.parameters()              # default recurse=True
        if recursive
        else mod.parameters(recurse=False)
    )
    for p in param_iter:
        p.requires_grad = False


def _check_every_pattern_used(matched_names: set[str], patterns: List[str]):
    unused = [p for p in patterns if not any(_match(n, p) for n in matched_names)]
    if unused:
        raise ValueError(f"These patterns matched nothing: {unused}")
