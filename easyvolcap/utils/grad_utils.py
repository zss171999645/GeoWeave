# Adapted from https://github.com/facebookresearch/vggt/blob/training/training/train_utils/gradient_clip.py
# This is a fine-grained gradient clipping utility that allows users to clip gradients for different parts of the model

import torch
import torch.nn as nn
from typing import Optional

from easyvolcap.engine import OPTIMIZERS


class _BaseGradientClipper:
    key_prefix = ""

    def __init__(self, configs, *args, **kwargs):
        self.configs = self._normalize_configs(configs)
        self.params_to_clip_by_config = None
        self.is_initialized = False

    @staticmethod
    def _normalize_configs(configs):
        normalized = []
        for config in configs:
            module_names = config["module_name"]
            if isinstance(module_names, str):
                module_names = [module_names]
            normalized.append({
                "module_names": module_names,
                "max_norm": float(config["max_norm"]) if config["max_norm"] is not None else None,
                "norm_type": config.get("norm_type", 2),
            })
        return normalized

    @staticmethod
    def _collect_params_by_config(model: nn.Module, configs):
        named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        params_to_clip_by_config = []
        all_clipped_params = set()

        for config in configs:
            current_params = []
            for name, param in named_params:
                if any(module_name in name for module_name in config["module_names"]):
                    current_params.append(param)
                    all_clipped_params.add(param)
            params_to_clip_by_config.append((config, current_params))

        remaining_params = [param for _, param in named_params if param not in all_clipped_params]
        return params_to_clip_by_config, remaining_params

    def setup_clipping(self, model: nn.Module) -> None:
        params_to_clip_by_config, remaining_params = self._collect_params_by_config(model, self.configs)

        if remaining_params:
            print(f"Found {len(remaining_params)} parameters that won't be clipped")
            print(remaining_params)
            raise ValueError("Some parameters are not configured for gradient clipping")

        self.params_to_clip_by_config = params_to_clip_by_config
        self.is_initialized = True

    def _format_key(self, module_names):
        return f"{self.key_prefix}{','.join(module_names)}"

    def __call__(self, model: nn.Module) -> Optional[torch.Tensor]:
        if not self.is_initialized:
            raise RuntimeError("GradientClipper must be initialized with setup_clipping() before use")

        grad_norms = {}
        for config, params_to_clip in self.params_to_clip_by_config:
            if not params_to_clip or config["max_norm"] is None:
                continue

            grad_norm = nn.utils.clip_grad_norm_(
                params_to_clip,
                max_norm=config["max_norm"],
                norm_type=config["norm_type"],
            )

            if grad_norm is None:
                continue

            grad_norms[self._format_key(config["module_names"])] = grad_norm.item()
        return grad_norms


@OPTIMIZERS.register_module()
class GradientClipper(_BaseGradientClipper):
    """
    Gradient clipping utils that works for both FSDP and DDP with support for different
    clipping configurations for different parts of the model.
    """
    key_prefix = "grad_norm_"


@OPTIMIZERS.register_module()
class OfficialGradientClipper(_BaseGradientClipper):
    """
    Gradient clipping utils aligned with official VGGT training.
    """


@OPTIMIZERS.register_module()
class NoopClipper:
    def __init__(self, configs, *args, **kwargs):
        pass

    def setup_clipping(self, model: nn.Module) -> None:
        pass

    def __call__(self, model: nn.Module) -> Optional[torch.Tensor]:
        return {}
