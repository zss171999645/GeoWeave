import inspect
import math

import torch.optim.lr_scheduler as lr_scheduler
from omegaconf import OmegaConf

from utils.registry import Registry


SCHEDULERS = Registry("schedulers")


def _supports_verbose(scheduler_cls) -> bool:
    return "verbose" in inspect.signature(scheduler_cls.__init__).parameters


@SCHEDULERS.register_module()
class MultiStepLR(lr_scheduler.MultiStepLR):
    def __init__(
        self,
        optimizer,
        milestones,
        total_steps,
        gamma=0.1,
        last_epoch=-1,
        verbose=False,
    ):
        super().__init__(
            optimizer=optimizer,
            milestones=[rate * total_steps for rate in milestones],
            gamma=gamma,
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class MultiStepWithWarmupLR(lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        milestones,
        total_steps,
        gamma=0.1,
        warmup_rate=0.05,
        warmup_scale=1e-6,
        last_epoch=-1,
        verbose=False,
    ):
        milestones = [rate * total_steps for rate in milestones]

        def multi_step_with_warmup(s):
            factor = 1.0
            for i in range(len(milestones)):
                if s < milestones[i]:
                    break
                factor *= gamma

            if s <= warmup_rate * total_steps:
                warmup_coefficient = 1 - (1 - s / warmup_rate / total_steps) * (
                    1 - warmup_scale
                )
            else:
                warmup_coefficient = 1.0
            return warmup_coefficient * factor

        super().__init__(
            optimizer=optimizer,
            lr_lambda=multi_step_with_warmup,
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class PolyLR(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, total_steps, power=0.9, last_epoch=-1, verbose=False):
        super().__init__(
            optimizer=optimizer,
            lr_lambda=lambda s: (1 - s / (total_steps + 1)) ** power,
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class ExpLR(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, total_steps, gamma=0.9, last_epoch=-1, verbose=False):
        super().__init__(
            optimizer=optimizer,
            lr_lambda=lambda s: gamma ** (s / total_steps),
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class CosineAnnealingLR(lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, total_steps, eta_min=0, last_epoch=-1, verbose=False):
        super().__init__(
            optimizer=optimizer,
            T_max=total_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
            verbose=verbose,
        )


def _warmup_cosine_lr(
    base_lr,
    step,
    decay_iter,
    warmup_iters=8000,
    warmup_factor=0.1,
    warmup_method="linear",
    warmup_start_lr=1e-8,
    min_lr=1e-8,
):
    if step < warmup_iters:
        if warmup_method == "constant":
            warmup_scale = warmup_factor
        elif warmup_method == "linear":
            warmup_scale = step / max(warmup_iters, 1)
        else:
            raise ValueError(f"Unsupported warmup_method={warmup_method}")
        return warmup_start_lr + (base_lr - warmup_start_lr) * warmup_scale

    cosine_iter = step - warmup_iters
    cosine_total = max(1, decay_iter - warmup_iters)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * cosine_iter / cosine_total))
    return max(base_lr * cosine_decay, min_lr)


@SCHEDULERS.register_module()
class WarmupCosineLR(lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        decay_iter=None,
        total_steps=None,
        warmup_iters=8000,
        warmup_factor=0.1,
        warmup_method="linear",
        warmup_start_lr=1e-8,
        min_lr=1e-8,
        last_epoch=-1,
        verbose=False,
        **_ignored_kwargs,
    ):
        if decay_iter is None or int(decay_iter) <= 0:
            decay_iter = total_steps
        if decay_iter is None or int(decay_iter) <= 0:
            raise ValueError("WarmupCosineLR requires positive decay_iter or total_steps")
        self.decay_iter = int(decay_iter)
        self.warmup_iters = int(warmup_iters)
        self.warmup_factor = float(warmup_factor)
        self.warmup_method = warmup_method
        self.warmup_start_lr = float(warmup_start_lr)
        self.min_lr = float(min_lr)

        kwargs = dict(optimizer=optimizer, last_epoch=last_epoch)
        if _supports_verbose(lr_scheduler._LRScheduler):
            kwargs["verbose"] = verbose
        super().__init__(**kwargs)

    def get_lr(self):
        return [
            _warmup_cosine_lr(
                base_lr=base_lr,
                step=self.last_epoch,
                decay_iter=self.decay_iter,
                warmup_iters=self.warmup_iters,
                warmup_factor=self.warmup_factor,
                warmup_method=self.warmup_method,
                warmup_start_lr=self.warmup_start_lr,
                min_lr=self.min_lr,
            )
            for base_lr in self.base_lrs
        ]


@SCHEDULERS.register_module()
class OneCycleLR(lr_scheduler.OneCycleLR):
    r"""Torch.optim.lr_scheduler.OneCycleLR, Block total_steps."""

    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps=None,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
        last_epoch=-1,
        verbose=False,
    ):
        if len(optimizer.param_groups) > 1:
            if not isinstance(max_lr, list):
                # max_lr = [max_lr * pg.get("lr_scale", 1.0) for pg in optimizer.param_groups]
                max_lr = [min(max_lr, pg.get("lr")) for pg in optimizer.param_groups]
            
            assert len(max_lr) == len(optimizer.param_groups)

        kwargs = dict(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=last_epoch,
        )
        if _supports_verbose(lr_scheduler.OneCycleLR):
            kwargs["verbose"] = verbose
        super().__init__(**kwargs)


def build_scheduler(cfg, optimizer):
    cfg_ = OmegaConf.to_container(cfg, resolve=True)
    cfg_["optimizer"] = optimizer
    return SCHEDULERS.build(cfg=cfg_)
