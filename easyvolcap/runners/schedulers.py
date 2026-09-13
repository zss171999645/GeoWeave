import math
import numpy as np
from typing import List, Callable
from torch.optim.optimizer import Optimizer
from easyvolcap.engine import SCHEDULERS
from easyvolcap.utils.base_utils import dotdict
from torch.optim.lr_scheduler import _LRScheduler, StepLR, LambdaLR, SequentialLR, OneCycleLR
from easyvolcap.official_vggt.train_utils.param_scheduler import build_param_scheduler

try:
    import sympy
except ModuleNotFoundError:
    sympy = None

SCHEDULERS.register_module()(StepLR)
SCHEDULERS.register_module()(LambdaLR)
SCHEDULERS.register_module()(OneCycleLR)

@SCHEDULERS.register_module()
class MultiLR(_LRScheduler):
    def __init__(self, optimizer, decay_iter, scheduler_cfgs, last_epoch=-1, verbose=False):
        raise ValueError('MultiLR has bugs! Please use other schedulers instead.')
        self.schedulers = dotdict()
        self.names = [param_group['name'] for param_group in optimizer.param_groups]
        # values = self._get_optimizer_lr(optimizer)
        for name, scheduler_cfg in scheduler_cfgs.items():
            scheduler = SCHEDULERS.build(scheduler_cfg, optimizer=optimizer, decay_iter=decay_iter)
            self.schedulers[name] = scheduler
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        result = []
        for name, sched in self.schedulers.items():
            idx = self.names.index(name)
            result.append(sched.get_last_lr()[idx])
        return result

    @staticmethod
    def _set_optimizer_lr(optimizer, values):
        for param_group in optimizer.param_groups:
            param_group['lr'] = values[param_group['name']]

    @staticmethod
    def _get_optimizer_lr(optimizer):
        values = dotdict()
        for param_group in optimizer.param_groups:
            values[param_group['name']] = param_group['lr']
        return values

    def step(self, epoch=None):
        if self.last_epoch != -1:
            values = self._get_optimizer_lr(self.optimizer)
            for name, sched in self.schedulers.items():
                sched.step()
                values[name] = self._get_optimizer_lr(self.optimizer)[name]
                self._set_optimizer_lr(self.optimizer, values)
        super().step()


@SCHEDULERS.register_module()
class NoopLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, **kwargs):
        super().__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        pass


@SCHEDULERS.register_module()
class ExponentialLR(_LRScheduler):
    def __init__(self,
                 optimizer,  # object
                 decay_iter,  # no default for this config
                 gamma=0.1,
                 min_lr=5e-5,
                 last_epoch=-1):
        self.decay_iter = decay_iter
        self.gamma = gamma
        self.min_lr = min_lr
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = [base_lr * self.gamma ** (self.last_epoch / self.decay_iter) for base_lr in self.base_lrs]
        lrs = [max(lr, self.min_lr) if base_lr > self.min_lr else lr for base_lr, lr in zip(self.base_lrs, lrs)]
        return lrs


@SCHEDULERS.register_module()
class WarmupExponentialLR(_LRScheduler):
    def __init__(self,
                 optimizer,  # object
                 decay_iter,
                 warmup_factor=1.0 / 3,
                 warmup_epochs=1,
                 warmup_method="linear",
                 gamma=0.1,
                 min_lr=1e-6,
                 last_epoch=-1):
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        self.decay_iter = decay_iter
        self.gamma = gamma
        self.min_lr = min_lr
        super(WarmupExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        lrs = [base_lr * warmup_factor * self.gamma ** (self.last_epoch / self.decay_iter) for base_lr in self.base_lrs]
        lrs = [max(lr, self.min_lr) for lr in lrs]
        return lrs


@SCHEDULERS.register_module()
class WarmupCosineLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 decay_iter,
                 warmup_iters=8000,
                 warmup_factor=0.1,
                 warmup_method="linear",
                 warmup_start_lr=1e-8,
                 min_lr=1e-8,
                 last_epoch=-1,
                 ):
        self.decay_iter = decay_iter
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self.warmup_method = warmup_method
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # `self.last_epoch` is the current iteration since we call `step()`
        # in the training loop every iteration
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                # alpha = self.last_epoch / self.warmup_iters
                # warmup_factor = self.warmup_factor * (1 - alpha) + alpha
                warmup_factor = self.last_epoch / max(self.warmup_iters, 1)
            lrs = [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * warmup_factor
                    for base_lr in self.base_lrs
            ]
        else:
            cosine_iter = self.last_epoch - self.warmup_iters
            cosine_total = max(1, self.decay_iter - self.warmup_iters)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * cosine_iter / cosine_total))
            lrs = [
                max(base_lr * cosine_decay, self.min_lr)
                    for base_lr in self.base_lrs
            ]

        return lrs


@SCHEDULERS.register_module()
class OfficialVGGTScheduler:
    def __init__(self, optimizer, decay_iter, option_schedulers=None, last_epoch=-1):
        if optimizer is None:
            raise ValueError("optimizer is required for OfficialVGGTScheduler")
        self.optimizer = optimizer
        self.decay_iter = max(int(decay_iter), 1)
        self.last_epoch = int(last_epoch)
        self.option_schedulers = self._build_option_schedulers(option_schedulers or {})
        self._apply(0.0)

    def _build_option_schedulers(self, option_schedulers):
        built = {}
        for option, cfg_list in option_schedulers.items():
            if not cfg_list:
                continue
            scheduler_cfg = cfg_list[0].get("scheduler", cfg_list[0])
            built[option] = build_param_scheduler(scheduler_cfg)
        return built

    def _where(self):
        if self.last_epoch < 0:
            return 0.0
        return min(max(self.last_epoch / float(self.decay_iter), 0.0), 1.0)

    def _apply(self, where: float):
        for option, scheduler in self.option_schedulers.items():
            value = scheduler(where)
            for group in self.optimizer.param_groups:
                group[option] = value

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = int(epoch)
        self._apply(self._where())

    def state_dict(self):
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state_dict):
        self.last_epoch = int(state_dict.get("last_epoch", -1))


@SCHEDULERS.register_module()
class NeuSScheduler(_LRScheduler):
    def __init__(self,
                 optimizer,
                 decay_iter,  # no default for this config
                 warm_up_end=500,
                 learning_rate_alpha=0.05,
                 last_epoch=-1):
        self.warm_up_end = warm_up_end
        self.learning_rate_alpha = learning_rate_alpha
        self.decay_iter = decay_iter
        super(NeuSScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        learning_factor = 1.0
        if self.last_epoch < self.warm_up_end:
            learning_factor = self.last_epoch / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.last_epoch - self.warm_up_end) / (self.decay_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        lrs = [base_lr * learning_factor for base_lr in self.base_lrs]
        return lrs


@SCHEDULERS.register_module()
class MultiStepWarmupScheduler(_LRScheduler):
    def __init__(self,
                 optimizer,
                 warm_up_end=5000,
                 milestones=[300000, 400000],
                 gamma=0.1,
                 last_epoch=-1,
                 decay_iter=None  # for compatibility
                 ):
        self.warm_up_end = warm_up_end
        self.milestones = milestones
        self.gamma = gamma
        super(MultiStepWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        learning_factor = 1.0
        if self.last_epoch < self.warm_up_end:
            learning_factor = self.last_epoch / self.warm_up_end
        else:
            index = np.searchsorted(self.milestones, self.last_epoch, side='left')
            learning_factor = self.gamma ** index
        lrs = [base_lr * learning_factor for base_lr in self.base_lrs]
        return lrs


@SCHEDULERS.register_module()
class ConfigurableLambdaLR(LambdaLR):
    def __init__(self,
                 optimizer,  # object
                 lr_lambda,  # no default for this config
                 last_epoch=-1,
                 verbose=False,
                 decay_iter=None  # for compatibility
                 ):
        # Deal with string lambda
        def parse_str_lambda(s: str) -> Callable[[int], float]:
            if sympy is None:
                raise ImportError(
                    "sympy is required when lr_lambda is a string; "
                    "please install sympy or pass a callable/list of callables."
                )
            return sympy.lambdify(
                sympy.symbols('epoch'), sympy.sympify(s), 'math'
            )

        if isinstance(lr_lambda, str):
            lr_lambda = parse_str_lambda(lr_lambda)
        elif isinstance(lr_lambda, list) and all(isinstance(s, str) for s in lr_lambda):
            lr_lambda = [parse_str_lambda(s) for s in lr_lambda]
        super().__init__(optimizer=optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch, verbose=verbose)


@SCHEDULERS.register_module()
class ConfigurableSequentialLR(SequentialLR):
    def __init__(self,
                 optimizer,  # object
                 scheduler_cfgs: List[dotdict],  # no default for this config
                 milestones: List[int] = None,
                 last_epoch=-1,
                 verbose=False,
                 decay_iter=None  # for compatibility
                 ):
        # Build schedulers
        schedulers = [
            SCHEDULERS.build(scheduler_cfg, optimizer=optimizer)
                for scheduler_cfg in scheduler_cfgs
        ]
        super().__init__(optimizer, schedulers=schedulers, milestones=milestones, last_epoch=last_epoch, verbose=verbose)
