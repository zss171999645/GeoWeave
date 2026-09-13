import torch
import itertools
from torch import nn
from typing import Iterator, Tuple, Mapping, Dict

from torch.optim import Adam, AdamW, SGD, LBFGS, Optimizer
try:
    from torch.distributed.optim import ZeroRedundancyOptimizer as TorchZeroRedundancyOptimizer
except Exception:
    TorchZeroRedundancyOptimizer = None
try:
    from torch.cuda.amp.grad_scaler import OptState
except Exception:
    OptState = None
from easyvolcap.engine import OPTIMIZERS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import *

OPTIMIZERS.register_module()(Adam)
OPTIMIZERS.register_module()(AdamW)
OPTIMIZERS.register_module()(SGD)
OPTIMIZERS.register_module()(LBFGS)

@OPTIMIZERS.register_module()
class ZeroRedundancyOptimizerWrapper(Optimizer if TorchZeroRedundancyOptimizer is None else TorchZeroRedundancyOptimizer):
    def __init__(self,
                 params,
                 optimizer_class,
                 process_group=None,
                 parameters_as_bucket_view=False,
                 overlap_with_ddp=False,
                 **defaults):
        if TorchZeroRedundancyOptimizer is None:
            raise RuntimeError('ZeroRedundancyOptimizer is not available in this PyTorch build.')
        optimizer_class_map = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
            "LBFGS": torch.optim.LBFGS,
        }
        if isinstance(optimizer_class, str):
            use_class = optimizer_class_map.get(optimizer_class, None)
            if use_class is None:
                use_class = getattr(torch.optim, optimizer_class, None)
        else:
            use_class = optimizer_class
        if use_class is None:
            raise ValueError(f"Unknown optimizer_class: {optimizer_class}")
        super().__init__(params, use_class,
                         process_group=process_group,
                         parameters_as_bucket_view=parameters_as_bucket_view,
                         overlap_with_ddp=overlap_with_ddp,
                         **defaults)
        self._codex_overlap_amp_scale = None
        self._codex_overlap_step_ready = False
        self._codex_overlap_found_inf = None
        self._codex_overlap_grad_transform = None
        self._codex_overlap_scalar_stats = None

    def codex_begin_overlap_step(self, scaler=None):
        if not getattr(self, "_overlap_with_ddp", False):
            return
        self._codex_overlap_step_ready = False
        self._codex_overlap_found_inf = None
        self._codex_overlap_scalar_stats = None
        self._codex_overlap_amp_scale = None
        if scaler is None or not getattr(scaler, "is_enabled", lambda: False)():
            return
        get_scale_async = getattr(scaler, "_get_scale_async", None)
        if callable(get_scale_async):
            self._codex_overlap_amp_scale = get_scale_async()
        else:
            scale = getattr(scaler, "_scale", None)
            if scale is not None:
                self._codex_overlap_amp_scale = scale

    def _codex_mark_overlap_step_ready(self, found_inf=None):
        self._codex_overlap_step_ready = True
        self._codex_overlap_found_inf = found_inf

    def codex_set_overlap_grad_transform(self, fn) -> None:
        self._codex_overlap_grad_transform = fn
        self._codex_overlap_scalar_stats = None

    def codex_apply_overlap_grad_transforms(self) -> None:
        if callable(self._codex_overlap_grad_transform):
            self._codex_overlap_scalar_stats = self._codex_overlap_grad_transform()
        else:
            self._codex_overlap_scalar_stats = None

    def codex_overlap_step_ready(self) -> bool:
        return bool(getattr(self, "_codex_overlap_step_ready", False))

    def codex_pop_overlap_scalar_stats(self):
        stats = self._codex_overlap_scalar_stats
        self._codex_overlap_scalar_stats = None
        return stats

    def codex_prepare_overlap_scaler_update(self, scaler) -> None:
        if scaler is None or not scaler.is_enabled():
            self._codex_overlap_step_ready = False
            return
        if OptState is None:
            raise RuntimeError("GradScaler OptState is unavailable in this PyTorch build")
        state = scaler._per_optimizer_states[id(self)]
        found_inf = self._codex_overlap_found_inf
        if found_inf is None:
            device = self.param_groups[0]["params"][0].device
            found_inf = torch.zeros((), dtype=torch.float32, device=device)
        state["stage"] = OptState.STEPPED
        state["found_inf_per_device"] = {found_inf.device: found_inf}
        self._codex_overlap_step_ready = False
        self._codex_overlap_found_inf = None

    def codex_finish_overlap_step(self) -> None:
        self._codex_overlap_step_ready = False
        self._codex_overlap_found_inf = None
        self._codex_overlap_scalar_stats = None


@OPTIMIZERS.register_module()
class MyFusedAdam(Adam):
    def step(self, closure=None):
        """Perform a single optimization step.

        Will disrespect weight decay, but significantly reduce kernel launches

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for i, group in enumerate(self.param_groups):
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps)

            from easyvolcap.utils.adam_utils import _single_tensor_adam
            _single_tensor_adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=getattr(self, "amsgrad", False),
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=getattr(self, "maximize", False),
                capturable=getattr(self, "capturable", False),
                differentiable=getattr(self, "differentiable", False),
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps
    ):
        # Older version of PyTorch doesn't have this method
        has_complex = False
        for p in group['params']:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    state['step'] = (
                        torch.tensor(0.0, dtype=torch.float32)
                    )
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                state_steps.append(state['step'])
        return has_complex


@OPTIMIZERS.register_module()
def ConfigurableOptimizer(named_params: Iterator[Tuple[str, nn.Parameter]],

                          # Default parameters
                          lr: float = 5e-3,
                          eps: float = 1e-15,
                          weight_decay: float = 0.0,

                          # Special parameters
                          lr_table: dotdict = dotdict(),  # empty special learning rate table
                          eps_table: dotdict = dotdict(),  # empty table
                          weight_decay_table: dotdict = dotdict(),  # empty table

                          fused: bool = None,
                          foreach: bool = None,

                          optimizer_cfg: dotdict = dotdict(type=Adam.__name__),
                          ) -> Optimizer:
    if isinstance(named_params, Iterator):
        first = next(named_params)
        if isinstance(first, Tuple):
            named_params = itertools.chain([first], named_params)
        elif isinstance(first, nn.Parameter):
            log(yellow(f'Passed in a list of parameters, assuming they are named sequentially.'))
            named_params = {str(i): first for i, first in enumerate(named_params)}.items()
        else:
            raise NotImplementedError
    elif isinstance(named_params, Dict):
        named_params = named_params.items()
    else:
        raise NotImplementedError

    lr_line = dotdict()
    lr_line.lr = lr
    lr_line.eps = eps
    lr_line.weight_decay = weight_decay
    if lr_line: log('Starting learning rate config:', line(lr_line))

    lr_line = dotdict()
    if len(lr_table): lr_line.lr = lr_table
    if len(eps_table): lr_line.eps = eps_table
    if len(weight_decay_table): lr_line.weight_decay = weight_decay_table
    if lr_line: log('Special learning rate config:', line(lr_line))

    # This is resulting in a lot of parameter groups, might reach cuda launch queue depth limit for optimization
    # One option is to consider settings the same type of parameters in the same group, but this might not be the intended behavior for the user
    # Another is to only perform optimization step on the parameters that received gradient
    param_groups = []
    for key, value in named_params:
        if not value.requires_grad:
            continue  # skip non-optimizable paramters
        v_lr = lr
        v_eps = eps
        v_weight_decay = weight_decay
        keys = key.split('.')
        for item in keys:
            if item in lr_table:
                v_lr = lr_table[item]
                break
        for item in keys:
            if item in eps_table:
                v_eps = eps_table[item]
                break
        for item in keys:
            if item in weight_decay_table:
                v_weight_decay = weight_decay_table[item]
                break
        param_groups.append(
            dotdict(
                params=[value],
                lr=v_lr,
                eps=v_eps,
                weight_decay=v_weight_decay,
                name=key,
                fused=fused,
                foreach=foreach,
            )
        )

    if not len(param_groups):
        log(red('optimizer got an empty parameter list, assume you\'re testing'))
        return None

    return OPTIMIZERS.build(optimizer_cfg, params=param_groups)


@OPTIMIZERS.register_module()
def OfficialVGGTOoptimizer(named_params: Iterator[Tuple[str, nn.Parameter]],
                           optimizer_cfg: dotdict = dotdict(type=AdamW)) -> Optimizer:
    params = [param for _, param in named_params if param.requires_grad]
    if not params:
        log(red('optimizer got an empty parameter list, assume you\'re testing'))
        return None

    opt_cfg = dotdict(optimizer_cfg)
    opt_type = opt_cfg.pop('type', AdamW)
    if isinstance(opt_type, str):
        opt_cls = OPTIMIZERS.get(opt_type)
        if opt_cls is None:
            opt_cls = getattr(torch.optim, opt_type, None)
    else:
        opt_cls = opt_type

    if opt_cls is None:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    return opt_cls(params, **opt_cfg)
