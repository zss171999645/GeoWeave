# fmt: off
from easyvolcap.utils.console_utils import *
import torch  # make early keyboard interrupt possible
import torch.distributed as dist

from typing import Callable
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP

import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs

from easyvolcap.engine import args, cfg  # commandline entrypoint
from easyvolcap.engine import RUNNERS, MODELS, DATALOADERS
from easyvolcap.engine import callable_from_cfg, call_from_cfg

from easyvolcap.utils.import_utils import discover_modules
discover_modules() # will launch through this interface

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from easyvolcap.dataloaders.datasets.volumetric_video_dataset import VolumetricVideoDataset, WillChangeToNoopIfGUIDataset
    from easyvolcap.dataloaders.datasamplers import SequentialSampler, BatchSampler
    from easyvolcap.dataloaders.volumetric_video_dataloader import VolumetricVideoDataloader
    from easyvolcap.runners.fsdp_runner import FSDPRunner
    from easyvolcap.runners.distributed_runner import DistributedRunner
    from easyvolcap.models.depth_model import DepthModel
    from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner

from easyvolcap.utils.data_utils import DataSplit
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.prof_utils import setup_profiler, profiler_start, profiler_stop
from easyvolcap.utils.net_utils import setup_deterministic, number_of_params, find_class_by_name
from easyvolcap.utils.dist_utils import synchronize, get_rank, get_distributed, get_world_size, get_local_size
from easyvolcap.utils.ddp_zero_overlap import register_zero_overlap_comm_hook
# fmt: on


def launcher(runner_function: Callable,  # viewer.run or runner.train or runner.test
             runner_object: "VolumetricVideoRunner" = None,
             exp_name='nerft',
             detect_anomaly: bool = False,
             profiler_cfg: dotdict = dotdict(),  # for debug use only

             *args,
             **kwargs
             ):

    # Perform the actual training
    setup_profiler(**profiler_cfg)  # MARK: overwritting default configs
    prev_anomaly = torch.is_anomaly_enabled()
    torch.set_anomaly_enabled(detect_anomaly)
    profiler_start()  # already setup

    # Give the user some time to save states
    log('Launching runner for experiment:', magenta(exp_name))
    cfg.runner = runner_object  # holds a global reference for hacky usage # MARK: GLOBAL
    runner_function()

    profiler_stop()  # already setup
    torch.set_anomaly_enabled(prev_anomaly)


def preflight(
    fix_random: bool = False,
    allow_tf32: bool = True,
    deterministic: bool = False,  # for debug use only
    benchmark: Union[bool, str] = True,  # for static sized input
    ignore_breakpoint: bool = False,
    hide_progress: bool = False,
    hide_output: bool = False,
    verbose: bool = False,
    **kwargs,
):
    # Some early on GUI specific configurations
    if ignore_breakpoint: disable_breakpoint()
    if hide_progress: disable_progress()
    if hide_output: disable_console()
    if verbose: enable_verbose_log()
    if benchmark == 'train': benchmark = args.type == 'train'  # for static sized input

    # Maybe make this run deterministic?
    setup_deterministic(fix_random, allow_tf32, deterministic, benchmark)  # whether to be deterministic throughout the whole training process?

    # Log the experiment name for later usage
    log(f"Starting experiment: {magenta(cfg.exp_name)}, command: {magenta(args.type)}")  # MARK: GLOBAL


@callable_from_cfg
def test(
    model_cfg: dotdict = dotdict(type="DepthModel"),
    val_dataloader_cfg: dotdict = dotdict(
        type="VolumetricVideoDataloader",
        max_iter=-1,
        sampler_cfg=dotdict(
            type="SequentialSampler",  # changed type
        ),
        dataset_cfg=dotdict(
            type="VolumetricVideoDataset",  # TODO: not overwritting, repeated
            split=DataSplit.VAL.name,
            ratio=0.25,  # faster visualization
        ),
    ),
    runner_cfg: dotdict = dotdict(type="VolumetricVideoRunner",
                                  optimizer_cfg=dotdict(type=None),
                                  scheduler_cfg=dotdict(type=None),
                                  ),

    # Reproducibility configuration
    base_device: str = 'cuda',

    record_images_to_tb: bool = False,  # MARK: insider config # this is slow
    print_test_progress: bool = True,  # MARK: insider config # this is slow
    dry_run: bool = False,
    distributed: bool = False,
    timeout_minutes: int = 60,
    **kwargs,
):
    if runner_cfg.get('test_with_ddp_sharding', False):
        assert distributed, "You must enable distributed=True when using test_with_ddp_sharding"
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'{base_device}:{local_rank}')
    if base_device == 'cuda':
        torch.cuda.set_device(device)
    if distributed:
        init_pg_kwargs = dict(backend="nccl", init_method="env://", timeout=timedelta(minutes=timeout_minutes))
        if base_device == 'cuda':
            init_pg_kwargs["device_id"] = device
        try:
            dist.init_process_group(**init_pg_kwargs)
        except TypeError:
            init_pg_kwargs.pop("device_id", None)
            dist.init_process_group(**init_pg_kwargs)
        synchronize()
    # Maybe make this run deterministic?
    preflight(**kwargs)  # whether to be deterministic throughout the whole training process?

    # Construct other parts of the training process
    val_dataloader: "VolumetricVideoDataloader" = DATALOADERS.build(val_dataloader_cfg)  # reuse the validataion

    # Handle model placement
    rank = get_rank()

    model: "DepthModel" = MODELS.build(model_cfg)
    model = model.to(base_device, non_blocking=True)
    model.eval()  # set to evaluation mode
    if base_device == 'cpu':
        with torch.no_grad():
            for param in model.parameters():
                param.set_(param.contiguous())  # pin memory for cpu
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    runner: "VolumetricVideoRunner" = RUNNERS.build(runner_cfg,
                                                    model=model,
                                                    dataloader=None,  # no training dataloader
                                                    test_only=True,  # no training
                                                    record_images_to_tb=record_images_to_tb,  # another default
                                                    print_test_progress=print_test_progress,  # another default
                                                    val_dataloader=val_dataloader)
    if hasattr(model, 'move_device'): model.move_device(device)

    if dry_run: return runner  # just construct everything, then return

    # Just run, no gossip
    launcher(**kwargs, runner_function=runner.test, runner_object=runner)


@callable_from_cfg
def train(
    model_cfg: dotdict = dotdict(type="DepthModel"),
    dataloader_cfg: dotdict = dotdict(type="VolumetricVideoDataloader"),
    val_dataloader_cfg: dotdict = dotdict(
        type="VolumetricVideoDataloader",
        max_iter=-1,  # awkward
        # num_workers=0,  # if multiprocessing, will mess up normal dataloader
        sampler_cfg=dotdict(
            type="SequentialSampler",  # changed type
        ),
        dataset_cfg=dotdict(
            type="VolumetricVideoDataset",  # TODO: not overwritting, repeated
            split=DataSplit.VAL.name,
            ratio=0.25,  # faster visualization
        ),
        batch_sampler_cfg=dotdict(type="BatchSampler",
                                  batch_size=1),
    ),
    runner_cfg: dotdict = dotdict(type="VolumetricVideoRunner"),

    # Distributed training
    distributed: bool = False,
    timeout_minutes: int = 60,
    bucket_cap_mb: int = None,  # 0 means no bucket
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = False,  # setting to True will save memory, but may conflict with inplace operations
    static_graph: bool = False,  # let DDP skip graph discovery when the autograd graph is stable
    broadcast_buffers: bool = True,  # whether to broadcast buffers in DDP
    fsdp_enabled: bool = False,  # use fsdp instead of ddp

    # Accelerated distributed training
    accelerated: bool = False,
    # Set the default parameters in the accelerate config for better compatibility
    # e.g., mixed_precision, gradient_accumulation_steps, ...
    # The reason not to set default here is that there will be conflicts between deepspeed config and accelerate config if you are using deepspeed
    accelerator_cfg: dotdict = dotdict(),  # TODO: maybe a better way to handle this?

    # Printing configuration
    dry_run: bool = False,  # only print network and exit
    print_model: bool = False,  # since the network is pretty complex, give the option to print
    print_parameters: bool = True,

    # Reproducibility configuration
    base_device: str = 'cuda',
    **kwargs,
):
    # Maybe make this run deterministic?
    preflight(**kwargs)  # whether to be deterministic throughout the whole training process?

    # Distributed training using accelerate
    if accelerated:
        # Create the accelerator
        # NOTE: the parameters are passed directly by commandline arguments
        # https://huggingface.co/docs/accelerate/v0.23.0/basic_tutorials/launch
        accelerator = Accelerator(**accelerator_cfg)
        # Since we are using a custom batch sampler, and QUOTED from the official documentation:
        # batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last.
        # `self.batch_size` will be set to None during self.__init__() in this case.
        # But this will cause the following error when start distributed training using accelerate:
        # FIXME: You need to use even_batches=False when the batch sampler has no batch size. If you are not calling this method directly, set accelerator.even_batches=False instead.
        # We have to set self.batch_size to the actual batch size here to avoid this error.
        # TODO: Need to check if this is any ill effect caused by this.
        accelerator.dataloader_config.even_batches = False

        # Handle model placement, more general
        device = accelerator.device

    # Distributed training using native torch
    else:
        # Handle model placement
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'{base_device}:{local_rank}')
        if base_device == 'cuda':
            torch.cuda.set_device(device)
            
        # Handle distributed training
        if distributed:
            init_pg_kwargs = dict(backend="nccl", init_method="env://", timeout=timedelta(minutes=timeout_minutes))
            if base_device == 'cuda':
                init_pg_kwargs["device_id"] = device
            try:
                dist.init_process_group(**init_pg_kwargs)
            except TypeError:
                init_pg_kwargs.pop("device_id", None)
                dist.init_process_group(**init_pg_kwargs)
            synchronize()

    # Construct the dataloader and dataset after the distributed process
    dataloader: "VolumetricVideoDataloader" = DATALOADERS.build(dataloader_cfg)
    use_main_val_suite = bool(runner_cfg.get('main_val_cfgs') or runner_cfg.get('main_val_core4_cfg')) and args.type == 'train'
    if use_main_val_suite:
        val_dataloader = None
    else:
        val_dataloader: "VolumetricVideoDataloader" = DATALOADERS.build(val_dataloader_cfg) if not get_rank() or fsdp_enabled or runner_cfg.get('test_with_ddp_sharding', False) else None

    # Model building and distributed training related stuff
    model: "DepthModel" = MODELS.build(model_cfg)  # some components are directly moved to cuda when building
    if not accelerated:
        model.to(device, non_blocking=True)  # move this model to this specific device
    model.train()  # set to training mode, this is the default mode

    if device.type == 'cpu':
        # with torch.no_grad():
        for name, param in model.named_parameters():
            # param.data = param.data.pin_memory()  # pin memory for cpu
            param.data = param.data.to('cuda', non_blocking=True).to('cpu', non_blocking=True)  # HACK: pin memory for cpu
        # from easyvolcap.utils.host_utils import host_empty_cache
        # torch.cuda.empty_cache()
        # host_empty_cache()
        # torch.cuda.reset_peak_memory_stats()

    # else:
        # model.to(device, non_blocking=True)  # move this model to this specific device

    # Distributed training using accelerate
    if accelerated:
        runner_cfg.type = "DistributedRunner"
        # Construct the runner (optimization loop controller)
        runner: "VolumetricVideoRunner" = RUNNERS.build(runner_cfg,
                                                        model=model,
                                                        dataloader=dataloader,
                                                        val_dataloader=val_dataloader,
                                                        accelerator=accelerator)  # pass the accelerator
    # Distributed training using FSDP
    elif fsdp_enabled:
        runner_cfg.type = "FSDPRunner"
        # Construct the runner (optimization loop controller)
        runner: "VolumetricVideoRunner" = RUNNERS.build(runner_cfg,
                                                        model=model,
                                                        dataloader=dataloader,
                                                        val_dataloader=val_dataloader)
    # Distributed training using native torch
    else:
        # Handle distributed model
        if get_distributed():
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                bucket_cap_mb=bucket_cap_mb,
                find_unused_parameters=find_unused_parameters,
                gradient_as_bucket_view=gradient_as_bucket_view,
                static_graph=static_graph,
                broadcast_buffers=broadcast_buffers,
            )
        # Construct the runner (optimization loop controller)
        runner: "VolumetricVideoRunner" = RUNNERS.build(runner_cfg,
                                                        model=model,
                                                        dataloader=dataloader,
                                                        val_dataloader=val_dataloader)
        if get_distributed() and isinstance(model, DDP):
            optimizer = getattr(runner, "optimizer", None)
            overlap_with_ddp = bool(getattr(optimizer, "_overlap_with_ddp", False)) if optimizer is not None else False
            if overlap_with_ddp:
                register_zero_overlap_comm_hook(
                    model,
                    optimizer,
                    use_grad_scaler_aware=getattr(runner.scaler, "is_enabled", lambda: False)(),
                    shard_buckets=False,
                )
                log(yellow("Registered ZeRO-overlap DDP comm hook (default-off experimental path)."))

    if hasattr(model, 'move_device'): model.move_device(device)

    if print_model and not get_rank():  # only print once
        # For some methods, both the network and the sampler or even the renderer contains optimizable parameters
        # But the sampler and render both has a reference to the network, which gets printed (not saved, tested)
        pprint(model)  # with indent guides
    if print_parameters:
        try:
            nop = number_of_params(model)
            if nop > 1e9:
                log(f'Number of optimizable parameters: {nop} ({nop / 1e9:.2f} B)')
            else:
                log(f'Number of optimizable parameters: {nop} ({nop / 1e6:.2f} M)')
        except ValueError as e:
            # Ignore: Attempted to use an uninitialized parameter in <method 'numel' of 'torch._C._TensorBase' objects>
            pass

    if dry_run: return runner  # just construct everything, then return

    # The actual calling, with grace full exit
    launcher(**kwargs, runner_function=runner.train, runner_object=runner)


@catch_throw
def main():
    if cfg.mocking: log(f'{green("Modules imported.")} Mode: {yellow(args.type)}. No config loaded, pass config file using `-c <PATH_TO_CONFIG>`')  # MARK: GLOBAL
    else: globals()[args.type](cfg)  # invoke this (call callable_from_cfg -> call_from_cfg)


# Module name == '__main__', this is the outermost commandline entry point
if __name__ == '__main__':
    main()
