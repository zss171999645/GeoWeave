# FSDP runner, we use FSDP for distributed training
# Maybe it's better to use FSDP2, but it's not stable yet?

# For type annotation
import time
import torch
import datetime
import functools
import inspect

from datetime import timedelta
import torch.distributed as dist
# https://docs.pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html, FSDP1
# https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html, FSDP2
# from torch.distributed.device_mesh import init_device_mesh, DeviceMesh  # https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html
# from torch.distributed.fsdp import fully_shard, FSDPModule  # FSDP2, maybe find out how to use this
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy  # FSDP1
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, BackwardPrefetch, CPUOffload  # FSDP1

try:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        apply_activation_checkpointing,
        CheckpointImpl,
    )
    _HAS_ACTIVATION_CHECKPOINTING = True
except Exception:
    _HAS_ACTIVATION_CHECKPOINTING = False

from easyvolcap.engine import cfg, args, call_from_cfg
from easyvolcap.runners.schedulers import ExponentialLR
from easyvolcap.runners.recorders import TensorboardRecorder
from easyvolcap.runners.moderators import DatasetRatioModerator
from easyvolcap.runners.optimizers import ConfigurableOptimizer, Adam
from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner
from easyvolcap.dataloaders.volumetric_video_dataloader import VolumetricVideoDataloader
from easyvolcap.runners.evaluators.volumetric_video_evaluator import VolumetricVideoEvaluator
from easyvolcap.runners.visualizers.volumetric_video_visualizer import VolumetricVideoVisualizer
from easyvolcap.engine import RUNNERS, OPTIMIZERS, SCHEDULERS, RECORDERS, VISUALIZERS, EVALUATORS, MODERATORS

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.timer_utils import timer
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.prof_utils import profiler_step
from easyvolcap.utils.data_utils import add_iter, to_cuda, get_dtype
from easyvolcap.utils.dist_utils import get_rank, get_local_size, init_intra_node_process_group, init_inter_node_process_group
from easyvolcap.utils.net_utils import load_model, load_network, load_controller, save_model_fsdp, save_npz_fsdp, find_class_by_name


@RUNNERS.register_module()
class FSDPRunner(VolumetricVideoRunner):  # an accelerator for distributed training
    def __init__(self,
                 val_dataloader: VolumetricVideoDataloader,  # enumerate this
                 optimizer_cfg: dotdict = dotdict(type=ConfigurableOptimizer.__name__),
                 scheduler_cfg: dotdict = dotdict(type=ExponentialLR.__name__),
                 moderator_cfg: dotdict = dotdict(type=DatasetRatioModerator.__name__),

                 recorder_cfg: dotdict = dotdict(type=TensorboardRecorder.__name__),
                 visualizer_cfg: dotdict = dotdict(type=VolumetricVideoVisualizer.__name__),
                 evaluator_cfg: dotdict = dotdict(type=VolumetricVideoEvaluator.__name__),

                 epochs: int = 400,  # total: ep_iter * epoch number of iterations
                 decay_epochs: int = -1,  # if -1, use epochs, else give user more control
                 ep_iter: int = 500,  # number of iterations per epoch
                 resume: bool = True,
                 test_only: bool = False,

                 fsdp_strategy: str = ShardingStrategy.HYBRID_SHARD,  # FSDP1 only, use HYBRID_SHARD for now
                 fsdp_shard_class_list: List[str] = [],  # list of classes to be fully sharded
                 fsdp_cpu_offload: bool = False,  # offload params/gradients to CPU
                 fsdp_ignored_param_names: List[str] = None,  # parameter name suffixes to ignore in FSDP
                 fsdp_ignored_module_names: List[str] = None,  # module name suffixes to ignore in FSDP
                 activation_checkpoint_cfg: dotdict = dotdict(),  # activation checkpoint/offload
                 timeout_minutes: int = 60,  # set it to a large value to avoid NCCL timeout

                 # FSDP2 specific, not used for now
                 fsdp_shard_variable_list: List[str] = [],  # list of modules to be fully sharded
                 mesh_type: int = 2,  # 1D-mesh or 2D-mesh
                 mesh_dim_names: List[str] = ['node', 'gpu'],  # mesh dimension names corresponding to the mesh_type
                 reshard_after_forward: bool = True,  # default reshard after forward pass, saves more memory

                 **kwargs
                 ):
        # Ignore things, since this will serve as a base class of classes supporting *args and **kwargs
        # The inspection of registration and config system only goes down one layer
        # Otherwise it would be to inefficient
        call_from_cfg(super().__init__,
                      kwargs,
                      val_dataloader=val_dataloader,
                      optimizer_cfg=optimizer_cfg,
                      scheduler_cfg=scheduler_cfg,
                      moderator_cfg=moderator_cfg,
                      recorder_cfg=recorder_cfg,
                      visualizer_cfg=visualizer_cfg,
                      evaluator_cfg=evaluator_cfg,
                      epochs=epochs,
                      decay_epochs=decay_epochs,
                      ep_iter=ep_iter,
                      resume=resume,
                      test_only=test_only,
                      )
        # FSDP need to do the test forward in every rank
        if get_rank() > 0:
            self.val_dataloader = val_dataloader  # different dataloader for validation
            self.evaluator: VolumetricVideoEvaluator = EVALUATORS.build(evaluator_cfg)
            self.visualizer: VolumetricVideoVisualizer = VISUALIZERS.build(visualizer_cfg)
            self.recorder: TensorboardRecorder = RECORDERS.build(recorder_cfg, resume=True, record_config=False)
        self.recorder_pool = []  # used for per-frame lazy recording

        # Load the full state dict only in rank 0
        if get_rank() == 0:
            self.load_model()

        # Delete the optimizer, scheduler, moderator and empty cache
        # This is needed for FSDP, since it will create a new model
        if hasattr(self, 'optimizer'): del self.optimizer
        if hasattr(self, 'scheduler'): del self.scheduler
        if hasattr(self, 'moderator'): del self.moderator
        torch.cuda.empty_cache()  # free up memory

        activation_checkpoint_cfg = dotdict(activation_checkpoint_cfg or {})
        if activation_checkpoint_cfg.get("enabled", False):
            if not _HAS_ACTIVATION_CHECKPOINTING:
                log(yellow("Activation checkpointing requested but checkpoint_wrapper is unavailable, skipping."))
            else:
                module_classes = activation_checkpoint_cfg.get("module_classes", [])
                module_classes = {str(name) for name in module_classes}
                if not module_classes:
                    log(yellow("Activation checkpointing enabled but module_classes is empty."))
                else:
                    use_reentrant = activation_checkpoint_cfg.get("use_reentrant", False)
                    offload_requested = activation_checkpoint_cfg.get("offload_to_cpu", False)

                    def _checkpoint_supports_offload():
                        try:
                            import inspect
                            return "offload_to_cpu" in inspect.signature(torch.utils.checkpoint.checkpoint).parameters
                        except Exception:
                            return False

                    offload_supported = _checkpoint_supports_offload()
                    if offload_requested and not offload_supported:
                        log(yellow("Activation checkpointing offload_to_cpu unsupported; falling back to non-offload checkpoint wrapper."))
                        offload_requested = False
                        activation_checkpoint_cfg.offload_to_cpu = False

                    if offload_supported:
                        checkpoint_impl = (
                            CheckpointImpl.REENTRANT
                            if use_reentrant
                            else CheckpointImpl.NO_REENTRANT
                        )
                        checkpoint_wrapper_fn = functools.partial(
                            checkpoint_wrapper,
                            checkpoint_impl=checkpoint_impl,
                            offload_to_cpu=offload_requested,
                        )
                    else:
                        class _CheckpointWrapper(torch.nn.Module):
                            def __init__(self, module):
                                super().__init__()
                                self.module = module

                            def forward(self, *args, **kwargs):
                                return torch.utils.checkpoint.checkpoint(
                                    self.module,
                                    *args,
                                    use_reentrant=use_reentrant,
                                    **kwargs,
                                )

                        def checkpoint_wrapper_fn(module):
                            return _CheckpointWrapper(module)

                    def _check_fn(module):
                        return module.__class__.__name__ in module_classes

                    apply_activation_checkpointing(
                        self.model,
                        checkpoint_wrapper_fn=checkpoint_wrapper_fn,
                        check_fn=_check_fn,
                    )

        # https://github.com/pytorch/pytorch/issues/135352#issuecomment-2335354163
        # Need to manually create the `HybridShardProcessGroupType` if using `ShardingStrategy.HYBRID_SHARD`, the reason is that
        # FSDP wrapper will create two intra and inter process groups with default parameters, which is not what we want (the `timeout` parameter is not set).
        # NOTE: we prefer `ShardingStrategy.HYBRID_SHARD` instead of `ShardingStrategy.SHARD_GRAD_OP` for better performance
        if fsdp_strategy == ShardingStrategy.HYBRID_SHARD:
            num_devices_per_node = get_local_size()
            timeout = timedelta(minutes=timeout_minutes)
            intra_node_subgroup = init_intra_node_process_group(num_devices_per_node, timeout=timeout)
            inter_node_subgroup = init_inter_node_process_group(dist.group.WORLD, num_devices_per_node, timeout=timeout)  # assume the world process group is already initialized
            process_group = (intra_node_subgroup, inter_node_subgroup)
        else:
            process_group = None

        fsdp_ignored_param_names = fsdp_ignored_param_names or []
        if isinstance(fsdp_ignored_param_names, str):
            fsdp_ignored_param_names = [fsdp_ignored_param_names]
        fsdp_ignored_param_names = [name for name in fsdp_ignored_param_names if name]
        ignored_parameters = []
        ignored_param_names = []
        if fsdp_ignored_param_names:
            def _match_ignored(name: str) -> bool:
                for needle in fsdp_ignored_param_names:
                    if name == needle or name.endswith(f".{needle}"):
                        return True
                return False

            for name, param in self.model.named_parameters():
                if _match_ignored(name):
                    ignored_parameters.append(param)
                    ignored_param_names.append(name)

            if get_rank() == 0:
                if ignored_param_names:
                    log(yellow(f"FSDP ignored parameters: {', '.join(ignored_param_names)}"))
                else:
                    log(yellow(f"FSDP ignored params requested but not found: {fsdp_ignored_param_names}"))

        if not ignored_parameters:
            ignored_parameters = None

        fsdp_ignored_module_names = fsdp_ignored_module_names or []
        if isinstance(fsdp_ignored_module_names, str):
            fsdp_ignored_module_names = [fsdp_ignored_module_names]
        fsdp_ignored_module_names = [name for name in fsdp_ignored_module_names if name]
        ignored_modules = []
        ignored_module_names = []
        if fsdp_ignored_module_names:
            def _match_ignored_module(name: str) -> bool:
                for needle in fsdp_ignored_module_names:
                    if name == needle or name.endswith(f".{needle}"):
                        return True
                return False

            for name, module in self.model.named_modules():
                if not name:
                    continue
                if _match_ignored_module(name):
                    ignored_modules.append(module)
                    ignored_module_names.append(name)

            if get_rank() == 0:
                if ignored_module_names:
                    log(yellow(f"FSDP ignored modules: {', '.join(ignored_module_names)}"))
                else:
                    log(yellow(f"FSDP ignored modules requested but not found: {fsdp_ignored_module_names}"))

        if not ignored_modules:
            ignored_modules = None

        # Initialize FSDP and shard the model
        cpu_offload = CPUOffload(offload_params=True) if fsdp_cpu_offload else None
        fsdp_kwargs = dict(
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=find_class_by_name(
                    self.model, fsdp_shard_class_list
                ),
            ),
            sharding_strategy=fsdp_strategy,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,  # friendly to the original parameters
            sync_module_states=True,  # sync the module states across the devices
            process_group=process_group,
            cpu_offload=cpu_offload,
        )
        if ignored_parameters:
            if "ignored_parameters" in inspect.signature(FSDP).parameters:
                fsdp_kwargs["ignored_parameters"] = ignored_parameters
            elif get_rank() == 0:
                log(yellow("FSDP ignored_parameters unsupported by this torch version; ignoring."))
        if ignored_modules:
            if "ignored_modules" in inspect.signature(FSDP).parameters:
                fsdp_kwargs["ignored_modules"] = ignored_modules
            elif get_rank() == 0:
                log(yellow("FSDP ignored_modules unsupported by this torch version; ignoring."))

        self.model = FSDP(self.model, **fsdp_kwargs)

        # Create the optimizer, scheduler, moderator on the sharded model again
        # TODO: maybe there is a better way to do this?
        if not test_only:
            self.optimizer: Adam = OPTIMIZERS.build(optimizer_cfg, named_params=((k, v) for k, v in self.model.named_parameters() if v.requires_grad))  # requires parameters
            self.scheduler: ExponentialLR = SCHEDULERS.build(scheduler_cfg, optimizer=self.optimizer, decay_iter=(epochs if decay_epochs < 0 else decay_epochs) * ep_iter)  # requires parameters
            self.moderator: DatasetRatioModerator = MODERATORS.build(moderator_cfg, runner=self, total_iter=epochs * ep_iter)  # after dataset init

        # Bookkeepings
        self.fsdp_strategy = fsdp_strategy
        self.fsdp_shard_class_list = fsdp_shard_class_list

    def load_controller(self):
        if self.pretrained_model:
            epoch = load_controller(model=self.model,
                                    optimizer=self.optimizer,
                                    scheduler=self.scheduler,
                                    moderator=self.moderator,
                                    model_dir=self.pretrained_model,
                                    ext=self.pretrained_model_ext,
                                    )

        epoch = load_controller(model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                moderator=self.moderator,
                                model_dir=self.trained_model,
                                resume=self.resume,
                                epoch=self.load_epoch,
                                )
        return epoch

    def save_network(self, epoch, latest: bool = True, **kwargs):
        try:
            save_model_fsdp(model=self.model,
                            model_dir=self.trained_model,
                            save_lim=self.save_lim,
                            epoch=epoch,
                            latest=latest,
                            **kwargs,
                            )
        except RuntimeError as e:
            log(red(e))
            torch.cuda.empty_cache()

    def save_npz(self, epoch, latest: bool = True, **kwargs):
        try:
            save_npz_fsdp(model=self.model,
                          model_dir=self.trained_model,
                          epoch=epoch,
                          **kwargs,
                          )
        except Exception as e:
            log(red(f'FSDP npz save failed at epoch {epoch}, latest={latest}: {type(e).__name__}: {e}'))
            stacktrace()
            torch.cuda.empty_cache()

    def save_model(
        self,
        epoch: int,
        latest: bool = True,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        save_moderator: bool = True,
        **kwargs,
    ):
        try:
            save_model_fsdp(model=self.model,
                            optimizer=self.optimizer if save_optimizer else None,
                            scheduler=self.scheduler if save_scheduler else None,
                            moderator=self.moderator if save_moderator else None,
                            model_dir=self.trained_model,
                            save_lim=self.save_lim,
                            epoch=epoch,
                            latest=latest,
                            **kwargs,
                            )
        except Exception as e:
            log(red(f'FSDP checkpoint save failed at epoch {epoch}, latest={latest}: {type(e).__name__}: {e}'))
            stacktrace()
            torch.cuda.empty_cache()

    # Epoch based runner
    def train(self):  # from begin epoch
        # Load the optimizer, scheduler, moderator from the checkpoint
        epoch = self.load_controller()

        # The actual training for this epoch
        train_generator = self.train_generator(epoch, self.ep_iter)  # yield every ep iter

        # train the network
        for epoch in range(epoch, self.epochs):

            # Possible to make this a decorator?
            next(train_generator, None)  # avoid reconstruction of the dataloader

            # Leave some breathing room for other applications
            if (epoch + 1) % self.empty_cache_ep == 0:
                log(green(f'Emptying cuda memory cache'))
                torch.cuda.empty_cache()
                log('Current memory info:', {
                    'mem': torch.cuda.memory_allocated() / 2**20,
                    'max_mem': torch.cuda.max_memory_allocated() / 2**20,
                    'mem_cache': torch.cuda.memory_cached() / 2**20,
                    'mem_reserved': torch.cuda.memory_reserved() / 2**20,
                })

            # Saving stuff to disk
            if (epoch + 1) % self.save_ep == 0:
                try:
                    self.save_model(epoch, latest=False)
                    self.save_npz(epoch, latest=False)
                except Exception as e:
                    log(red('Error in model saving, ignored and continuing'))
                    stacktrace()
                    stop_prog()  # stop it, otherwise multiple lives

            if (epoch + 1) % self.save_latest_ep == 0:
                try:
                    self.save_model(epoch, latest=True)
                    self.save_npz(epoch, latest=True)  # for inference and smaller file size
                except Exception as e:
                    log(red('Error in model saving, ignored and continuing'))
                    stacktrace()
                    stop_prog()  # stop it, otherwise multiple lives

            # Perform validation run if required
            if self.eval_ep and self.eval_ep > 0 and (epoch + 1) % self.eval_ep == 0:
                try:
                    # The barrier matters
                    self.test_epoch(epoch + 1)  # will this provoke a live display?
                    # Only record and visualize in the main process
                    if not get_rank():
                        # Individual statistic recording
                        for record in self.recorder_pool:
                            self.recorder.update_scalar_stats(record.scalar_stats)
                            if self.record_images_to_tb: self.recorder.update_image_stats(record.image_stats)
                            self.recorder.record(self.val_dataloader.dataset.split.name + "_" + "FRAME")
                        # Summarize the statistics
                        self.recorder.update_scalar_stats(self.evaluator.summarize())
                        if self.record_images_to_tb: self.recorder.update_image_stats(self.visualizer.summarize())
                        self.recorder.record(self.val_dataloader.dataset.split.name)
                # Catch all exceptions, so that the training does not stop
                except Exception as e:
                    log(red('Error in validation pass, ignored and continuing'))
                    stacktrace()
                    stop_prog()  # stop it, otherwise multiple lives
                    if not self.ignore_eval_error:
                        raise e

            if (epoch + 1) % self.reset_ep == 0:
                log(green(f'Resetting the training generator'))
                train_generator = self.train_generator(epoch + 1, self.ep_iter)  # yield every ep iter

    # Iteration based runner

    def train_generator(self, begin_epoch: int, yield_every: int = 1):
        # Train for one epoch (iterator style)
        # Actual start of the execution
        epoch = begin_epoch  # set starting epoch
        self.maybe_jit_model()
        self.model.train()  # set the network (model) to training mode (recursive to all modules)
        start_time = time.perf_counter()

        # Overlap backward pass of previous iteration with dataloading of next iteration
        enumerater = enumerate(self.dataloader)
        index = 0
        batch = dotdict()
        data_stream: torch.cuda.Stream = torch.cuda.Stream() if self.parallel_dataloading else torch.cuda.current_stream()

        # Get next data and start copying
        with torch.cuda.stream(data_stream):
            index, flying_batch = next(enumerater, (None, None))
            if flying_batch is None:
                flying_batch = dotdict(meta=dotdict(iter=-1))
            else:
                flying_batch = add_iter(flying_batch, begin_epoch * self.ep_iter + index, self.total_iter)  # is this bad naming
                flying_batch = to_cuda(flying_batch)  # cpu -> cuda, note that DDP will move all cpu tensors to cuda as well
                if hasattr(self.model, 'prepare_params'): self.model.prepare_params(self, flying_batch)  # perform some action on the gradients
                elif hasattr(self.model, 'module') and hasattr(self.model.module, 'prepare_params'): self.model.module.prepare_params(self, flying_batch)  # perform some action on the gradients
        torch.cuda.current_stream().wait_stream(data_stream)  # wait for data transfer to finish for the first iteration

        data_time = 0
        batch_time = 0
        forward_time = 0
        backward_time = 0
        optimization_time = 0
        while flying_batch.meta.iter >= 0:  # control number of iterations explicitly
            iter = flying_batch.meta.iter.item()
            batch = flying_batch

            data_end_time = time.perf_counter()
            data_time += data_end_time - start_time
            timer.record('data transfer')

            # Model forwarding
            with torch.cuda.amp.autocast(enabled=self.train_use_amp, dtype=self.amp_dtype):  # maybe perform AMP
                output: dotdict = self.model(batch)  # random dict storing various forms of output
                if not isinstance(output, dotdict):
                    output = dotdict(output)  # FSDP will change the default output type to a dict
            loss: torch.Tensor = output.loss.mean()  # final optimizable loss variable to backward
            image_stats: dotdict = output.image_stats  # things to report to recorder (all image tensors (float32))
            scalar_stats: dotdict = output.scalar_stats  # things to report to logger and recorder (all scalars)
            forward_end_time = time.perf_counter()
            forward_time += forward_end_time - data_end_time
            timer.record('model forwarding')
            self._maybe_capture_cuda_mem_trace("fwd", iter)

            # Optimizer reset
            if self.retain_last_grad: self.optimizer.zero_grad(set_to_none=True)

            # Backward pass
            self.scaler.scale(loss).backward()  # maybe perform AMP
            backward_end_time = time.perf_counter()
            backward_time += backward_end_time - forward_end_time
            timer.record('model backwarding')
            self._maybe_capture_cuda_mem_trace("bwd", iter)

            # Get next data and start copying
            with torch.cuda.stream(data_stream):
                index, flying_batch = next(enumerater, (None, None))
                if flying_batch is None:
                    flying_batch = dotdict(meta=dotdict(iter=-1))
                else:
                    flying_batch = add_iter(flying_batch, begin_epoch * self.ep_iter + index, self.total_iter)  # is this bad naming
                    flying_batch = to_cuda(flying_batch)  # cpu -> cuda, note that DDP will move all cpu tensors to cuda as well
                    if hasattr(self.model, 'prepare_params'): self.model.prepare_params(self, flying_batch)  # maybe move parameters to cpu
                    elif hasattr(self.model, 'module') and hasattr(self.model.module, 'prepare_params'): self.model.module.prepare_params(self, flying_batch)  # perform some action on the gradients
            timer.record('data preparation')

            # Make sure the data transfer is finished for following calls
            torch.cuda.current_stream().wait_stream(data_stream)  # wait for data transfer to finish, the backward pass should be slow enough # MARK: STREAM SYNC

            # Decorate gradient
            if self.clip_grad_norm > 0: FSDP.clip_grad_norm_(self.model, self.clip_grad_norm)
            if self.clip_grad_value > 0: torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad_value)
            if hasattr(self.model, 'decorate_grads'): self.model.decorate_grads(self, batch)  # perform some action on the gradients
            elif hasattr(self.model, 'module') and hasattr(self.model.module, 'decorate_grads'): self.model.module.decorate_grads(self, batch)  # perform some action on the gradients
            timer.record('decorating gradients')

            # Optimization step
            self.scaler.step(self.optimizer)
            if not self.retain_last_grad: self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            self.moderator.step()
            self.scaler.update()
            optimization_end_time = time.perf_counter()
            optimization_time += optimization_end_time - backward_end_time
            timer.record('optimization step')
            self._maybe_capture_cuda_mem_trace("opt", iter)

            # Final parameter update
            if hasattr(self.model, 'decorate_params'): self.model.decorate_params(self, batch)  # perform some action on the gradients
            elif hasattr(self.model, 'module') and hasattr(self.model.module, 'decorate_params'): self.model.module.decorate_params(self, batch)  # perform some action on the gradients
            if not get_rank() and self.recorder is not None:
                lr_tag = f"{self.dataloader.dataset.split.name}/lr"
                self.recorder.add_scalar(lr_tag, self.optimizer.param_groups[0]['lr'], iter)

            # Make sure the data transfer is finished for following calls
            data_stream.wait_stream(torch.cuda.current_stream())  # wait for update params to finish for the data stream # MARK: STREAM SYNC
            timer.record('decorating parameters')

            if self.debugging_signal_received:
                breakpoint()
                self.debugging_signal_received = False

            # Records data and batch forwarding time
            end_time = time.perf_counter()
            batch_time += end_time - start_time
            start_time = end_time  # note that all logging and profiling time are accumuated into data_time
            if (iter + 1) % self.empty_cache_interval == 0 and (iter + 1) % self.log_interval == 0:
                # Free as much memory as we can
                log(green(f'Emptying device memory cache'))
                torch.cuda.empty_cache()

            if (iter + 1) % self.host_empty_cache_interval == 0 and (iter + 1) % self.log_interval == 0:
                # Free as much memory as we can
                from easyvolcap.utils.host_utils import host_empty_cache
                log(green(f'Emptying host memory cache'))
                host_empty_cache()

            if (iter + 1) % self.log_interval == 0 and not get_rank():

                # For recording onto the tensorboard
                scalar_stats = dotdict(
                    {
                        k: (v.mean().item() if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray) else v)
                        for k, v in scalar_stats.items()
                    }
                )  # MARK: SYNC

                lr = self.optimizer.param_groups[0]['lr']  # TODO: skechy lr query, only lr of the first param will be saved
                max_mem = torch.cuda.max_memory_allocated() / 2**20
                torch.cuda.reset_peak_memory_stats()
                scalar_stats.data = data_time / self.log_interval
                scalar_stats.batch = batch_time / self.log_interval
                scalar_stats.forward = forward_time / self.log_interval
                scalar_stats.backward = backward_time / self.log_interval
                scalar_stats.optimization = optimization_time / self.log_interval
                scalar_stats.lr = lr
                scalar_stats.max_mem = max_mem
                data_time = 0
                batch_time = 0
                forward_time = 0
                backward_time = 0
                optimization_time = 0

                self.recorder.iter = iter
                self.recorder.epoch = epoch
                self.recorder.update_scalar_stats(scalar_stats)
                if self.record_images_to_tb: self.recorder.update_image_stats(image_stats)  # NOTE: recording images is slow

                # For logging onto the console
                eta_seconds = self.recorder.scalar_stats.batch.global_avg * (self.total_iter - self.recorder.iter)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                log_stats = dotdict()
                log_stats.eta = eta_string
                log_stats.update(self.recorder.log_stats)

                # Render table to screen
                display_table(log_stats)  # render dict as a table (live, console, table)

                # Actually uploading information to tensorboard if needed
                if (iter + 1) % self.record_interval == 0:
                    try: self.recorder.record(self.dataloader.dataset.split.name)  # actual writing to the tensorboard logger
                    except Exception as e:
                        log(red('Error in recording, ignored and continuing'))
                        stacktrace()
                        stop_prog()  # stop it, otherwise multiple lives

            # Maybe save model or perform validation
            if yield_every > 0 and (iter + 1) % yield_every == 0:
                yield output
                self.model.train()

            # Actual start of the execution
            if (iter + 1) % self.ep_iter == 0: epoch = epoch + 1

            # Do profiling if appicable
            profiler_step()  # record a step for the profiler, extracted logic
            timer.record('logging & recording')

    def test_generator(self, epoch: int, yield_every: int = 1):
        # validation for one epoch
        self.maybe_jit_model()
        self.model.train(self.test_using_train_mode)  # set the network (model) to training mode (recursive to all modules)
        self.recorder_pool.clear()  # clear the recorder pool, since we are evaluating a new epoch
        for index, batch in enumerate(tqdm(self.val_dataloader, disable=not self.print_test_progress or get_rank())):
            iter = epoch * self.ep_iter - 1  # some indexing trick
            batch = add_iter(batch, iter, self.total_iter)  # is this bad naming
            batch = to_cuda(batch)  # cpu -> cuda, note that DDP will move all cpu tensors to cuda as well
            with torch.inference_mode(self.test_using_inference_mode), torch.no_grad(), torch.cuda.amp.autocast(enabled=self.test_use_amp, dtype=self.amp_dtype, cache_enabled=self.test_amp_cached):
                output: dotdict = self.model(batch)
                if not isinstance(output, dotdict): output = dotdict(output)  # FSDP1 will change the default output type to a dict
                scalar_stats = self.evaluator.evaluate(output, batch, print=not get_rank())
                image_stats = self.visualizer.visualize(output, batch)

            # Record the statistics into the recorder pool
            self.recorder.iter = iter
            self.recorder.epoch = epoch
            self.recorder_pool.append(dotdict(
                prefix=self.val_dataloader.dataset.split.name + "_" + "FRAME",  # per frame records, to make things cleaner
                step=iter,
                scalar_stats=scalar_stats,
                image_stats=image_stats,
            ))

            if yield_every > 0 and (iter + 1) % yield_every == 0:
                # break  # dataloader could be infinite
                yield output
                self.model.train(self.test_using_train_mode)

            profiler_step()  # record a step for the profiler, extracted logic

        # Empty the cache and synchronize the device
        torch.cuda.empty_cache()  # for better memory recording
        torch.cuda.synchronize()  # wait for all strange copies in the non-training pass to finish
