# Distributed runner, we use accelerate for distributed training
# Add gradient accumulation support
# TODO: check whether `self.accelerator.is_main_process` is equivalent to `get_rank() == 0`?

# For type annotation
import time
import torch
import datetime
import accelerate
from accelerate.utils import AutocastKwargs

from easyvolcap.engine import RUNNERS
from easyvolcap.engine import cfg, args, call_from_cfg
from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.timer_utils import timer
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.dist_utils import get_rank
from easyvolcap.utils.prof_utils import profiler_step
from easyvolcap.utils.data_utils import add_iter, to_cuda, average_dict


@RUNNERS.register_module()
class DistributedRunner(VolumetricVideoRunner):  # an accelerator for distributed training
    def __init__(self,
                 accelerator: accelerate.Accelerator,

                 # Super configurations
                 log_interval: int = 10,
                 test_only: bool = False,
                 **kwargs
                 ):
        # Ignore things, since this will serve as a base class of classes supporting *args and **kwargs
        # The inspection of registration and config system only goes down one layer
        # Otherwise it would be to inefficient
        call_from_cfg(super().__init__, kwargs, log_interval=log_interval, test_only=test_only)

        # Accelerator
        self.accelerator = accelerator

        # Use accelerator to set up distributed training
        if not test_only:
            # FIXME: whether to let accelerate handle the scheduler?
            # accelerate will multiple the learning rate by the number of GPUs in the scheduler...
            self.dataloader, self.model, self.optimizer = accelerator.prepare(
                self.dataloader,
                self.model,
                self.optimizer
            )

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
                flying_batch = to_cuda(flying_batch, device=self.accelerator.device)  # cpu -> cuda, note that DDP will move all cpu tensors to cuda as well
                if hasattr(self.model, 'prepare_params'): self.model.prepare_params(self, flying_batch)  # perform some action on the gradients
                elif hasattr(self.model, 'module') and hasattr(self.model.module, 'prepare_params'): self.model.module.prepare_params(self, flying_batch)  # perform some action on the gradients
        torch.cuda.current_stream().wait_stream(data_stream)  # wait for data transfer to finish for the first iteration

        data_time = 0
        batch_time = 0
        while flying_batch.meta.iter >= 0:  # control number of iterations explicitly
            iter = flying_batch.meta.iter.item()
            batch = flying_batch

            data_time += time.perf_counter() - start_time
            timer.record('data transfer')

            # Support gradient accumulation
            with self.accelerator.accumulate(self.model):

                # Model forwarding
                # TODO: determine accelerate or torch autocast?
                # with torch.cuda.amp.autocast(enabled=self.train_use_amp, dtype=self.amp_dtype):
                with self.accelerator.autocast(AutocastKwargs(enabled=self.train_use_amp)):
                    output: dotdict = self.model(batch)  # random dict storing various forms of output
                loss: torch.Tensor = output.loss.mean()  # final optimizable loss variable to backward
                image_stats: dotdict = output.image_stats  # things to report to recorder (all image tensors (float32))
                scalar_stats: dotdict = output.scalar_stats  # things to report to logger and recorder (all scalars)
                timer.record('model forwarding')
                self._maybe_capture_cuda_mem_trace("fwd", iter)

                # Optimizer reset
                if self.retain_last_grad: self.optimizer.zero_grad(set_to_none=True)

                # Backward pass
                self.accelerator.backward(loss)  # use the accelerator to perform the backward pass
                timer.record('model backwarding')
                self._maybe_capture_cuda_mem_trace("bwd", iter)

                # Get next data and start copying
                with torch.cuda.stream(data_stream):
                    index, flying_batch = next(enumerater, (None, None))
                    if flying_batch is None:
                        flying_batch = dotdict(meta=dotdict(iter=-1))
                    else:
                        flying_batch = add_iter(flying_batch, begin_epoch * self.ep_iter + index, self.total_iter)  # is this bad naming
                        flying_batch = to_cuda(flying_batch, device=self.accelerator.device)  # cpu -> cuda, note that DDP will move all cpu tensors to cuda as well
                        if hasattr(self.model, 'prepare_params'): self.model.prepare_params(self, flying_batch)  # maybe move parameters to cpu
                        elif hasattr(self.model, 'module') and hasattr(self.model.module, 'prepare_params'): self.model.module.prepare_params(self, flying_batch)  # perform some action on the gradients
                timer.record('data preparation')

                # Make sure the data transfer is finished for following calls
                torch.cuda.current_stream().wait_stream(data_stream)  # wait for data transfer to finish, the backward pass should be slow enough # MARK: STREAM SYNC

                # Decorate gradient
                if self.clip_grad_norm > 0: grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                if self.clip_grad_value > 0: self.accelerator.clip_grad_value_(self.model.parameters(), self.clip_grad_value)
                if hasattr(self.model, 'decorate_grads'): self.model.decorate_grads(self, batch)  # perform some action on the gradients
                elif hasattr(self.model, 'module') and hasattr(self.model.module, 'decorate_grads'): self.model.module.decorate_grads(self, batch)  # perform some action on the gradients
                scalar_stats.grad_norm = grad_norm.item()  # record the gradient statistics
                timer.record('decorating gradients')

                # Optimization step
                self.optimizer.step()  # no need to use the scaler step, since the accelerator will handle it
                if not self.retain_last_grad: self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.moderator.step()
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

            if (iter + 1) % self.log_interval == 0:

                # For recording onto the tensorboard
                scalar_stats = [dotdict(
                    {
                        k: (v.mean().item() if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray) else v)
                        for k, v in scalar_stats.items()
                    }
                )]  # MARK: SYNC

                # Gather all the scalars from all processes
                scalar_stats = self.accelerator.gather_for_metrics(scalar_stats, use_gather_object=True)  # MARK: SYNC

                # Only the main process will do the logging
                if not get_rank():  # TODO: test `self.accelerator.is_main_process`
                    # Average the scalars
                    scalar_stats = dotdict(average_dict(scalar_stats))

                    lr = self.optimizer.param_groups[0]['lr']  # TODO: skechy lr query, only lr of the first param will be saved
                    max_mem = torch.cuda.max_memory_allocated() / 2**20
                    torch.cuda.reset_peak_memory_stats()
                    scalar_stats.data = data_time / self.log_interval
                    scalar_stats.batch = batch_time / self.log_interval
                    scalar_stats.lr = lr
                    scalar_stats.max_mem = max_mem
                    data_time = 0
                    batch_time = 0

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
