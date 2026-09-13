# Default runner, no fancy business here
# Perform the training loop and log stuff out (tensor board etc.)
# Also responsible for saving the model and the optimizer states
# Sometimes performs validation and also writes things to tensorboard

# For type annotation
from collections import defaultdict, Counter
import copy
import json
from typing import Dict, List, Optional
import os
import shutil
import gc
import time
from pathlib import Path
from easyvolcap.dataloaders.datasamplers import DistributedSequentialSampler
from easyvolcap.utils.cam_utils import decode_camera_params
import torch
import signal
import datetime
import platform
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from easyvolcap.engine import cfg, args, Config  # need this for initialization?
from easyvolcap.runners.schedulers import ExponentialLR
from easyvolcap.runners.recorders import TensorboardRecorder
from easyvolcap.runners.moderators import DatasetRatioModerator
from easyvolcap.runners.optimizers import ConfigurableOptimizer, Adam
from easyvolcap.models.depth_model import DepthModel
from easyvolcap.dataloaders.volumetric_video_dataloader import VolumetricVideoDataloader
from easyvolcap.runners.evaluators.volumetric_video_evaluator import VolumetricVideoEvaluator
from easyvolcap.runners.visualizers.volumetric_video_visualizer import VolumetricVideoVisualizer
from easyvolcap.engine import RUNNERS, OPTIMIZERS, SCHEDULERS, RECORDERS, VISUALIZERS, EVALUATORS, MODERATORS, DATALOADERS  # controls the optimization loop of a particular epoch

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.timer_utils import timer
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.dist_utils import get_rank, get_distributed
from easyvolcap.utils.grad_utils import NoopClipper
from easyvolcap.utils.prof_utils import profiler_step
from easyvolcap.utils.data_utils import add_iter, remove_batch_nomod, to_cuda, get_dtype
from easyvolcap.utils.net_utils import save_model, load_model, load_network, save_npz
from aidi.utils.core4_main_val import extract_core4_tb_scalars, run_core4_main_val
from aidi.utils.vggt_main_val import normalize_eval_cfg_entries


# The outer most training loop sets lr scheduler, constructs objects etc
# The inner loop call training, logs stuff


def _should_defer_epoch_boundary_prefetch(iter_step: int, yield_every: int) -> bool:
    return yield_every > 0 and (int(iter_step) + 1) % int(yield_every) == 0


@RUNNERS.register_module()
class VolumetricVideoRunner:  # a plain and simple object controlling the training loop
    def __init__(self,
                 model: DepthModel,  # the network to train
                 dataloader: VolumetricVideoDataloader,  # enumerate this
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
                 eval_ep: int = 10,  # report validation stats
                 save_ep: int = 10,  # separately save networks (might be heavy on storage)
                 save_lim: int = 64,  # only this number of files will be kept
                 empty_cache_ep: int = 1e10,  # neven empty cache
                 save_latest_ep: int = 1,  # just in case, save regularly
                 log_interval: int = 1,  # 10ms, tune this if in realtime
                 empty_cache_interval: int = 1e10,  # MARK: SLOW
                 host_empty_cache_interval: int = 1e10,  # MARK: SLOW
                 record_interval: int = 1,  # ?ms, tune this if in realtime
                 torch_vram_frac_limit: float = 1.0,
                 parallel_dataloading: bool = True,
                 strict: bool = True,  # strict loading of network and modules?

                 resume: bool = True,
                 test_only: bool = False,
                 exp_name: str = cfg.exp_name,  # name of the experiment
                 pretrained_model: str = '',  # load this model first
                 pretrained_model_ext: str = '.pt',  # ['.pt', '.npz']
                 pretrained_load_training_state: bool = True,
                 trained_model: str = f'data/trained_model/{cfg.exp_name}',  # MARK: global configuration
                 load_epoch: int = -1,  # load different epoch to start with
                 reset_ep: int = 1e9,  # reset the training generator every x epoches
                 test_with_ddp_sharding: bool = False,

                 clip_grad_norm: float = -1,  # 1e-3,
                 clip_grad_value: float = -1,  # 40.0,
                 gradient_clipper_cfg: dotdict = dotdict(type=NoopClipper.__name__),
                 retain_last_grad: bool = False,  # setting this to true might lead to excessive VRAM usage
                 ignore_eval_error: bool = True,  # errors in evaluation will not affect training
                 record_images_to_tb: bool = True,  # when testing, save images to tensorboard
                 print_test_progress: bool = True,  # when testing, print a progress bar for indication
                 test_using_train_mode: bool = False,  # when testing, call model.train() instead of model.eval()
                 test_using_inference_mode: bool = args.type != 'train',  # MARK: global configuration
                 show_live_table: bool = False,  # avoid noisy live table in long cluster training logs
                 show_cuda_mem_snapshot: bool = False,  # avoid per-log-interval cuda_mem_fwd/bwd/opt spam by default

                 # 导出预测结果（用于后处理/PGO），默认关闭
                 export_predictions: bool = False,
                 export_dir: str = "outputs/predictions",

                 test_amp_cached: bool = True,
                 train_use_amp: bool = False,
                 test_use_amp: bool = False,
                 amp_dtype: torch.dtype = torch.bfloat16,  # torch.float16, torch.bfloat16, torch.float32
                 use_jit_trace: bool = False,  # almost will never work
                 use_jit_script: bool = False,  # almost will never work
                 use_torch_compile: bool = False,  # almost will never work
                 #  torch_compile_mode: str = None,
                 torch_compile_mode: str = 'max-autotune-no-cudagraphs',
                 float32_matmul_precision: str = 'highest',  # 'highest', 'high', 'medium'

                 # Gradient accumulation
                 gradient_accumulation_steps: int = 1,  # Number of steps to accumulate gradients

                 # Debugging
                 collect_timing: bool = False,  # will lose 1 fps over copying
                 timer_sync_cuda: bool = True,  # will explicitly call torch.cuda.synchronize() before collecting
                 timer_record_to_file: bool = False,  # will write to a json file for collected analysis of the timing
                 debug_first_iter: bool = False,
                 debug_data_loading: bool = False,
                 debug_data_loading_for_testing: bool = False,
                 test_before_first_epoch: bool = False,
                 cuda_mem_trace_cfg: dotdict = dotdict(),

                 # Main validation suite
                 main_val_cfgs: Optional[List] = None,
                 main_val_core4_cfg: dotdict = dotdict(),

                 # Extra evaluation
                 extra_eval_cfgs: Optional[List[str]] = None,
                 extra_eval_every: int = 0,  # run every N eval rounds, 0 disables
                 extra_eval_prefix: str = "EVAL_EXTRA",
                 extra_eval_record_images: bool = False,
                 extra_eval_use_ddp: bool = True,
                 ):
        self.model = model  # possibly already a ddp model?

        # Used in evaluation
        if not get_rank() or test_with_ddp_sharding:  # only build these in main process
            self.val_dataloader = val_dataloader  # different dataloader for validation
            self.evaluator: VolumetricVideoEvaluator = EVALUATORS.build(evaluator_cfg)
            self.visualizer: VolumetricVideoVisualizer = VISUALIZERS.build(visualizer_cfg)
            self.recorder: TensorboardRecorder = RECORDERS.build(recorder_cfg, resume=resume) if get_rank() == 0 else None
        
        # TODO: xiaoyang: more elegant way to check if the sampler is a DDP sharding sampler?
        if getattr(self, 'val_dataloader', None) is not None:
            _DDP_SHARDING_SAMPLER = (DistributedSequentialSampler,)
            _val_sampler = self.val_dataloader.batch_sampler.sampler
            if test_with_ddp_sharding:
                assert isinstance(_val_sampler, _DDP_SHARDING_SAMPLER), f"DDP sharding requires DistributedSequentialSampler, but got {type(_val_sampler)}"
            else:
                assert not isinstance(_val_sampler, _DDP_SHARDING_SAMPLER), f"DDP sharding requires DistributedSequentialSampler, but got {type(_val_sampler)}"
        
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value
        self.gradient_accumulation_steps = max(int(gradient_accumulation_steps), 1)

        if not test_only:
            self.dataloader = dataloader
            self.optimizer: Adam = OPTIMIZERS.build(optimizer_cfg, named_params=((k, v) for k, v in model.named_parameters() if v.requires_grad))  # requires parameters
            decay_iter = (epochs if decay_epochs < 0 else decay_epochs) * ep_iter
            # Scheduler steps once per optimizer update, so align decay_iter with update count.
            decay_iter = max(decay_iter // self.gradient_accumulation_steps, 1)
            self.scheduler: ExponentialLR = SCHEDULERS.build(scheduler_cfg, optimizer=self.optimizer, decay_iter=decay_iter)  # requires parameters
            self.moderator: DatasetRatioModerator = MODERATORS.build(moderator_cfg, runner=self, total_iter=epochs * ep_iter)  # after dataset init
            self.clipper: NoopClipper = OPTIMIZERS.build(gradient_clipper_cfg)
            self.clipper.setup_clipping(self.model)
            if not isinstance(self.clipper, NoopClipper):
                assert clip_grad_norm <= 0 and clip_grad_value <= 0, "Cannot use both gradient clipping and custom gradient clipper at the same time, please choose one of them"

        self.exp_name = exp_name
        self.epochs = epochs
        self.ep_iter = ep_iter
        self.eval_ep = eval_ep
        self.save_ep = save_ep
        self.save_lim = save_lim
        self.reset_ep = reset_ep
        self.empty_cache_ep = empty_cache_ep
        self.save_latest_ep = save_latest_ep
        self.log_interval = log_interval
        self.record_interval = record_interval
        self.empty_cache_interval = empty_cache_interval
        self.host_empty_cache_interval = host_empty_cache_interval
        self.parallel_dataloading = parallel_dataloading

        self.resume = resume
        self.strict = strict
        self.load_epoch = load_epoch
        self.trained_model = trained_model
        self.pretrained_model = pretrained_model
        self.pretrained_model_ext = pretrained_model_ext
        self.pretrained_load_training_state = pretrained_load_training_state
        self.test_with_ddp_sharding = test_with_ddp_sharding

        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value
        self.retain_last_grad = retain_last_grad

        # Use auto mixed precision
        self.test_use_amp = test_use_amp
        self.train_use_amp = train_use_amp
        self.test_amp_cached = test_amp_cached
        self.amp_dtype = get_dtype(amp_dtype)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.test_use_amp or self.train_use_amp) and self.amp_dtype != torch.bfloat16)

        # Trace model for faster inference
        self.use_jit_script = use_jit_script
        self.use_jit_trace = use_jit_trace
        self.use_torch_compile = use_torch_compile
        self.torch_compile_mode = torch_compile_mode

        # Precision
        self.float32_matmul_precision = float32_matmul_precision
        # NOTE: important, which may cause precision issues when dealing with like camera pose, etc.
        # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
        torch.set_float32_matmul_precision(self.float32_matmul_precision)  # set the precision for matmul

        self.ignore_eval_error = ignore_eval_error
        self.record_images_to_tb = record_images_to_tb
        self.print_test_progress = print_test_progress
        self.test_using_train_mode = test_using_train_mode
        self.test_using_inference_mode = test_using_inference_mode
        self.show_live_table = show_live_table
        self.show_cuda_mem_snapshot = show_cuda_mem_snapshot

        # Export predictions for downstream processing (e.g., PGO). Default off.
        self.export_predictions = export_predictions
        self.export_dir = export_dir

        # Setting VRAM limit on Windows might make the framerate more stable
        try:
            torch.cuda.set_per_process_memory_fraction(torch_vram_frac_limit)  # set vram usage limit to current device
        except:
            pass

        # HACK: GLOBAL VARIABLE, when dumping config, should ignore this one
        cfg.runner = self
        # cfg\..* = # this search will find all global config assignment
        # We need to perform the dumping before the global config to keep things clean

        # Debugging
        self.collect_timing = collect_timing  # another fancy self.timer (different from fps counter)
        self.timer_sync_cuda = timer_sync_cuda  # this enables accurate time recording for each section, but would slow down the programs
        self.timer_record_to_file = timer_record_to_file
        self.debugging_signal_received = debug_first_iter
        self.debug_data_loading = debug_data_loading
        self.debug_data_loading_for_testing = debug_data_loading_for_testing
        self.test_before_first_epoch = test_before_first_epoch
        self._mem_trace_root = os.getcwd()
        self._init_cuda_mem_trace(cuda_mem_trace_cfg)

        self.main_val_cfgs = normalize_eval_cfg_entries(main_val_cfgs)
        self.main_val_core4_cfg = dotdict(copy.deepcopy(main_val_core4_cfg or dotdict()))
        self.extra_eval_cfgs = extra_eval_cfgs or []
        self.extra_eval_every = int(extra_eval_every) if extra_eval_every else 0
        self.extra_eval_prefix = extra_eval_prefix
        self.extra_eval_record_images = extra_eval_record_images
        self.extra_eval_use_ddp = extra_eval_use_ddp
        self._extra_eval_counter = 0
        self._extra_eval_dataloaders = {}
        self.eval_start_step = self._infer_eval_start_step()

        def signal_handler(signum, frame):
            self.debugging_signal_received = True
            log(yellow(f"Received debugging signal: {signum}, marking for breakpoint: {self.debugging_signal_received}"))

        signal.signal(signal.SIGUSR1 if os.name != 'nt' else signal.SIGBREAK, signal_handler)

    @property
    def collect_timing(self):
        return not timer.disabled

    @property
    def timer_sync_cuda(self):
        return timer.sync_cuda

    @property
    def timer_record_to_file(self):
        return timer.record_to_file

    def _infer_eval_start_step(self) -> int:
        model = self.model.module if hasattr(self.model, "module") else self.model
        indexer_cfg = getattr(model, "indexer_cfg", None)
        if not indexer_cfg:
            return 0
        if not indexer_cfg.get("enabled", False):
            return 0
        if not indexer_cfg.get("enable_sparse", True):
            return 0
        warmup_steps = int(indexer_cfg.get("warmup_steps", 0))
        sparse_start = int(indexer_cfg.get("sparse_start_step", warmup_steps))
        return max(0, sparse_start)

    @staticmethod
    def _bytes_to_gib(value: int) -> float:
        return value / float(2**30)

    def _cuda_memory_snapshot(self) -> dotdict:
        if not torch.cuda.is_available():
            return dotdict()
        stats = torch.cuda.memory_stats()
        return dotdict(
            device=torch.cuda.current_device(),
            alloc_gib=self._bytes_to_gib(torch.cuda.memory_allocated()),
            reserved_gib=self._bytes_to_gib(torch.cuda.memory_reserved()),
            max_alloc_gib=self._bytes_to_gib(torch.cuda.max_memory_allocated()),
            max_reserved_gib=self._bytes_to_gib(torch.cuda.max_memory_reserved()),
            active_gib=self._bytes_to_gib(stats.get("active_bytes.all.current", 0)),
            inactive_gib=self._bytes_to_gib(stats.get("inactive_split_bytes.all.current", 0)),
        )

    @staticmethod
    def _format_cuda_snapshot(snapshot: dotdict) -> str:
        return (
            f"alloc={snapshot.alloc_gib:.2f} GiB, reserved={snapshot.reserved_gib:.2f} GiB, "
            f"max_alloc={snapshot.max_alloc_gib:.2f} GiB, max_reserved={snapshot.max_reserved_gib:.2f} GiB, "
            f"active={snapshot.active_gib:.2f} GiB, inactive={snapshot.inactive_gib:.2f} GiB"
        )

    def _init_cuda_mem_trace(self, cuda_mem_trace_cfg: dotdict) -> None:
        defaults = dotdict(
            enabled=False,
            interval=1,
            stages=["fwd", "bwd", "opt"],
            min_peak_delta_mib=0.0,
            topk=20,
            capture_snapshot=True,
            capture_full_snapshot=False,
            output_dir="cuda_mem",
            sync_cuda=True,
            record_history=False,
            include_inactive=False,
            include_pending_free=True,
            log_json=False,
            log_snapshot=False,
            log_snapshot_max_chars=0,
            log_hist=False,
            max_snapshots=0,
        )
        cfg_local = dotdict(defaults)
        if cuda_mem_trace_cfg:
            try:
                cfg_local.update(cuda_mem_trace_cfg)
            except Exception:
                cfg_local.update(dict(cuda_mem_trace_cfg))

        def _env_flag(name: str) -> bool:
            return os.getenv(name, "").lower() in ("1", "true", "yes", "y", "on")

        if _env_flag("EVC_CUDA_MEM_TRACE"):
            cfg_local.enabled = True
        if _env_flag("EVC_CUDA_MEM_TRACE_ALL_RANKS"):
            cfg_local.all_ranks = True
        if _env_flag("EVC_CUDA_MEM_TRACE_HISTORY"):
            cfg_local.record_history = True
        if _env_flag("CUDA_MEM_TRACE_LOG_SNAPSHOT"):
            cfg_local.log_snapshot = True
        if _env_flag("CUDA_MEM_TRACE_LOG_HIST"):
            cfg_local.log_hist = True
        if _env_flag("CUDA_MEM_TRACE_LOG_JSON"):
            cfg_local.log_json = True

        stages = cfg_local.get("stages", [])
        if isinstance(stages, str):
            stages = [item.strip() for item in stages.replace(";", ",").split(",") if item.strip()]
        cfg_local.stages = stages
        cfg_local.interval = max(int(cfg_local.get("interval", 1)), 1)
        cfg_local.topk = max(int(cfg_local.get("topk", 0)), 0)
        cfg_local.min_peak_delta_mib = float(cfg_local.get("min_peak_delta_mib", 0.0) or 0.0)
        cfg_local.max_snapshots = max(int(cfg_local.get("max_snapshots", 0) or 0), 0)
        cfg_local.log_snapshot_max_chars = max(int(cfg_local.get("log_snapshot_max_chars", 0) or 0), 0)

        self.cuda_mem_trace_cfg = cfg_local
        self._cuda_mem_trace_state = dotdict(
            max_alloc=0,
            max_reserved=0,
            snapshot_count=0,
            history_enabled=False,
            history_warned=False,
            snapshot_warned=False,
        )
        self._maybe_enable_cuda_mem_history()

    def _cuda_mem_trace_enabled(self) -> bool:
        if not torch.cuda.is_available():
            return False
        cfg_local = getattr(self, "cuda_mem_trace_cfg", dotdict())
        if not cfg_local.get("enabled", False):
            return False
        if get_rank() != 0 and not cfg_local.get("all_ranks", False):
            return False
        return True

    def _maybe_enable_cuda_mem_history(self) -> None:
        cfg_local = getattr(self, "cuda_mem_trace_cfg", dotdict())
        if not cfg_local.get("record_history", False):
            return
        if self._cuda_mem_trace_state.get("history_enabled", False):
            return
        record_fn = getattr(torch.cuda.memory, "record_memory_history", None)
        if record_fn is None:
            record_fn = getattr(torch.cuda.memory, "_record_memory_history", None)
        if record_fn is None:
            if not self._cuda_mem_trace_state.get("history_warned", False):
                log("[cuda_mem_trace] record_memory_history not available, stack frames may be empty")
                self._cuda_mem_trace_state.history_warned = True
            return
        try:
            record_fn(enabled=True)
        except TypeError:
            try:
                record_fn(True)
            except Exception as exc:
                if not self._cuda_mem_trace_state.get("history_warned", False):
                    log(f"[cuda_mem_trace] enable record_memory_history failed: {type(exc).__name__}")
                    self._cuda_mem_trace_state.history_warned = True
                return
        except Exception as exc:
            if not self._cuda_mem_trace_state.get("history_warned", False):
                log(f"[cuda_mem_trace] enable record_memory_history failed: {type(exc).__name__}")
                self._cuda_mem_trace_state.history_warned = True
            return
        self._cuda_mem_trace_state.history_enabled = True

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        if num_bytes >= 2**30:
            return f"{num_bytes / 2**30:.2f} GiB"
        return f"{num_bytes / 2**20:.1f} MiB"

    def _shorten_path(self, path: str) -> str:
        if not path:
            return "unknown"
        try:
            if path.startswith(self._mem_trace_root):
                return os.path.relpath(path, self._mem_trace_root)
        except Exception:
            pass
        return os.path.basename(path)

    def _format_snapshot_location(self, block: dict) -> str:
        frames = block.get("frames") or block.get("stack") or block.get("stack_trace") or block.get("traceback") or []
        if isinstance(frames, dict):
            frames = frames.get("frames", []) or []
        if isinstance(frames, str):
            return frames
        if frames:
            frame = frames[0] or {}
            filename = self._shorten_path(frame.get("filename") or frame.get("file") or "")
            line = frame.get("line") or frame.get("lineno") or 0
            func = frame.get("name") or frame.get("function") or ""
            if func:
                return f"{filename}:{line} {func}"
            return f"{filename}:{line}"
        return "unknown"

    @staticmethod
    def _to_builtin(obj):
        if isinstance(obj, dotdict):
            return {k: VolumetricVideoRunner._to_builtin(v) for k, v in obj.items()}
        if isinstance(obj, dict):
            return {k: VolumetricVideoRunner._to_builtin(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [VolumetricVideoRunner._to_builtin(v) for v in obj]
        return obj

    @staticmethod
    def _sum_tensor_bytes(value) -> int:
        if torch.is_tensor(value):
            if value.is_cuda:
                return int(value.numel()) * int(value.element_size())
            return 0
        if isinstance(value, dict):
            return sum(VolumetricVideoRunner._sum_tensor_bytes(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(VolumetricVideoRunner._sum_tensor_bytes(v) for v in value)
        return 0

    def _summarize_model_cuda_bytes(self) -> dotdict:
        model = self.model.module if hasattr(self.model, "module") else self.model
        params_bytes = 0
        grads_bytes = 0
        buffers_bytes = 0
        for param in model.parameters():
            if param.is_cuda:
                params_bytes += int(param.numel()) * int(param.element_size())
            if param.grad is not None and torch.is_tensor(param.grad) and param.grad.is_cuda:
                grads_bytes += int(param.grad.numel()) * int(param.grad.element_size())
        for buf in model.buffers():
            if buf.is_cuda:
                buffers_bytes += int(buf.numel()) * int(buf.element_size())
        optim_bytes = 0
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            for state in optimizer.state.values():
                optim_bytes += self._sum_tensor_bytes(state)
        return dotdict(
            params=params_bytes,
            grads=grads_bytes,
            buffers=buffers_bytes,
            optimizer=optim_bytes,
        )

    def _summarize_cuda_snapshot(self, snapshot: dict, topk: int, include_inactive: bool, include_pending_free: bool) -> dotdict:
        totals_by_state = defaultdict(int)
        totals_by_segment_type = defaultdict(int)
        active_bytes = 0
        active_requested_bytes = 0
        by_location = defaultdict(int)
        unknown_sizes = Counter()
        inactive_sizes = Counter()

        if isinstance(snapshot, list):
            segments = snapshot
        elif isinstance(snapshot, dict):
            if "segments" in snapshot:
                segments = snapshot.get("segments", [])
            elif "data" in snapshot:
                segments = snapshot.get("data", [])
            else:
                segments = []
        else:
            segments = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            seg_type = segment.get("segment_type", segment.get("type", "unknown"))
            totals_by_segment_type[seg_type] += int(segment.get("total_size", 0) or 0)
            blocks = segment.get("blocks", []) or []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                state = block.get("state", "unknown")
                size = int(block.get("size", block.get("allocated_size", 0)) or 0)
                requested = int(block.get("requested_size", block.get("requested", size)) or size)
                totals_by_state[state] += size
                if str(state).startswith("active"):
                    active_bytes += size
                    active_requested_bytes += requested
                if str(state).startswith("inactive"):
                    inactive_sizes[size] += 1
                if not include_inactive and str(state).startswith("inactive"):
                    continue
                if not include_pending_free and state == "active_pending_free":
                    continue
                if state == "active_allocated" or (include_pending_free and str(state).startswith("active")):
                    location = self._format_snapshot_location(block)
                    by_location[location] += size
                    if location == "unknown":
                        unknown_sizes[size] += 1

        top_allocations = []
        if topk > 0:
            for location, size in sorted(by_location.items(), key=lambda item: item[1], reverse=True)[:topk]:
                top_allocations.append(dotdict(location=location, bytes=size))

        return dotdict(
            totals_by_state=dotdict({k: int(v) for k, v in totals_by_state.items()}),
            totals_by_segment_type=dotdict({k: int(v) for k, v in totals_by_segment_type.items()}),
            active_bytes=int(active_bytes),
            active_requested_bytes=int(active_requested_bytes),
            top_allocations=top_allocations,
            unknown_sizes=unknown_sizes,
            inactive_sizes=inactive_sizes,
        )

    def _resolve_cuda_mem_trace_dir(self) -> str:
        cfg_local = getattr(self, "cuda_mem_trace_cfg", dotdict())
        output_dir = cfg_local.get("output_dir", "") or "cuda_mem"
        if os.path.isabs(output_dir):
            return output_dir
        if getattr(self, "recorder", None) is not None and hasattr(self.recorder, "record_dir"):
            return os.path.join(self.recorder.record_dir, output_dir)
        if getattr(self, "trained_model", None):
            return os.path.join(self.trained_model, output_dir)
        return os.path.join(self._mem_trace_root, output_dir)

    def _write_cuda_mem_trace_summary(self, summary: dict, tag: str, snapshot: Optional[dict]) -> None:
        out_dir = self._resolve_cuda_mem_trace_dir()
        os.makedirs(out_dir, exist_ok=True)
        summary_path = os.path.join(out_dir, f"summary_{tag}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)
        cfg_local = getattr(self, "cuda_mem_trace_cfg", dotdict())
        if cfg_local.get("capture_full_snapshot", False) and snapshot is not None:
            snapshot_path = os.path.join(out_dir, f"snapshot_{tag}.json")
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=True)
        log(f"[cuda_mem_peak] summary_saved={summary_path}")

    def _mem_trace_phase(self, step: int) -> str:
        warmup_total = max(0, int(getattr(self, "eval_start_step", 0) or 0))
        if warmup_total <= 0:
            return "train"
        return "warmup" if (step + 1) <= warmup_total else "sparse"

    def _maybe_capture_cuda_mem_trace(self, stage: str, step: int) -> None:
        if not self._cuda_mem_trace_enabled():
            return
        cfg_local = getattr(self, "cuda_mem_trace_cfg", dotdict())
        if cfg_local.get("stages", []) and stage not in cfg_local.get("stages", []):
            return
        if (step + 1) % cfg_local.interval != 0:
            return
        if cfg_local.max_snapshots and self._cuda_mem_trace_state.snapshot_count >= cfg_local.max_snapshots:
            return
        if cfg_local.get("sync_cuda", True):
            torch.cuda.synchronize()

        max_alloc = int(torch.cuda.max_memory_allocated())
        max_reserved = int(torch.cuda.max_memory_reserved())
        delta_bytes = int(cfg_local.min_peak_delta_mib * 2**20)
        if max_alloc <= self._cuda_mem_trace_state.max_alloc + delta_bytes and max_reserved <= self._cuda_mem_trace_state.max_reserved + delta_bytes:
            return

        self._cuda_mem_trace_state.max_alloc = max(self._cuda_mem_trace_state.max_alloc, max_alloc)
        self._cuda_mem_trace_state.max_reserved = max(self._cuda_mem_trace_state.max_reserved, max_reserved)

        cur_alloc = int(torch.cuda.memory_allocated())
        cur_reserved = int(torch.cuda.memory_reserved())
        phase = self._mem_trace_phase(step)

        summary = dotdict(
            iter=step,
            step=step + 1,
            stage=stage,
            phase=phase,
            rank=int(get_rank()),
            max_alloc_bytes=max_alloc,
            max_reserved_bytes=max_reserved,
            current_alloc_bytes=cur_alloc,
            current_reserved_bytes=cur_reserved,
        )

        snapshot = None
        if cfg_local.get("capture_snapshot", True):
            snapshot_fn = getattr(torch.cuda, "memory_snapshot", None)
            if snapshot_fn is None:
                if not self._cuda_mem_trace_state.get("snapshot_warned", False):
                    log("[cuda_mem_trace] memory_snapshot not available, skip allocation breakdown")
                    self._cuda_mem_trace_state.snapshot_warned = True
            else:
                try:
                    snapshot = snapshot_fn()
                except Exception as exc:
                    if not self._cuda_mem_trace_state.get("snapshot_warned", False):
                        log(f"[cuda_mem_trace] memory_snapshot failed: {type(exc).__name__}")
                        self._cuda_mem_trace_state.snapshot_warned = True
                else:
                    summary.update(
                        self._summarize_cuda_snapshot(
                            snapshot,
                            cfg_local.topk,
                            cfg_local.get("include_inactive", False),
                            cfg_local.get("include_pending_free", True),
                        )
                    )
                    if cfg_local.get("log_snapshot", False):
                        try:
                            payload = json.dumps(snapshot, ensure_ascii=True)
                        except Exception as exc:
                            log(f"[cuda_mem_peak_snapshot] encode_failed={type(exc).__name__}")
                        else:
                            max_chars = int(cfg_local.get("log_snapshot_max_chars", 0) or 0)
                            if max_chars > 0 and len(payload) > max_chars:
                                payload = payload[:max_chars] + "...TRUNCATED"
                            log(f"[cuda_mem_peak_snapshot] {payload}")

        summary.tensor_bytes = self._summarize_model_cuda_bytes()
        tag = f"iter{step + 1:06d}_stage{stage}_phase{phase}_rank{get_rank()}"
        self._write_cuda_mem_trace_summary(summary, tag, snapshot)
        self._log_cuda_mem_trace(summary)
        self._cuda_mem_trace_state.snapshot_count += 1

    def _log_cuda_mem_trace(self, summary: dotdict) -> None:
        log(
            "[cuda_mem_peak] "
            f"iter={summary.step} stage={summary.stage} phase={summary.phase} "
            f"max_alloc={self._format_bytes(summary.max_alloc_bytes)} "
            f"max_reserved={self._format_bytes(summary.max_reserved_bytes)} "
            f"current_alloc={self._format_bytes(summary.current_alloc_bytes)} "
            f"current_reserved={self._format_bytes(summary.current_reserved_bytes)}"
        )
        tensor_bytes = summary.get("tensor_bytes", dotdict())
        if tensor_bytes:
            log(
                "[cuda_mem_peak] tensors "
                f"params={self._format_bytes(tensor_bytes.get('params', 0))} "
                f"grads={self._format_bytes(tensor_bytes.get('grads', 0))} "
                f"buffers={self._format_bytes(tensor_bytes.get('buffers', 0))} "
                f"optimizer={self._format_bytes(tensor_bytes.get('optimizer', 0))}"
            )
        totals_by_state = summary.get("totals_by_state", None)
        if totals_by_state:
            items = [f"{k}={self._format_bytes(v)}" for k, v in totals_by_state.items()]
            log("[cuda_mem_peak] state_bytes " + " ".join(items))
        totals_by_segment = summary.get("totals_by_segment_type", None)
        if totals_by_segment:
            items = [f"{k}={self._format_bytes(v)}" for k, v in totals_by_segment.items()]
            log("[cuda_mem_peak] segment_bytes " + " ".join(items))
        top_allocs = summary.get("top_allocations", [])
        if top_allocs:
            log("[cuda_mem_peak] top_allocations")
            for idx, entry in enumerate(top_allocs, start=1):
                log(f"  {idx:02d}. {self._format_bytes(entry.bytes)} {entry.location}")
        cfg_local = getattr(self, "cuda_mem_trace_cfg", dotdict())
        if cfg_local.get("log_hist", False):
            unknown_sizes = summary.get("unknown_sizes", None)
            if unknown_sizes:
                log("[cuda_mem_peak] unknown_size_hist")
                for size, count in sorted(unknown_sizes.items(), key=lambda item: item[0], reverse=True)[:10]:
                    log(f"  {self._format_bytes(size)} x{count} total={self._format_bytes(size * count)}")
            inactive_sizes = summary.get("inactive_sizes", None)
            if inactive_sizes:
                log("[cuda_mem_peak] inactive_size_hist")
                for size, count in sorted(inactive_sizes.items(), key=lambda item: item[0], reverse=True)[:10]:
                    log(f"  {self._format_bytes(size)} x{count} total={self._format_bytes(size * count)}")
        if cfg_local.get("log_json", False):
            try:
                payload = json.dumps(self._to_builtin(summary), ensure_ascii=True)
            except Exception as exc:
                log(f"[cuda_mem_peak_json] encode_failed={type(exc).__name__}")
            else:
                log(f"[cuda_mem_peak_json] {payload}")

    @collect_timing.setter
    def collect_timing(self, val: bool):
        timer.disabled = not val

    @timer_sync_cuda.setter
    def timer_sync_cuda(self, val: bool):
        timer.sync_cuda = val

    @timer_record_to_file.setter
    def timer_record_to_file(self, val: bool):
        timer.record_to_file = val
        if timer.record_to_file and hasattr(self, 'recorder'):
            log(yellow(f'Will record timing results to {blue(join(self.recorder.record_dir, f"timing.json"))}'))
            timer.exp_name = self.exp_name
            timer.record_dir = self.recorder.record_dir
            if not hasattr(timer, 'timing_record'):
                timer.timing_record = dotdict()

    @property
    def total_iter(self):
        return self.epochs * self.ep_iter

    def load_network(self):
        if self.pretrained_model:  # maybe load pretrain model
            epoch = load_network(model=self.model,  # only loading the network, without recorder?
                                 model_dir=self.pretrained_model,
                                 strict=self.strict,
                                 )  # loads the next epoch to use
            self._after_pretrained_model_loaded()

        epoch = load_network(model=self.model,  # only loading the network, without recorder?
                             model_dir=self.trained_model,
                             resume=self.resume,
                             epoch=self.load_epoch,
                             strict=self.strict,
                             )  # loads the next epoch to use
        return epoch

    def load_model(self):
        epoch = 0
        if self.pretrained_model:  # maybe load pretrain model
            if self.pretrained_load_training_state:
                epoch = load_model(model=self.model,
                                   optimizer=self.optimizer,
                                   scheduler=self.scheduler,
                                   moderator=self.moderator,
                                   model_dir=self.pretrained_model,
                                   ext=self.pretrained_model_ext,
                                   strict=self.strict,
                                   )  # loads the next epoch to use
            else:
                epoch = load_model(model=self.model,
                                   optimizer=None,
                                   scheduler=None,
                                   moderator=None,
                                   model_dir=self.pretrained_model,
                                   ext=self.pretrained_model_ext,
                                   strict=self.strict,
                                   )  # loads the next epoch to use
            self._after_pretrained_model_loaded()

        resume_epoch = load_model(model=self.model,
                                  optimizer=self.optimizer,
                                  scheduler=self.scheduler,
                                  moderator=self.moderator,
                                  model_dir=self.trained_model,
                                  resume=self.resume,
                                  epoch=self.load_epoch,
                                  strict=self.strict,
                                  )  # loads the next epoch to use
        return self._resolve_begin_epoch(epoch, resume_epoch)

    def _resolve_begin_epoch(self, pretrained_epoch: int, resume_epoch: int):
        if resume_epoch > 0:
            return resume_epoch
        # Fresh experiments that only load weights from a warmup/pretrain checkpoint
        # should restart scheduling and logging from epoch 0.
        if self.pretrained_model and not self.pretrained_load_training_state:
            return 0
        return pretrained_epoch

    def _after_pretrained_model_loaded(self):
        model = self.model.module if isinstance(self.model, DDP) else self.model
        hook = getattr(model, "after_pretrained_model_loaded", None)
        if hook is not None:
            hook(self)

    def save_network(self, epoch, latest: bool = True, **kwargs):
        try:
            save_model(model=self.model,
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
            save_npz(model=self.model,
                     model_dir=self.trained_model,
                     epoch=epoch,
                     **kwargs,
                     )
        except RuntimeError as e:
            log(red(e))
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
            save_model(model=self.model,
                       optimizer=self.optimizer if save_optimizer else None,
                       scheduler=self.scheduler if save_scheduler else None,
                       moderator=self.moderator if save_moderator else None,
                       model_dir=self.trained_model,
                       save_lim=self.save_lim,
                       epoch=epoch,
                       latest=latest,
                       **kwargs,
                       )
        except RuntimeError as e:
            log(red(e))
            torch.cuda.empty_cache()

    def save_warmup_checkpoint(self, step: int, tag: str = "warmup_end"):
        if get_rank():
            return
        try:
            epoch = int(step // self.ep_iter) if self.ep_iter > 0 else 0
            log(green(f"Saving warmup checkpoint to {self.trained_model} ({tag})"))
            self.save_model(
                epoch,
                latest=True,
                save_optimizer=False,
                save_scheduler=False,
                save_moderator=False,
            )
            src = join(self.trained_model, "latest.pt")
            dst = join(self.trained_model, f"{tag}.pt")
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                log(green(f"Saved warmup checkpoint to {dst}"))
            else:
                log(red(f"Warmup checkpoint source missing: {src}"))
        except Exception as e:
            log(red("Error in warmup checkpoint saving, ignored and continuing"))
            stacktrace()
            stop_prog()

    def maybe_jit_model(self, batch: dotdict = None):
        if not isinstance(self.model, torch.jit.ScriptModule):
            if self.use_jit_script:
                log(green(f'Scripting the model'))
                self.model = torch.jit.script(self.model)
            elif self.use_jit_trace:
                log(green(f'Tracing the model'))
                self.model = torch.jit.trace(self.model, batch)
            elif self.use_torch_compile:
                log(green(f'Compiling the model'))
                self.model = torch.compile(self.model, mode=self.torch_compile_mode)

    def _sync_grad_scaler_found_inf(self) -> None:
        if not get_distributed() or not self.scaler.is_enabled():
            return
        optimizer_state = self.scaler._per_optimizer_states.get(id(self.optimizer), None)
        if not optimizer_state:
            return
        found_inf_per_device = optimizer_state.get("found_inf_per_device", None)
        if not found_inf_per_device:
            return

        found_values = list(found_inf_per_device.values())
        global_found_inf = found_values[0].detach().clone()
        for value in found_values[1:]:
            global_found_inf = global_found_inf + value.detach().to(device=global_found_inf.device)
        dist.all_reduce(global_found_inf, op=dist.ReduceOp.MAX)
        for value in found_values:
            value.copy_(global_found_inf.to(device=value.device, dtype=value.dtype))

    # Single epoch testing api
    def test(self):  # from begin epoch
        epoch = self.load_network()
        self.test_epoch(epoch)

    # Epoch based runner
    def train(self):  # from begin epoch
        epoch = self.load_model()

        # The actual training for this epoch
        train_generator = self.train_generator(epoch, self.ep_iter)  # yield every ep iter

        def should_eval(step: int) -> bool:
            return step >= self.eval_start_step

        # train the network
        if self.debug_data_loading_for_testing or self.test_before_first_epoch:
            if (not get_rank() or self.test_with_ddp_sharding):
                eval_step = epoch * self.ep_iter
                if should_eval(eval_step):
                    self._run_validation(epoch)  # only for debugging
                    log(green(f"Test before first epoch done"))

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
                    if not get_rank():
                        log(green(f'Saving model to {self.trained_model}'))
                    self.save_model(epoch, latest=False)
                    if not get_rank():
                        log("Saving model to disk done")
                except Exception as e:
                    log(red('Error in model saving, ignored and continuing'))
                    stacktrace()
                    stop_prog()  # stop it, otherwise multiple lives

            if (epoch + 1) % self.save_latest_ep == 0:
                try:
                    if not get_rank():
                        log(green(f'Saving model to {self.trained_model} (latest)'))
                    self.save_model(epoch, latest=True)
                    if not get_rank():
                        self.save_npz(epoch, latest=True)  # for inference and smaller file size
                        log("Saving model to disk done (latest)")
                except Exception as e:
                    log(red('Error in model saving, ignored and continuing'))
                    stacktrace()
                    stop_prog()  # stop it, otherwise multiple lives
                
            # Perform validation run if required
            if self.eval_ep > 0 and (epoch + 1) % self.eval_ep == 0 and (not get_rank() or self.test_with_ddp_sharding):
                eval_step = (epoch + 1) * self.ep_iter
                if should_eval(eval_step):
                    try:
                        log(green(f'Performing validation pass, epoch {epoch + 1}'))
                        self._run_validation(epoch + 1)  # will this provoke a live display?
                        log("Validation pass done")
                        self._maybe_run_extra_evals(epoch + 1)
                    except Exception as e:
                        log(red('Error in validation pass, ignored and continuing'))
                        stacktrace()
                        stop_prog()  # stop it, otherwise multiple lives
                        if not self.ignore_eval_error:
                            raise e

            if (epoch + 1) % self.reset_ep == 0:
                log(green(f'Resetting the training generator'))
                train_generator = self.train_generator(epoch + 1, self.ep_iter)  # yield every ep iter

    def save_predictions_to_pkl(self, epoch, iter, batch, output, save_dir, metrics=None):
        import pickle
        import cv2

        os.makedirs(save_dir, exist_ok=True)
        if not hasattr(self, 'dataset_statistics'):
            self.dataset_statistics = defaultdict(lambda: defaultdict(int))
        dataset_statistics = self.dataset_statistics

        N = batch.rgb.shape[1]
        H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
        scalar_stats = {k: (v.item() if hasattr(v, 'item') else v) for k, v in output.scalar_stats.items()}
        scalar_stats['H'] = H
        scalar_stats['W'] = W
        scalar_stats['iter'] = iter
        
        # if batch.rgb.shape
        rgbs = batch.rgb[0].reshape(N, H, W, 3).cpu().numpy()
        c2ws = batch.c2ws[0].reshape(N, 4, 4).cpu().numpy()
        Ks = batch.ixts[0].reshape(N, 3, 3).cpu().numpy()
        depth_gts = batch.dpt[0].reshape(N, H, W).cpu().numpy()
        gt_mask = batch.msk[0].reshape(N, H, W).cpu().numpy()
        seg = batch.get('seg', None)
        if seg is not None:
            seg = seg[0].reshape(N, H, W).cpu().numpy()
            
        
        depth_preds = output.dpt_map[0].reshape(N, H, W).detach().cpu().numpy()
        xyzs = output.xyz_map[0].reshape(N, H, W, 3).detach().cpu().numpy()
        cam_map = output.cam_map[:1]
        conf_preds = output.dpt_cnf[0].reshape(N, H, W).detach().cpu().numpy()

        depth_preds = depth_preds * gt_mask.astype(np.float32)
        
        w2cs_pred, ixts_pred = decode_camera_params(
            cam_map, H, W,
        )
        w2cs_pred = w2cs_pred.reshape(N, 3, 4).cpu().numpy()
        w2cs_pred = np.pad(w2cs_pred, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
        w2cs_pred[:, 3, 3] = 1.0
        w2cs_pred = w2cs_pred.reshape(N, 4, 4)
        c2ws_pred = np.linalg.inv(w2cs_pred)
        ixts_pred = ixts_pred.reshape(N, 3, 3).cpu().numpy()

        assert c2ws.shape == c2ws_pred.shape
        assert Ks.shape == ixts_pred.shape

        data = {
            'rgbs': rgbs,
            'c2ws': c2ws,
            'Ks': Ks,
            'depth_gts': depth_gts,
            'depth_preds': depth_preds,
            'gt_mask': gt_mask,
            'xyzs': xyzs,
            'c2ws_pred': c2ws_pred,
            'Ks_pred': ixts_pred,
            'conf_preds': conf_preds,
            'seg': seg,
            'meta': {
                'H': H,
                'W': W,
                'iter': iter,
                'N': N,
                'scalar_stats': scalar_stats,
                'aspect_ratio': batch.meta.aspect_ratio[0].item(),
                'original_aspect_ratio': batch.meta.original_aspect_ratio[0].item() if hasattr(batch.meta, 'original_aspect_ratio') else "unknown",
                'data_root': batch.meta.data_root[0] if hasattr(batch.meta, 'data_root') else "unknown",
                'dataset_name': batch.meta.dataset_name[0] if hasattr(batch.meta, 'dataset_name') else "unknown",
                'metrics': metrics,
            }
        }

        dataset_name = data['meta']['dataset_name'][0]
        dataset_statistics[dataset_name]['total'] += 1
        total_iter = sum(dataset_statistics[dataset_name]['total'] for dataset_name in dataset_statistics)
        # save to pickle
        save_path = os.path.join(save_dir, f'log_iter_{iter}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

        # draw image
        N, H, W, _ = rgbs.shape
        rgbs = [x.copy() for x in rgbs]
        meta = data['meta']
        for i in range(N):
            K = Ks[i]
            # draw text on the image with small text size
            lines = [
                # split data root into 2 lines if too long
                f"dataset_name: {meta['dataset_name']}",
                f"data_root: {meta['data_root'][0][:50]}",
                f"{meta['data_root'][0][50:]}" if len(meta['data_root'][0]) > 50 else "",
                f"fx,fy,cx,cy: {K[0, 0]:.2f}, {K[1, 1]:.2f}, {K[0, 2]:.2f}, {K[1, 2]:.2f}",
                f"2cx,2cy: {2 * K[0, 2]:.2f}, {2 * K[1, 2]:.2f}",
                f"WxH: {W}x{H}",
                f"aspect ratio: {H / W:.2f}, target: {meta['aspect_ratio']:.2f}, original: {meta['original_aspect_ratio']:.2f}",
                f"iter: {iter}",
            ]
            for line_id in range(len(lines)):
                cv2.putText(rgbs[i], lines[line_id], (10, 10 + line_id * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        rgbs = np.vstack(rgbs)
        save_path = os.path.join(save_dir, f'image_iter_{iter}.png')
        cv2.imwrite(save_path, rgbs[:, :, ::-1] * 255)
        return save_path


    # Iteration based runner
    def train_generator(self, begin_epoch: int, yield_every: int = 1):
        # Train for one epoch (iterator style)
        # Actual start of the execution
        epoch = begin_epoch  # set starting epoch
        self.maybe_jit_model()
        self.model.train()  # set the network (model) to training mode (recursive to all modules)
        start_time = time.perf_counter()
        launch_time = start_time

        # Overlap backward pass of previous iteration with dataloading of next iteration
        enumerater = enumerate(self.dataloader)
        index = 0
        batch = dotdict()
        data_stream: torch.cuda.Stream = torch.cuda.Stream() if self.parallel_dataloading else torch.cuda.current_stream()

        def prefetch_next_batch():
            with torch.cuda.stream(data_stream):
                next_index, next_batch = next(enumerater, (None, None))
                if next_batch is None:
                    next_batch = dotdict(meta=dotdict(iter=-1))
                else:
                    next_batch = add_iter(next_batch, begin_epoch * self.ep_iter + next_index, self.total_iter)  # is this bad naming
                    next_batch = to_cuda(next_batch)  # cpu -> cuda, note that DDP will move all cpu tensors to cuda as well
                    if hasattr(self.model, 'prepare_params'): self.model.prepare_params(self, next_batch)  # perform some action on the gradients
                    elif hasattr(self.model, 'module') and hasattr(self.model.module, 'prepare_params'): self.model.module.prepare_params(self, next_batch)  # perform some action on the gradients
                return next_index, next_batch

        # Get next data and start copying
        index, flying_batch = prefetch_next_batch()
        torch.cuda.current_stream().wait_stream(data_stream)  # wait for data transfer to finish for the first iteration

        data_time = 0
        max_data_time = 0
        batch_time = 0
        forward_time = 0
        backward_time = 0
        optimization_time = 0
        while flying_batch.meta.iter >= 0:  # control number of iterations explicitly

            iter = flying_batch.meta.iter.item()
            batch = flying_batch
            mem_probe = (
                self.show_cuda_mem_snapshot
                and (iter + 1) % self.log_interval == 0
                and not get_rank()
                and torch.cuda.is_available()
            )
            mem_snapshots = {}

            iter_start_time = time.perf_counter()

            # Get the context manager for the model
            # https://muellerzr.github.io/blog/gradient_accumulation.html
            # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/14
            # https://discuss.pytorch.org/t/gradient-accumulation-with-ddp-no-sync-interface/169593/3
            cast_context = torch.cuda.amp.autocast(enabled=self.train_use_amp, dtype=self.amp_dtype)
            use_no_sync = hasattr(self.model, "no_sync") and self.gradient_accumulation_steps > 1 and (iter + 1) % self.gradient_accumulation_steps != 0
            sync_context = self.model.no_sync() if use_no_sync else nullcontext()

            with sync_context:  # maybe skip gradient sync for DDP models
                # Model forwarding
                if self.debug_data_loading:
                    with cast_context:
                        with torch.no_grad():
                            output: dotdict = self.model(batch)  # random dict storing various forms of output
                        # save the inference results into files for debugging
                        # please visualize the results by running `python3 scripts/vggt/viser_visualizer.py ./debug_data_loading`
                        self.save_predictions_to_pkl(epoch, iter, batch, output, './debug_data_loading')
                        with torch.cuda.stream(data_stream):
                            index, flying_batch = next(enumerater, (None, None))
                            flying_batch = add_iter(flying_batch, begin_epoch * self.ep_iter + index, self.total_iter)  # is this bad naming
                            flying_batch = to_cuda(flying_batch)  # cpu -> cuda, note that DDP will move all cpu tensors to cuda as well
                        continue

                batch.loss_scaler = self.scaler.get_scale() / self.gradient_accumulation_steps
                batch.no_sync_context = self.model.no_sync if hasattr(self.model, "no_sync") else nullcontext
                with cast_context:  # maybe perform AMP
                    output: dotdict = self.model(batch)  # random dict storing various forms of output
                loss: torch.Tensor = output.loss.mean()  # final optimizable loss variable to backward
                image_stats: dotdict = output.image_stats  # things to report to recorder (all image tensors (float32))
                scalar_stats: dotdict = output.scalar_stats  # things to report to logger and recorder (all scalars)
                forward_end_time = time.perf_counter()
                forward_time += forward_end_time - iter_start_time
                timer.record('model forwarding')
                if mem_probe:
                    mem_snapshots["fwd"] = self._cuda_memory_snapshot()
                self._maybe_capture_cuda_mem_trace("fwd", iter)

                # Backward pass
                self.scaler.scale(loss / self.gradient_accumulation_steps).backward()  # Scale loss by accumulation steps
                backward_end_time = time.perf_counter()
                backward_time += backward_end_time - forward_end_time
                timer.record('model backwarding')
                if mem_probe:
                    mem_snapshots["bwd"] = self._cuda_memory_snapshot()
                self._maybe_capture_cuda_mem_trace("bwd", iter)

            defer_prefetch = _should_defer_epoch_boundary_prefetch(iter, yield_every)
            if defer_prefetch:
                # Keep all ranks at the same post-backward point before epoch-level
                # validation. The next epoch batch is fetched only after validation.
                data_end_time = backward_end_time
            else:
                # Get next data and start copying
                index, flying_batch = prefetch_next_batch()
                timer.record('data preparation')
                data_end_time = time.perf_counter()
            data_time += data_end_time - backward_end_time
            max_data_time = max(max_data_time, data_end_time - backward_end_time)
            timer.record('data transfer')

            # Only update weights after accumulating gradients for specified number of steps
            if (iter + 1) % self.gradient_accumulation_steps == 0:
                # Optimizer reset
                if self.retain_last_grad: self.optimizer.zero_grad(set_to_none=True)

                if self.scaler.is_enabled():
                    # Unscale once before clipping so thresholds are applied to true gradients.
                    self.scaler.unscale_(self.optimizer)
                    # GradScaler skip decisions are local by default. ZeRO step
                    # contains collectives, so every rank must make the same
                    # step/skip decision to avoid collective-order mismatch.
                    self._sync_grad_scaler_found_inf()

                # Decorate gradient
                if self.clip_grad_norm > 0: scalar_stats.grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)  # coarsed-grained gradient clipping based on norm
                if self.clip_grad_value > 0: scalar_stats.grad_value = torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad_value)  # coarsed-grained gradient clipping based on value
                if self.clipper is not None: scalar_stats.update(self.clipper(self.model))  # fine-grained gradient clipping based on norm
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
                if mem_probe:
                    mem_snapshots["opt"] = self._cuda_memory_snapshot()
                self._maybe_capture_cuda_mem_trace("opt", iter)

                # Final parameter update
                if hasattr(self.model, 'decorate_params'): self.model.decorate_params(self, batch)  # perform some action on the gradients
                elif hasattr(self.model, 'module') and hasattr(self.model.module, 'decorate_params'): self.model.module.decorate_params(self, batch)  # perform some action on the gradients
                if not get_rank() and self.recorder is not None:
                    lr_tag = f"{self.dataloader.dataset.split.name}/lr"
                    lr_step = iter // self.gradient_accumulation_steps
                    self.recorder.add_scalar(lr_tag, self.optimizer.param_groups[0]['lr'], lr_step)

                # Make sure the data transfer is finished for following calls
                data_stream.wait_stream(torch.cuda.current_stream())  # wait for update params to finish for the data stream # MARK: STREAM SYNC
                timer.record('decorating parameters')

            # Make sure the data transfer is finished for following calls
            torch.cuda.current_stream().wait_stream(data_stream)  # wait for data transfer to finish, the backward pass should be slow enough # MARK: STREAM SYNC

            if self.debugging_signal_received:
                breakpoint()
                self.debugging_signal_received = False

            # Records data and batch forwarding time
            end_time = time.perf_counter()
            batch_time += end_time - start_time
            start_time = end_time  # note that all logging and profiling time are accumuated into data_time
            if (iter + 1) % self.empty_cache_interval == 0 and (iter + 1) % self.log_interval == 0 or iter < 5:
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
                try:
                    scalar_stats.left_time_est = batch_time / self.log_interval * (self.total_iter - iter) / 3600. / 24.
                    scalar_stats.left_time_real = (time.perf_counter() - launch_time) / (index + 1) * (self.total_iter - iter) / 3600. / 24.
                except:
                    scalar_stats.left_time_est = 0.0
                    scalar_stats.left_time_real = 0.0 
                loss_total = scalar_stats.get("loss", scalar_stats.get("loss_objective", None))
                if isinstance(loss_total, (int, float)):
                    loss_msg = f"iter {iter}, loss={loss_total:.6f}"
                    loss_obj = scalar_stats.get("loss_objective", None)
                    if isinstance(loss_obj, (int, float)) and loss_obj != loss_total:
                        loss_msg += f", loss_objective={loss_obj:.6f}"
                    indexer_loss = scalar_stats.get("indexer_loss", None)
                    if isinstance(indexer_loss, (int, float)):
                        loss_msg += f", indexer_loss={indexer_loss:.6f}"
                    log(loss_msg)
                log(f"iter {iter}, max data loading time: {max_data_time:.2f}s, avg data loading time: {scalar_stats.data:.2f}s")
                log(f"iter {iter}, avg forward/backward/optimization time: {scalar_stats.forward:.2f} / {scalar_stats.backward:.2f} / {scalar_stats.optimization:.2f} s")
                log(f"iter {iter}, avg batch time: {scalar_stats.batch:.2f} s")
                log(f"iter {iter}, left time: {scalar_stats.left_time_est:.2f} days, left time real: {scalar_stats.left_time_real:.2f} days")
                data_time = 0
                max_data_time = 0
                batch_time = 0
                forward_time = 0
                backward_time = 0
                optimization_time = 0

                self.recorder.iter = iter // self.gradient_accumulation_steps  # record the actual iteration, not the accumulated one
                self.recorder.epoch = epoch
                self.recorder.update_scalar_stats(scalar_stats)
                if self.record_images_to_tb: self.recorder.update_image_stats(image_stats)  # NOTE: recording images is slow

                # For logging onto the console
                eta_seconds = self.recorder.scalar_stats.batch.global_avg * (self.total_iter - iter)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                log_stats = dotdict()
                log_stats.eta = eta_string
                log_stats.update(self.recorder.log_stats)
                warmup_total = max(0, self.eval_start_step)
                sparse_total = max(0, self.total_iter - warmup_total)
                iter_done = iter + 1

                def _format_progress(done: int, total: int, width: int = 20) -> str:
                    if total <= 0:
                        return "n/a"
                    done = max(0, min(done, total))
                    filled = int(round(done / total * width))
                    bar = "#" * filled + "-" * (width - filled)
                    return f"[{bar}] {done}/{total}"

                warmup_done = min(iter_done, warmup_total) if warmup_total > 0 else 0
                sparse_done = min(max(0, iter_done - warmup_total), sparse_total)
                log_stats.phase = "warmup" if warmup_total > 0 and iter_done <= warmup_total else "sparse"
                log_stats.warmup_prog = _format_progress(warmup_done, warmup_total)
                log_stats.sparse_prog = _format_progress(sparse_done, sparse_total)

                # Render table to screen
                if self.show_live_table:
                    display_table(log_stats)  # render dict as a table (live, console, table)
                if mem_probe and mem_snapshots:
                    for tag, snapshot in mem_snapshots.items():
                        log(
                            f"iter {iter}, cuda_mem_{tag} ({log_stats.phase}, dev {snapshot.device}): "
                            f"{self._format_cuda_snapshot(snapshot)}"
                        )

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
                # reset start time for next epoch
                start_time = time.perf_counter()
                self.model.train()
                if defer_prefetch:
                    index, flying_batch = prefetch_next_batch()
                    torch.cuda.current_stream().wait_stream(data_stream)
                    timer.record('epoch boundary data preparation')

            # Actual start of the execution
            if (iter + 1) % self.ep_iter == 0: epoch = epoch + 1

            # Do profiling if appicable
            profiler_step()  # record a step for the profiler, extracted logic
            timer.record('logging & recording')

            del loss, output, batch, image_stats, scalar_stats

    def test_generator(
        self,
        epoch: int,
        yield_every: int = 1,
        record_prefix: Optional[str] = None,
        skip_visualizer: bool = False,
    ):
        # validation for one epoch
        self.maybe_jit_model()
        self.model.train(self.test_using_train_mode)  # set the network (model) to training mode (recursive to all modules)

        if self.test_with_ddp_sharding:
            torch.distributed.barrier()

        # NOTE: 如果 val_dataloader 为空，下面的循环不会执行，iter 将不会被赋值，
        # 但在 epoch 结束的 recorder.record 日志里仍会引用它。
        iter = epoch * self.ep_iter - 1
        record_prefix = record_prefix or self.val_dataloader.dataset.split.name

        for index, batch in enumerate(self.val_dataloader):
            if self.test_with_ddp_sharding:
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                global_index = index * world_size + rank
                if self.print_test_progress:
                    log(f"(rank {rank}) Evaluating epoch {epoch}, {global_index} / {len(self.val_dataloader) * world_size}")
            else:
                global_index = index
                if self.print_test_progress:
                    log(f"(no ddp sharding) Evaluating epoch {epoch}, {global_index} / {len(self.val_dataloader)}")
            iter = epoch * self.ep_iter - 1  # some indexing trick
            batch = add_iter(batch, iter, self.total_iter)  # is this bad naming
            batch = to_cuda(batch)  # cpu -> cuda, note that DDP will move all cpu tensors to cuda as well
            
            with torch.inference_mode(self.test_using_inference_mode), torch.no_grad(), torch.cuda.amp.autocast(enabled=self.test_use_amp, dtype=self.amp_dtype, cache_enabled=self.test_amp_cached):
                output: dotdict = self.model(batch, compute_loss=True)
                scalar_stats = self.evaluator.evaluate(output, batch)
                if skip_visualizer:
                    image_stats = dotdict()
                else:
                    image_stats = self.visualizer.visualize(output, batch)

            # Optionally导出预测结果以便后处理（例如 PGO）
            self.maybe_export_predictions(batch, output, global_index)
            
            if self.debug_data_loading_for_testing:
                self.save_predictions_to_pkl(epoch, global_index, batch, output, os.environ.get("DEBUG_DIR", "./debug_data_loading"), metrics=scalar_stats)
            
            if self.test_with_ddp_sharding:
                # sync all ranks
                torch.distributed.barrier()
                log(f"(rank {get_rank()}) sync all ranks before gather objects")

                # gather all objects
                assert yield_every == -1, "yield_every must be -1 when using DDP sharding"
                obj = {
                    'rank': torch.distributed.get_rank(),
                    'global_index': global_index,
                    'scalar_stats': scalar_stats,
                    'prefix': self.val_dataloader.dataset.split.name + "_" + "FRAME",
                }
                objects = [None] * torch.distributed.get_world_size()
                objects[torch.distributed.get_rank()] = obj
                torch.distributed.all_gather_object(objects, obj)
                rank = torch.distributed.get_rank()
                if rank == 0:
                    num_valid_samples = self.val_dataloader.batch_sampler.sampler.get_num_total_valid_samples()  # TODO: fix this trivial logic
                    for obj in objects:
                        _obj_rank, _global_index, _scalar_stats, _prefix = obj['rank'], obj['global_index'], obj['scalar_stats'], obj['prefix']
                        if _global_index >= num_valid_samples:
                            log(f"(obj_rank {_obj_rank}) Skip recording {_global_index} / {num_valid_samples} due to out of range")
                            continue
                        log(f"(obj_rank {_obj_rank}) Recording {_global_index} / {num_valid_samples}")
                        self.recorder.iter = iter // self.gradient_accumulation_steps  # record the actual iteration, not the accumulated one
                        self.recorder.epoch = epoch
                        self.recorder.update_scalar_stats(_scalar_stats)
                        if _obj_rank == rank and self.record_images_to_tb:  # TODO: collect image_stats from all ranks?
                            self.recorder.update_image_stats(image_stats)
                        log(f"(obj_rank {_obj_rank}) calling self.recorder.record")
                        # self.recorder.record(_prefix, scalar_stats=_scalar_stats)
                        log(f"(obj_rank {_obj_rank}) done calling self.recorder.record")
                del objects, obj
                torch.distributed.barrier()
            else:
                self.recorder.iter = iter // self.gradient_accumulation_steps  # record the actual iteration, not the accumulated one
                self.recorder.epoch = epoch
                self.recorder.update_scalar_stats(scalar_stats)
                if self.record_images_to_tb: self.recorder.update_image_stats(image_stats)
                # self.recorder.record(self.val_dataloader.dataset.split.name + "_" + "FRAME")  # per frame records, to make things cleaner

            if yield_every > 0 and (iter + 1) % yield_every == 0:
                # break  # dataloader could be infinite
                yield output
                self.model.train(self.test_using_train_mode)

            profiler_step()  # record a step for the profiler, extracted logic
            del output, batch, scalar_stats, image_stats  # free memory

        if self.test_with_ddp_sharding:
            num_valid_samples = self.val_dataloader.batch_sampler.sampler.get_num_total_valid_samples()
            self.evaluator.synchronize_metrics(max_size=num_valid_samples)
            if hasattr(self.visualizer, "synchronize"):
                self.visualizer.synchronize()

        if not self.test_with_ddp_sharding or get_rank() == 0:
            if self.print_test_progress:
                log(f"evaluator.metrics len: {len(self.evaluator.metrics)}")
            scalar_stats = self.evaluator.summarize()
            image_stats = self.visualizer.summarize()
            self.recorder.update_scalar_stats(scalar_stats)
            if self.record_images_to_tb: self.recorder.update_image_stats(image_stats)
            if self.print_test_progress:
                log(f"(rank {get_rank()}) Test recording to tensorboard: {record_prefix}, iter {iter}, epoch {epoch}")
            self.recorder.record(record_prefix)
            if self.print_test_progress:
                log(f"(rank {get_rank()}) done recording to tensorboard")
        if self.test_with_ddp_sharding:
            torch.distributed.barrier()
        torch.cuda.empty_cache()  # for better memory recording
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()  # wait for all strange copies in the non-training pass to finish

    def train_epoch(self, epoch: int):
        train_generator = self.train_generator(epoch, self.ep_iter)
        for _ in train_generator: pass  # the actual calling

    def test_epoch(self, epoch: int, record_prefix: Optional[str] = None, skip_visualizer: bool = False):
        test_generator = self.test_generator(
            epoch,
            -1,
            record_prefix=record_prefix,
            skip_visualizer=skip_visualizer,
        )  # nevel yield (special logic)
        for _ in test_generator: pass  # the actual calling

    def _run_validation(self, epoch: int):
        if self.main_val_core4_cfg:
            distributed_core4 = bool(get_distributed() and self.test_with_ddp_sharding)
            if distributed_core4:
                torch.distributed.barrier()
            if distributed_core4 or not get_rank():
                self._run_main_val_core4(epoch, distributed_sharding=distributed_core4)
            if distributed_core4:
                torch.distributed.barrier()
            return

        if not self.main_val_cfgs:
            self.test_epoch(epoch)
            return

        self._run_main_val_suite(epoch)

    def _main_val_core4_step(self, epoch: int) -> int:
        iter_step = max(epoch * self.ep_iter - 1, 0)
        return iter_step // max(int(self.gradient_accumulation_steps), 1)

    def _record_core4_scalars(self, dataset_name: str, scalars: Dict[str, float], step: int) -> None:
        if not self.recorder:
            return
        for key, value in scalars.items():
            self.recorder.add_scalar(f"VAL/{dataset_name}/{key}", value, step)

    def _run_main_val_core4(self, epoch: int, distributed_sharding: bool = False):
        step = self._main_val_core4_step(epoch)
        if self.recorder:
            self.recorder.iter = step
            self.recorder.epoch = epoch

        aggregate = run_core4_main_val(
            model=self.model,
            raw_cfg=self.main_val_core4_cfg,
            epoch=epoch,
            global_step=max(epoch * self.ep_iter - 1, 0),
            record_dir=self.recorder.record_dir if self.recorder else os.getcwd(),
            repo_root=Path(os.getcwd()),
            distributed_sharding=distributed_sharding,
        )

        if get_rank():
            return

        for dataset_result in aggregate.get("datasets", []):
            dataset_name = dataset_result["name"]
            runtime_sec = float(dataset_result.get("runtime_sec", 0.0))
            payload = dataset_result.get("payload", {})
            scalars = extract_core4_tb_scalars(dataset_name, payload)
            self._record_core4_scalars(dataset_name, scalars, step)
            if self.recorder:
                self.recorder.add_scalar(f"VAL/{dataset_name}/runtime_sec", runtime_sec, step)
            log(green(f"Main validation done: {dataset_name}, scalars={len(scalars)}, runtime={runtime_sec:.2f}s"))

        summary_path = aggregate.get("summary_path", "")
        if summary_path:
            log(green(f"Core4 main validation summary saved to {summary_path}"))

    def _resolve_eval_cfg_path(self, cfg_path: str) -> str:
        if exists(cfg_path):
            return cfg_path
        candidate = join(os.getcwd(), cfg_path)
        if exists(candidate):
            return candidate
        return cfg_path

    def _build_eval_dataloader(self, cfg_path: str, use_ddp: bool = True, max_iter: Optional[int] = None) -> VolumetricVideoDataloader:
        cfg_path = self._resolve_eval_cfg_path(cfg_path)
        eval_cfg = Config.fromfile(cfg_path)
        if not hasattr(eval_cfg, "val_dataloader_cfg"):
            raise ValueError(f"Missing val_dataloader_cfg in {cfg_path}")

        val_cfg = dotdict(copy.deepcopy(eval_cfg.val_dataloader_cfg))
        val_cfg.type = val_cfg.get("type", "VolumetricVideoDataloader")
        val_cfg.max_iter = val_cfg.get("max_iter", -1)
        if max_iter is not None:
            val_cfg.max_iter = max_iter
        if "fix_random" in eval_cfg and "fix_random" not in val_cfg:
            val_cfg.fix_random = bool(eval_cfg.fix_random)
        if "deterministic" in eval_cfg and "deterministic" not in val_cfg:
            val_cfg.deterministic = bool(eval_cfg.deterministic)
        if "allow_tf32" in eval_cfg and "allow_tf32" not in val_cfg:
            val_cfg.allow_tf32 = bool(eval_cfg.allow_tf32)

        sampler_cfg = dotdict(val_cfg.get("sampler_cfg", dotdict()))
        use_dist_sampler = get_distributed() and use_ddp
        if use_dist_sampler:
            if sampler_cfg.get("type") == "SequentialSampler":
                log(yellow("evaluation sampler forced to DistributedSequentialSampler under DDP"))
            sampler_cfg.type = "DistributedSequentialSampler"
        else:
            sampler_cfg.type = sampler_cfg.get("type", "SequentialSampler")
        val_cfg.sampler_cfg = sampler_cfg

        batch_sampler_cfg = dotdict(val_cfg.get("batch_sampler_cfg", dotdict()))
        batch_sampler_cfg.type = batch_sampler_cfg.get("type", "ImageBasedBatchSampler")
        batch_sampler_cfg.batch_size = batch_sampler_cfg.get("batch_size", 1)
        val_cfg.batch_sampler_cfg = batch_sampler_cfg

        dataset_cfg = dotdict(val_cfg.get("dataset_cfg", dotdict()))
        dataset_cfg.split = dataset_cfg.get("split", "VAL")
        # Align extra-eval loader knobs with the main validation defaults to avoid
        # spawning heavy parallel loaders during training.
        base_val_cfg = getattr(cfg, "val_dataloader_cfg", dotdict())
        for key in ("num_workers", "prefetch_factor", "pin_memory"):
            if key in base_val_cfg:
                val_cfg[key] = base_val_cfg[key]
        base_dataset_cfg = dotdict(base_val_cfg.get("dataset_cfg", dotdict()))
        for key in ("parallel_loading", "dataloading_workers", "proc_align_size", "proc_max_size", "center_crop"):
            if key in base_dataset_cfg:
                dataset_cfg[key] = base_dataset_cfg[key]
        val_cfg.dataset_cfg = dataset_cfg

        return DATALOADERS.build(val_cfg)

    def _prepare_main_val_dataloader(self, name: str, cfg_path: str, max_iter: Optional[int] = None) -> VolumetricVideoDataloader:
        cache_key = ("main", name, cfg_path, max_iter, bool(self.test_with_ddp_sharding))
        if cache_key not in self._extra_eval_dataloaders:
            self._extra_eval_dataloaders[cache_key] = self._build_eval_dataloader(
                cfg_path,
                use_ddp=self.test_with_ddp_sharding,
                max_iter=max_iter,
            )
        return self._extra_eval_dataloaders[cache_key]

    def _prepare_extra_eval_dataloader(self, cfg_path: str) -> VolumetricVideoDataloader:
        cache_key = ("extra", cfg_path, bool(self.extra_eval_use_ddp))
        if cache_key not in self._extra_eval_dataloaders:
            self._extra_eval_dataloaders[cache_key] = self._build_eval_dataloader(
                cfg_path,
                use_ddp=self.extra_eval_use_ddp,
            )
        return self._extra_eval_dataloaders[cache_key]

    def _build_extra_evaluator(self, tag: str) -> VolumetricVideoEvaluator:
        evaluator_cfg = dotdict(copy.deepcopy(cfg.runner_cfg.evaluator_cfg))
        evaluator_cfg.metrics_file = f"metrics_{tag}.json"
        return EVALUATORS.build(evaluator_cfg)

    def _run_main_val_suite(self, epoch: int):
        base_val = getattr(self, "val_dataloader", None)
        base_eval = self.evaluator
        base_record_images = self.record_images_to_tb

        for entry in self.main_val_cfgs:
            record_prefix = f"VAL/{entry.name}"
            try:
                self.val_dataloader = self._prepare_main_val_dataloader(
                    entry.name,
                    entry.cfg,
                    max_iter=entry.max_iter,
                )
                self.evaluator = self._build_extra_evaluator(entry.name)
                log(green(f"Performing main validation: {entry.name}"))
                self.test_epoch(epoch, record_prefix=record_prefix, skip_visualizer=not self.record_images_to_tb)
                log(green(f"Main validation done: {entry.name}"))
            except Exception:
                log(red(f"Error in main validation for {entry.name}, ignored and continuing"))
                stacktrace()
                stop_prog()
                if not self.ignore_eval_error:
                    raise
            finally:
                self.val_dataloader = base_val
                self.evaluator = base_eval
                self.record_images_to_tb = base_record_images

    def _maybe_run_extra_evals(self, epoch: int):
        if not self.extra_eval_cfgs or self.extra_eval_every <= 0:
            return
        self._extra_eval_counter += 1
        if self._extra_eval_counter % self.extra_eval_every != 0:
            return

        base_val = self.val_dataloader
        base_eval = self.evaluator
        base_record_images = self.record_images_to_tb

        for cfg_path in self.extra_eval_cfgs:
            tag = os.path.splitext(os.path.basename(cfg_path))[0]
            record_prefix = f"{self.extra_eval_prefix}/{tag}" if self.extra_eval_prefix else tag
            try:
                self.val_dataloader = self._prepare_extra_eval_dataloader(cfg_path)
                self.evaluator = self._build_extra_evaluator(tag)
                self.record_images_to_tb = self.extra_eval_record_images
                log(green(f"Performing extra evaluation: {tag}"))
                self.test_epoch(epoch, record_prefix=record_prefix, skip_visualizer=not self.extra_eval_record_images)
                log(green(f"Extra evaluation done: {tag}"))
            except Exception:
                log(red(f"Error in extra evaluation for {tag}, ignored and continuing"))
                stacktrace()
                stop_prog()
                if not self.ignore_eval_error:
                    raise
            finally:
                self.val_dataloader = base_val
                self.evaluator = base_eval
                self.record_images_to_tb = base_record_images

    def maybe_export_predictions(self, batch, output, global_index: int):
        """导出预测结果供后处理使用，默认仅在主进程启用"""
        if not self.export_predictions or get_rank():
            return
        try:
            import numpy as np
        except ImportError:
            log(yellow("numpy 未安装，跳过预测导出"))
            return

        def _to_numpy(t):
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().numpy()
            return t

        # 构建保存路径
        save_dir = os.path.join(self.export_dir, f"batch_{global_index:06d}")
        os.makedirs(save_dir, exist_ok=True)

        # 尝试收集帧名
        names = None
        meta = batch.get('meta', None) if hasattr(batch, 'get') else None
        if meta is not None:
            names = getattr(meta, 'name', None) or (meta.get('name') if hasattr(meta, 'get') else None)
        if names is not None:
            np.save(os.path.join(save_dir, "names.npy"), np.array(names))

        # 导出常见预测张量
        for key in ['cam', 'pose', 'dpt', 'xyz', 'tra', 'dpt_conf', 'xyz_conf']:
            val = output.get(key) if hasattr(output, 'get') else None
            if val is None:
                continue
            try:
                np.save(os.path.join(save_dir, f"{key}.npy"), _to_numpy(val))
            except Exception as e:
                log(yellow(f"导出 {key} 失败，已跳过: {e}"))

        # 额外记录标量指标
        scalar_path = os.path.join(save_dir, "metrics.yaml")
        try:
            import yaml
            yaml.safe_dump({k: float(v) if isinstance(v, (int, float)) else v for k, v in output.get('scalar_stats', {}).items()}, open(scalar_path, "w"))
        except Exception:
            pass
