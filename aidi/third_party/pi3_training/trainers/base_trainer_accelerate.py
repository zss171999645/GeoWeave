import argparse
import datetime
import itertools
import os
import random
import traceback
from accelerate import Accelerator
import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import yaml
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from copy import deepcopy
from easydict import EasyDict
import time
import json
import math
import sys
from PIL import Image
import shutil
from pathlib import Path
from utils.basic import seed_anything, count_parameters

from datasets import create_dataloader
# from model.network import Network
from utils.misc import get_logger, is_logging_process, pretty_print_hydra_config, move_to_device, get_rank
from utils.basic import seed_anything
from utils.vggt_validation import VggtStylePi3MetricAccumulator
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
from utils.dist import (
    MetricLogger,
    SmoothedValue,
    init_distributed_mode,
    setup_for_distributed,
)
from accelerate import DistributedDataParallelKwargs
from transformers.trainer_pt_utils import get_model_param_count
from accelerate import (
    DistributedType,
)
from accelerate.utils import (
    DataLoaderConfiguration,
    DynamoBackend,
    GradientAccumulationPlugin,
    ProjectConfiguration,
    TorchDynamoPlugin,
    set_seed,
)
import numpy as np

class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self._tb_writer = None
        self._direct_tensorboard = False

        with open_dict(cfg):
            cfg.job_logging_cfg = HydraConfig.get().job_logging

        # random seed
        if cfg.random_seed is None:
            cfg.random_seed = random.randint(1, 10000)
        seed_anything(cfg.random_seed, deterministic=False)              # deterministic=True for reproduction
        self.eval_only = self._truthy_config_value(cfg.get("eval_only", False))

        ## 1. Build accelerator
        self.build_accelerator()

        if is_logging_process():
            pretty_print_hydra_config(cfg)

        ## 2. Prepare model
        self.log_info("Preparing model...")
        self.model = self.prepare_model()
        self.n_learnable_parameters = get_model_param_count(
            self.model, trainable_only=True
        )
        self.n_fix_parameters = get_model_param_count(
            self.model, trainable_only=False
        )
        self.accelerator.wait_for_everyone()

        ## 3. Prepare dataloader
        self.log_info("Making train dataloader...")
        self.train_loader = create_dataloader(cfg, 'train')
        self.validation_enabled = self.cfg.test.iters_per_test != 0
        if self.validation_enabled:
            self.log_info("Making test dataloader...")
            self.test_loader = create_dataloader(cfg, 'test')
        else:
            self.log_info("Validation disabled by test.iters_per_test=0. Skipping test dataloader.")
            self.test_loader = None
        self.accelerator.wait_for_everyone()
        self._log_memory_snapshot("after_dataloader_build")

        ## 5. Prepare optimizer and scheduler (fsdp should after preparing the model using accelerate)
        if self.cfg.get("fsdp_plugin"):
            self.model = self.accelerator.prepare(self.model)
            self.accelerator.wait_for_everyone()

            self.optimizer = self.build_optimizer(self.cfg.train.optimizer, self.model)
            self.log_info(f"optimizer: {self.optimizer}")
        else:
            self.optimizer = self.build_optimizer(self.cfg.train.optimizer, self.model)
            self.log_info(f"optimizer: {self.optimizer}")

            self.model = self.accelerator.prepare(self.model)
            self.accelerator.wait_for_everyone()

        # Create the LR scheduler
        self.iters_per_epoch = self.cfg.train.iters_per_epoch if self.cfg.train.iters_per_epoch > 0 else len(self.train_loader)
        if self.validation_enabled:
            self.iters_per_test = self.cfg.test.iters_per_test if self.cfg.test.iters_per_test > 0 else len(self.test_loader)
        else:
            self.iters_per_test = 0
        self.cfg.train.lr_scheduler.total_steps = self.cfg.train.num_epoch * self.iters_per_epoch
        self.log_info(f"Total step for lr scheduler: {self.cfg.train.lr_scheduler.total_steps} ({self.cfg.train.num_epoch} * {self.iters_per_epoch})")
        self.lr_scheduler = build_scheduler(
            self.cfg.train.lr_scheduler, optimizer=self.optimizer
        )
        self.log_info(f"LRScheduler: {self.lr_scheduler}")

        ## 6. Prepare accelerate training
        self.prepare_training()
        self._log_memory_snapshot("after_prepare_training")

    @staticmethod
    def _grad_norm_to_float(grad_norm):
        if grad_norm is None:
            return 0.0
        if hasattr(grad_norm, "item"):
            return float(grad_norm.item())
        return float(grad_norm)

    @staticmethod
    def _combine_grad_norms(*grad_norms):
        return math.sqrt(sum(float(grad_norm) ** 2 for grad_norm in grad_norms))

    def build_optimizer(self, cfg_optimizer, model, param_group_fn=None):
        return build_optimizer(cfg_optimizer, model, param_group_fn=param_group_fn)

    def prepare_training(self):
        # report model details
        self.log_info(
            f"total number of learnable params: {self.n_learnable_parameters / 1e6} M"
        )
        self.log_info(
            f"total number of fixed params: {self.n_fix_parameters / 1e6} M"
        )

        # Wrap the model, optmizer, and scheduler with accelerate
        self.log_info("before accelerator.prepare")

        # (
        #     self.model,
        #     self.train_loader,
        #     self.test_loader,
        #     self.optimizer,
        #     self.lr_scheduler,
        # ) = self.accelerator.prepare(
        #     self.model, self.train_loader, self.test_loader, self.optimizer, self.lr_scheduler
        # )

        # don't wrap dataloader
        (
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.optimizer, self.lr_scheduler
        )

        if self.accelerator.is_main_process:
            if self._direct_tensorboard:
                self._init_direct_tensorboard()
            elif self.cfg.log.use_wandb or self.cfg.log.use_tensorboard:
                self.accelerator.init_trackers(os.path.basename(self.cfg.log.output_dir))

        # Report the training info
        self.total_batch_size = (
            self.cfg.train.batch_size
            * self.accelerator.num_processes
            * self.cfg.train.gradient_accumulation_steps
        )
        self.log_info("***** Running training *****")
        self.log_info(f"LR = {self.cfg.train.optimizer.lr:.8f}")
        self.log_info(f"Weigth Decay = {self.cfg.train.optimizer.weight_decay:.8f}")
        self.log_info(f"Instantaneous batch size per device = {self.cfg.train.batch_size}")
        self.log_info(f"Total Batch size = {self.total_batch_size}")
        self.log_info(
            f"Gradient Accumulation steps = {self.accelerator.gradient_accumulation_steps}"
        )
        self.log_info(f"Number of epochs = {self.cfg.train.num_epoch}")
        self.log_info(
            f"Number of training steps per epoch = {self.iters_per_epoch}"
        )
        self.log_info(
            f"Number of total training steps = {self.iters_per_epoch * self.cfg.train.num_epoch}"
        )
        # self.log_info(f"Number of training examples per epoch = {len(self.dataloader.dataset)}")
        self.log_info(
            f"Number of model parameters = {self.n_fix_parameters / 1e6:.2f}M"
        )
        self.log_info(
            f"Number of model trainable parameters = {self.n_learnable_parameters / 1e6:.2f}M"
        )

        # Auto resume the checkpoint
        latest_epoch = self.auto_resume()
        self.initial_global_step = self.iters_per_epoch * latest_epoch
        self.first_epoch = latest_epoch

        os.makedirs(self.cfg.log.ckpt_dir, exist_ok=True)

    def prepare_model(self):
        model = hydra.utils.instantiate(self.cfg.model)
        count_parameters(model)
        return model
    
    def before_epoch(self, epoch):
        pass

    def train(self):
        # Start Train!
        start_time = time.time()
        self.accelerator.wait_for_everyone()
        self._log_memory_snapshot("train_start")

        # Initialize variable to track the best validation metric
        best_val_metric = float('inf')  # For metrics like loss; use -float('inf') for accuracy
        best_model_path = None

        max_checkpoints = self.cfg.log.max_checkpoints  # Maximum number of recent checkpoints to keep
        saved_checkpoints = []  # List to track saved checkpoint paths

        if self.core4_main_val_first_eval_enabled():
            first_core4_stats = self.run_core4_main_val(-1)
            if first_core4_stats:
                log_stats = {
                    **{f"val_{k}": v for k, v in first_core4_stats.items()},
                    "epoch": -1,
                    "n_parameters": self.n_learnable_parameters,
                }
                if self.accelerator.is_main_process:
                    with open(
                        os.path.join(self.cfg.log.ckpt_dir, "log.txt"),
                        mode="a",
                        encoding="utf-8",
                    ) as f:
                        f.write(json.dumps(log_stats) + "\n")
                    self.log_all(log_stats, step=0)
            self.accelerator.wait_for_everyone()

        if self.native_validation_first_eval_enabled():
            self.run_native_validation_before_first_epoch()
            self.accelerator.wait_for_everyone()
        elif self.eval_only:
            if not self.validation_enabled:
                raise ValueError("eval_only requires test.iters_per_test != 0")
            self.run_native_validation_before_first_epoch()
            self.accelerator.wait_for_everyone()

        if self.eval_only:
            self.log_info("Eval-only mode finished after native validation.")
            return

        for epoch in range(self.first_epoch, self.cfg.train.num_epoch):
            torch.cuda.reset_peak_memory_stats()

            self.before_epoch(epoch)

            train_stats = self.train_one_epoch(epoch)

            if self.native_validation_epoch_enabled(epoch):
                # Perform validation at the end of each epoch
                val_stats = self.validate(epoch)
                if self.accelerator.is_main_process:
                    self.log_all(val_stats, step=self.global_step, prefix='val')

                current_val_metric = val_stats.get("loss", float('inf'))  # Replace "val_loss" with your metric key
                if current_val_metric < best_val_metric:
                    best_val_metric = current_val_metric
                    best_model_path = os.path.join(
                        self.cfg.log.ckpt_dir,
                        "best_model",
                    )
                    saved = self._save_checkpoint_path(
                        best_model_path,
                        self.cfg.log.get("best_model_save_mode", "full_state"),
                        f"best_model_epoch_{epoch}",
                    )
                    if saved:
                        self.log_info(f"Saved best model at epoch {epoch} with val_metric: {best_val_metric:.4f}")
            else:
                val_stats = {}

            core4_stats = self.run_core4_main_val(epoch)
            if core4_stats:
                val_stats.update(core4_stats)

            self.accelerator.wait_for_everyone()

            if (
                epoch + 1
            ) % self.cfg.log.ckpt_interval == 0 or epoch + 1 == self.cfg.train.num_epoch:
                if self.accelerator.sync_gradients:
                    self.global_step = self.iters_per_epoch * (epoch + 1)
                    save_path = os.path.join(
                        self.cfg.log.ckpt_dir,
                        f"checkpoint_{epoch}",
                    )
                    saved = self._save_checkpoint_path(
                        save_path,
                        self.cfg.log.get("checkpoint_save_mode", "full_state"),
                        f"checkpoint_epoch_{epoch}",
                    )
                    if saved:
                        self.log_info(
                            f"Saved state for global step {self.global_step}"
                        )

                    # Manage saved checkpoints
                    if saved:
                        saved_checkpoints.append(save_path)
                    if self.accelerator.is_main_process and len(saved_checkpoints) > max_checkpoints:
                        oldest_checkpoint = saved_checkpoints.pop(0)
                        if os.path.exists(oldest_checkpoint):
                            shutil.rmtree(oldest_checkpoint)
                            self.log_info(f"Removed old checkpoint: {oldest_checkpoint}")

                self.accelerator.wait_for_everyone()

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
                "n_parameters": self.n_learnable_parameters,
            }

            if self.accelerator.is_main_process:
                with open(
                    os.path.join(self.cfg.log.ckpt_dir, "log.txt"),
                    mode="a",
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(log_stats) + "\n")

                self.log_all(log_stats, step=self.global_step)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.log_info("Training time {}".format(total_time_str))

        if self._tb_writer is not None:
            self._tb_writer.close()

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    @staticmethod
    def _read_int_file(path):
        try:
            text = Path(path).read_text().strip()
        except OSError:
            return None
        if text == "max":
            return None
        try:
            value = int(text)
        except ValueError:
            return None
        if value <= 0 or value >= 1 << 60:
            return None
        return value

    @classmethod
    def _read_cgroup_memory_current_bytes(cls):
        for path in (
            "/sys/fs/cgroup/memory.current",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ):
            value = cls._read_int_file(path)
            if value is not None:
                return value
        return None

    @classmethod
    def _read_cgroup_memory_limit_bytes(cls):
        for path in (
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ):
            value = cls._read_int_file(path)
            if value is not None:
                return value
        return None

    @staticmethod
    def _read_rss_bytes():
        return BaseTrainer._read_proc_rss_bytes(os.getpid())

    @staticmethod
    def _read_proc_rss_bytes(pid):
        try:
            with open(f"/proc/{int(pid)}/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) * 1024
        except (OSError, ValueError):
            return None
        return None

    @staticmethod
    def _read_proc_children(pid):
        children = set()
        task_root = Path(f"/proc/{int(pid)}/task")
        try:
            task_dirs = list(task_root.iterdir())
        except (OSError, ValueError):
            return children
        for task_dir in task_dirs:
            children_path = task_dir / "children"
            try:
                text = children_path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            for child in text.split():
                try:
                    children.add(int(child))
                except ValueError:
                    continue
        return children

    @classmethod
    def _read_process_tree_rss_bytes(cls, root_pid=None):
        root_pid = int(root_pid or os.getpid())
        visited = set()
        queue = [root_pid]
        total_rss = 0
        process_count = 0
        while queue:
            pid = queue.pop()
            if pid in visited:
                continue
            visited.add(pid)
            rss = cls._read_proc_rss_bytes(pid)
            if rss is None:
                continue
            total_rss += rss
            process_count += 1
            queue.extend(child for child in cls._read_proc_children(pid) if child not in visited)
        return total_rss, process_count

    @staticmethod
    def _parse_memory_stat_text(text):
        stats = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                value = int(parts[1])
            except ValueError:
                continue
            if value >= 0:
                stats[parts[0]] = value
        return stats

    @classmethod
    def _read_cgroup_memory_stat_bytes(cls):
        for path in (
            "/sys/fs/cgroup/memory.stat",
            "/sys/fs/cgroup/memory/memory.stat",
        ):
            try:
                text = Path(path).read_text(encoding="utf-8")
            except OSError:
                continue
            return cls._parse_memory_stat_text(text)
        return {}

    @staticmethod
    def _bytes_to_gib(value):
        if value is None or value < 0:
            return "unknown"
        return f"{value / (1024 ** 3):.2f}GiB"

    @staticmethod
    def _memory_stat_value(memory_stat, *names):
        for name in names:
            value = memory_stat.get(name)
            if value is not None and value >= 0:
                return value
        return None

    @classmethod
    def _compute_cgroup_working_set_bytes(cls, current_bytes, memory_stat):
        if current_bytes is None:
            return None
        inactive_file = cls._memory_stat_value(memory_stat, "inactive_file", "total_inactive_file")
        if inactive_file is None:
            return current_bytes
        return max(0, int(current_bytes) - int(inactive_file))

    @staticmethod
    def _normalize_memory_abort_metric(value):
        metric = str(value or "raw").strip().lower().replace("-", "_")
        aliases = {
            "": "raw",
            "current": "raw",
            "cgroup": "raw",
            "cgroup_current": "raw",
            "usage": "raw",
            "working": "working_set",
            "workingset": "working_set",
            "non_reclaimable": "working_set",
            "nonreclaimable": "working_set",
        }
        metric = aliases.get(metric, metric)
        if metric not in {"raw", "working_set"}:
            raise ValueError(
                f"Unsupported memory_abort_metric {value!r}; expected raw or working_set."
            )
        return metric

    def _log_memory_snapshot(self, label, check_abort=True):
        report_enabled = self._truthy_config_value(self.cfg.log.get("memory_report", False))
        try:
            memory_abort_fraction = float(self.cfg.log.get("memory_abort_fraction", 0.0) or 0.0)
        except (TypeError, ValueError):
            memory_abort_fraction = 0.0
        memory_abort_metric = self._normalize_memory_abort_metric(
            self.cfg.log.get("memory_abort_metric", "raw")
        )

        if not report_enabled and not (check_abort and memory_abort_fraction > 0):
            return

        rss_bytes = self._read_rss_bytes()
        tree_rss_bytes, tree_process_count = self._read_process_tree_rss_bytes()
        current_bytes = self._read_cgroup_memory_current_bytes()
        limit_bytes = self._read_cgroup_memory_limit_bytes()
        memory_stat = self._read_cgroup_memory_stat_bytes()
        working_set_bytes = self._compute_cgroup_working_set_bytes(current_bytes, memory_stat)
        ratio = -1.0
        if current_bytes is not None and limit_bytes is not None and limit_bytes > 0:
            ratio = float(current_bytes) / float(limit_bytes)
        working_set_ratio = -1.0
        if working_set_bytes is not None and limit_bytes is not None and limit_bytes > 0:
            working_set_ratio = float(working_set_bytes) / float(limit_bytes)

        def stat_value(name):
            return float(memory_stat.get(name, -1))

        local_stats = torch.tensor(
            [
                float(self.accelerator.process_index),
                float(rss_bytes if rss_bytes is not None else -1),
                float(tree_rss_bytes if tree_rss_bytes is not None else -1),
                float(tree_process_count),
                float(current_bytes if current_bytes is not None else -1),
                float(limit_bytes if limit_bytes is not None else -1),
                ratio,
                float(working_set_bytes if working_set_bytes is not None else -1),
                working_set_ratio,
                stat_value("anon"),
                stat_value("file"),
                stat_value("kernel"),
                stat_value("kernel_stack"),
                stat_value("pagetables"),
                stat_value("slab"),
                stat_value("shmem"),
                stat_value("inactive_file"),
                stat_value("active_file"),
            ],
            dtype=torch.float64,
            device=self.accelerator.device,
        )

        gathered = local_stats.reshape(1, -1)
        try:
            gathered = self.accelerator.gather(local_stats).reshape(-1, local_stats.numel())
        except Exception as exc:
            if self.accelerator.is_main_process:
                self.log_info(f"Memory snapshot {label}: failed to gather process stats: {exc}")

        if report_enabled and self.accelerator.is_main_process:
            rows = gathered.detach().cpu().tolist()
            max_rss_row = max(rows, key=lambda row: row[1])
            max_tree_rss_row = max(rows, key=lambda row: row[2])
            if any(row[6] >= 0 for row in rows):
                max_ratio_row = max(rows, key=lambda row: row[6])
                ratio_text = f"{max_ratio_row[6]:.3f}"
                cgroup_text = self._bytes_to_gib(max_ratio_row[4])
                limit_text = self._bytes_to_gib(max_ratio_row[5])
                ratio_rank = int(max_ratio_row[0])
            else:
                max_ratio_row = None
                ratio_text = "unknown"
                cgroup_text = "unknown"
                limit_text = "unknown"
                ratio_rank = -1
            if max_ratio_row is not None:
                stat_text = (
                    f"anon={self._bytes_to_gib(max_ratio_row[9])} "
                    f"file={self._bytes_to_gib(max_ratio_row[10])} "
                    f"kernel={self._bytes_to_gib(max_ratio_row[11])} "
                    f"kernel_stack={self._bytes_to_gib(max_ratio_row[12])} "
                    f"pagetables={self._bytes_to_gib(max_ratio_row[13])} "
                    f"slab={self._bytes_to_gib(max_ratio_row[14])} "
                    f"shmem={self._bytes_to_gib(max_ratio_row[15])} "
                    f"inactive_file={self._bytes_to_gib(max_ratio_row[16])} "
                    f"active_file={self._bytes_to_gib(max_ratio_row[17])}"
                )
            else:
                stat_text = "anon=unknown file=unknown kernel=unknown slab=unknown"
            if any(row[8] >= 0 for row in rows):
                max_working_set_row = max(rows, key=lambda row: row[8])
                working_set_text = self._bytes_to_gib(max_working_set_row[7])
                working_set_ratio_text = f"{max_working_set_row[8]:.3f}"
                working_set_rank = int(max_working_set_row[0])
            else:
                working_set_text = "unknown"
                working_set_ratio_text = "unknown"
                working_set_rank = -1
            self.log_info(
                f"Memory snapshot {label}: "
                f"max_rss_rank={int(max_rss_row[0])} max_rss={self._bytes_to_gib(max_rss_row[1])}, "
                f"max_tree_rss_rank={int(max_tree_rss_row[0])} "
                f"max_tree_rss={self._bytes_to_gib(max_tree_rss_row[2])} "
                f"tree_processes={int(max_tree_rss_row[3])}, "
                f"max_cgroup_rank={ratio_rank} cgroup_current={cgroup_text} "
                f"cgroup_limit={limit_text} cgroup_ratio={ratio_text} "
                f"max_working_set_rank={working_set_rank} working_set={working_set_text} "
                f"working_set_ratio={working_set_ratio_text} "
                f"memory_abort_metric={memory_abort_metric} {stat_text}"
            )

        rows = gathered.detach().cpu().tolist()
        if any(row[6] >= 0 for row in rows):
            max_ratio_row = max(rows, key=lambda row: row[6])
            max_ratio = max_ratio_row[6]
            max_ratio_rank = int(max_ratio_row[0])
            max_ratio_current = max_ratio_row[4]
            max_ratio_limit = max_ratio_row[5]
        else:
            max_ratio = ratio
            max_ratio_rank = int(self.accelerator.process_index)
            max_ratio_current = current_bytes if current_bytes is not None else -1
            max_ratio_limit = limit_bytes if limit_bytes is not None else -1

        if memory_abort_metric == "working_set" and any(row[8] >= 0 for row in rows):
            max_abort_row = max(rows, key=lambda row: row[8])
            max_abort_ratio = max_abort_row[8]
            max_abort_rank = int(max_abort_row[0])
            max_abort_current = max_abort_row[7]
            max_abort_limit = max_abort_row[5]
        else:
            max_abort_ratio = max_ratio
            max_abort_rank = max_ratio_rank
            max_abort_current = max_ratio_current
            max_abort_limit = max_ratio_limit

        local_abort = int(
            check_abort
            and memory_abort_fraction > 0
            and max_abort_ratio >= memory_abort_fraction
        )
        abort_tensor = torch.tensor([local_abort], dtype=torch.int32, device=self.accelerator.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(abort_tensor, op=dist.ReduceOp.MAX)

        if int(abort_tensor.item()) > 0:
            raise RuntimeError(
                f"Memory guard abort at {label}: {memory_abort_metric} memory ratio reached "
                f"{max_abort_ratio:.3f} on rank {max_abort_rank} "
                f"({self._bytes_to_gib(max_abort_current)}/{self._bytes_to_gib(max_abort_limit)}), "
                f"local_raw_ratio={ratio:.3f}, local_working_set_ratio={working_set_ratio:.3f}, "
                f"memory_abort_fraction={memory_abort_fraction:.3f}. "
                "This fails before the platform OOM killer can terminate a rank."
            )

    @staticmethod
    def _normalize_checkpoint_save_mode(value, default="full_state"):
        mode = str(value or default).strip().lower()
        aliases = {
            "false": "none",
            "0": "none",
            "off": "none",
            "skip": "none",
            "model": "model_only",
            "weights": "model_only",
            "weights_only": "model_only",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"full_state", "model_only", "none"}:
            raise ValueError(
                f"Unsupported checkpoint save mode {value!r}; expected full_state, model_only, or none."
            )
        return mode

    def _save_model_only(self, save_path, label):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)
            state_dict = self.accelerator.unwrap_model(self.model).state_dict()
            self.accelerator.save(state_dict, os.path.join(save_path, "pytorch_model.bin"))
            with open(os.path.join(save_path, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "save_mode": "model_only",
                        "label": label,
                        "global_step": int(getattr(self, "global_step", 0)),
                    },
                    f,
                    indent=2,
                )
            del state_dict
        self.accelerator.wait_for_everyone()

    def _save_checkpoint_path(self, save_path, mode_value, label):
        mode = self._normalize_checkpoint_save_mode(mode_value)
        if mode == "none":
            self.log_info(f"Skip checkpoint save for {label}: save mode is none")
            self.accelerator.wait_for_everyone()
            return False

        self._log_memory_snapshot(f"before_{label}_{mode}")
        if mode == "full_state":
            self.accelerator.save_state(save_path, safe_serialization=False)
        elif mode == "model_only":
            self._save_model_only(save_path, label)
        self._log_memory_snapshot(f"after_{label}_{mode}", check_abort=False)
        return True

    @staticmethod
    def _truthy_config_value(value):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def core4_main_val_cfg(self):
        if hasattr(self.cfg, "get"):
            cfg = self.cfg.get("main_val_core4_cfg", None)
        else:
            cfg = getattr(self.cfg, "main_val_core4_cfg", None)
        if cfg is None:
            return None
        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
        return cfg

    def core4_main_val_first_eval_enabled(self):
        cfg = self.core4_main_val_cfg()
        if not cfg:
            return False
        if hasattr(cfg, "get"):
            value = cfg.get("run_first_eval", False)
        else:
            value = getattr(cfg, "run_first_eval", False)
        return self._truthy_config_value(value)

    def native_validation_first_eval_enabled(self):
        if not self.validation_enabled:
            return False
        return self._truthy_config_value(self.cfg.test.get("before_first_epoch", False))

    def native_validation_epoch_enabled(self, epoch):
        if not self.validation_enabled:
            return False
        interval = int(self.cfg.test.get("eval_interval", 1))
        if interval <= 0:
            return False
        return (epoch + 1) % interval == 0

    def vggt_style_metrics_cfg(self):
        cfg = self.cfg.test.get("vggt_style_metrics", None)
        if cfg is None:
            return {"enabled": False}
        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
        return cfg

    def run_native_validation_before_first_epoch(self):
        self.before_epoch(0)
        val_stats = self.validate(-1)
        log_stats = {
            **{f"val_{k}": v for k, v in val_stats.items()},
            "epoch": -1,
            "n_parameters": self.n_learnable_parameters,
        }
        if self.accelerator.is_main_process:
            with open(
                os.path.join(self.cfg.log.ckpt_dir, "log.txt"),
                mode="a",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(log_stats) + "\n")
            self.log_all(val_stats, step=0, prefix='val')
            self.log_all(log_stats, step=0)
        return val_stats

    def core4_main_val_enabled(self):
        cfg = self.core4_main_val_cfg()
        if not cfg:
            return False
        datasets = cfg.get("datasets", {})
        return any(
            self._truthy_config_value(dataset_cfg.get("enabled", False))
            for dataset_cfg in datasets.values()
            if hasattr(dataset_cfg, "get")
        )

    def run_core4_main_val(self, epoch):
        cfg = self.core4_main_val_cfg()
        if not self.core4_main_val_enabled():
            return {}

        from aidi.utils.core4_main_val import extract_core4_tb_scalars, run_core4_main_val

        repo_root = Path(__file__).resolve().parents[4]
        global_step = int(getattr(self, "global_step", self.iters_per_epoch * (epoch + 1)))
        record_dir = str(self.cfg.log.output_dir)

        self.accelerator.wait_for_everyone()
        self.log_info(f"Start core4 main validation for epoch {epoch}")
        aggregate = run_core4_main_val(
            model=self.model,
            raw_cfg=cfg,
            epoch=epoch,
            global_step=global_step,
            record_dir=record_dir,
            repo_root=repo_root,
            distributed_sharding=self.accelerator.num_processes > 1,
        )

        stats = {}
        if self.accelerator.is_main_process:
            for dataset in aggregate.get("datasets", []):
                dataset_name = str(dataset.get("name", "unknown"))
                for key, value in extract_core4_tb_scalars(dataset_name, dataset.get("payload", {})).items():
                    stats[f"core4/{dataset_name}/{key}"] = value
            if aggregate.get("summary_path"):
                self.log_info(f"Core4 main validation summary: {aggregate['summary_path']}")
        self.accelerator.wait_for_everyone()
        return stats

    def validate(self, epoch):
        max_test_iters = int(self.cfg.test.get("iters_per_test", -1))
        if max_test_iters == 0:
            self.log_info(f"Skip validation for epoch {epoch} because test.iters_per_test=0")
            return {"loss": 0.0}

        self.model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        header = f"Validation Epoch: [{epoch}]"

        val_loss = 0.0
        total_samples = 0
        vggt_metric_accumulator = VggtStylePi3MetricAccumulator(self.vggt_style_metrics_cfg())

        self.log_info(f"Start validation for epoch {epoch}")
        self._log_memory_snapshot(f"before_validation_epoch_{epoch}")
        val_log_max_iters = max_test_iters if max_test_iters > 0 else None
        with torch.no_grad():
            for it, batch in enumerate(metric_logger.log_every(
                self.test_loader,
                self.cfg.test.print_freq,
                header,
                max_iters=val_log_max_iters,
            )):
                batch = move_to_device(batch, self.accelerator.device)

                # Forward pass
                forward_outputs = self.forward_batch(batch, mode='test')
                vggt_batch_metrics = vggt_metric_accumulator.compute_batch_metrics(
                    forward_outputs[0] if isinstance(forward_outputs, (list, tuple)) else forward_outputs,
                    batch,
                )
                outputs = self.calculate_loss(forward_outputs, batch, mode='test')
                vggt_metric_accumulator.update(
                    vggt_metric_accumulator.add_training_metrics(vggt_batch_metrics, outputs)
                )
                loss = outputs.loss

                # Gather statistics
                loss_value = loss.item()
                val_loss += loss_value * len(batch)
                total_samples += len(batch)

                # self.log_all(outputs, self.global_step, prefix='val')

                metric_logger.update(**outputs)

        # Average the validation loss
        val_loss /= max(total_samples, 1)

        # Gather the stats from all processes
        metric_logger.synchronize_between_processes()
        self.log_info(f"Validation results: {metric_logger}")

        val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        val_stats.update(vggt_metric_accumulator.summarize())
        self._log_memory_snapshot(f"after_validation_epoch_{epoch}")
        return val_stats

    def train_one_epoch(self, epoch):
        self.model.train()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter(
            "min_lr", SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
        # metric_logger.add_meter(
        #     "dataloader", SmoothedValue(window_size=1, fmt="{value:.6f}")
        # )
        header = "Epoch: [{}]".format(epoch)
        loss_details_dict = {}
        start_steps = epoch * self.iters_per_epoch
        self.global_step = start_steps

        self.log_info(
            "Start training epoch {}, {} iters per inner epoch. Training dtype {}".format(
                epoch, self.iters_per_epoch, self.cfg.train.model_dtype
            )
        )

        for it, batch in enumerate(metric_logger.log_every(
            self.train_loader,
            self.cfg.train.print_freq,
            header,
            max_iters=self.iters_per_epoch,
        )):
            with self.accelerator.accumulate(self.model):
                # Perform the forward using the accerlate
                batch = move_to_device(batch, device=self.accelerator.device)
                with self.accelerator.autocast():
                    forward_output = self.forward_batch(batch, mode='train')
                batch_output = self.calculate_loss(forward_output, batch, mode='train')
                loss_items = self._loss_items(batch_output.loss)
                loss = self._sum_loss_items(loss_items)
                if loss > self.cfg.train.clip_loss:
                    loss_items = [loss_item * 0.0 for loss_item in loss_items]
                    loss = self._sum_loss_items(loss_items)

                # Check if the loss is nan
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    rank = get_rank()
                    print(
                        f"Rank {rank}: Loss is {loss_value}, stopping training at iter {it} (epoch {epoch}, global step {self.global_step}).",
                        force=True,
                    )
                    sys.exit(1)

                self._backward_loss_items(loss_items)
                batch_output.loss = loss.detach()

                for item in batch_output:
                    if 'loss' in item:
                        batch_output[item] = self.accelerator.gather(batch_output[item]).mean().item()
                        if item in loss_details_dict:
                            loss_details_dict[item] += batch_output[item] / self.cfg.train.gradient_accumulation_steps if loss_value != 0 else 0.0
                        else:
                            loss_details_dict[item] = batch_output[item] / self.cfg.train.gradient_accumulation_steps if loss_value != 0 else 0.0

                # clip the gradient
                if self.accelerator.sync_gradients:
                    def get_gradient_norm(parameters):
                        norm = 0
                        for param in parameters:
                            if param.grad is None:
                                continue
                            local_norm = param.grad.detach().data.norm(2)
                            norm += local_norm.item() ** 2
                        norm = norm**0.5
                        return norm

                    grad_group_norms = {}
                    separate_indexer_clip = bool(self.cfg.train.get("clip_grad_separate_indexer", False))
                    if separate_indexer_clip:
                        indexer_params = []
                        non_indexer_params = []
                        for group in self.optimizer.param_groups:
                            group_params = list(group.get("params", []))
                            if group.get("is_indexer", False):
                                indexer_params.extend(group_params)
                            else:
                                non_indexer_params.extend(group_params)

                        if non_indexer_params:
                            non_indexer_grad_norm_pre_clip = self._grad_norm_to_float(
                                self.accelerator.clip_grad_norm_(
                                    non_indexer_params, self.cfg.train.clip_grad
                                )
                            )
                            grad_group_norms["grad_norm_non_indexer"] = non_indexer_grad_norm_pre_clip
                        else:
                            non_indexer_grad_norm_pre_clip = 0.0
                        if indexer_params:
                            indexer_clip_grad = self.cfg.train.get("clip_indexer_grad", None)
                            if indexer_clip_grad is None:
                                indexer_clip_grad = self.cfg.train.clip_grad
                            indexer_grad_norm_pre_clip = self._grad_norm_to_float(
                                self.accelerator.clip_grad_norm_(
                                    indexer_params, float(indexer_clip_grad)
                                )
                            )
                            grad_group_norms["grad_norm_indexer"] = indexer_grad_norm_pre_clip
                        else:
                            indexer_grad_norm_pre_clip = 0.0
                        grad_norm_pre_clip = self._combine_grad_norms(
                            non_indexer_grad_norm_pre_clip,
                            indexer_grad_norm_pre_clip,
                        )
                        if non_indexer_params:
                            grad_group_norms["grad_norm_non_indexer_post_clip"] = get_gradient_norm(non_indexer_params)
                        if indexer_params:
                            grad_group_norms["grad_norm_indexer_post_clip"] = get_gradient_norm(indexer_params)
                    else:
                        params_to_clip = self.model.parameters()
                        grad_norm_pre_clip = self._grad_norm_to_float(
                            self.accelerator.clip_grad_norm_(
                                params_to_clip, self.cfg.train.clip_grad
                            )
                        )

                    grad_norm_post_clip = get_gradient_norm(self.model.parameters())

                if self.accelerator.state.deepspeed_plugin is None:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.lr_scheduler.step()

            if self.accelerator.sync_gradients:
                    start_steps += 1

                    # Report to tensorboard
                    batch_output.update(loss_details_dict)
                    loss_details_dict = {}

                    if start_steps % self.cfg.train.print_freq == 0:
                        self.log_all(batch_output, start_steps, prefix='train')
                        self._log_memory_snapshot(f"train_step_{start_steps}")
                    metric_logger.update(**batch_output)

                    min_lr = 10.0
                    max_lr = 0.0
                    for group in self.optimizer.param_groups:
                        min_lr = min(min_lr, group["lr"])
                        max_lr = max(max_lr, group["lr"])

                    metric_logger.update(lr=max_lr)
                    metric_logger.update(min_lr=min_lr)
                    self._log_scalars({"lr": max_lr}, step=start_steps)
                    self._log_scalars({"min_lr": min_lr}, step=start_steps)
                    self._log_scalars({"TRAIN/lr": max_lr, "TRAIN/min_lr": min_lr}, step=start_steps)

                    weight_decay_value = None
                    for group in self.optimizer.param_groups:
                        if group["weight_decay"] > 0:
                            weight_decay_value = group["weight_decay"]
                    metric_logger.update(weight_decay=weight_decay_value)
                    metric_logger.update(grad_norm=grad_norm_pre_clip)
                    metric_logger.update(grad_norm_post_clip=grad_norm_post_clip)
                    for key, value in grad_group_norms.items():
                        metric_logger.update(**{key: value})
                    self._log_scalars({"weight_decay": weight_decay_value}, step=start_steps)
                    grad_norm_scalars = {
                        "grad_norm": grad_norm_pre_clip,
                        "grad_norm_post_clip": grad_norm_post_clip,
                    }
                    grad_norm_scalars.update(grad_group_norms)
                    self._log_scalars(grad_norm_scalars, step=start_steps)
                    train_alias_scalars = {
                        "TRAIN/weight_decay": weight_decay_value,
                        "TRAIN/grad_norm": grad_norm_pre_clip,
                        "TRAIN/grad_norm_post_clip": grad_norm_post_clip,
                    }
                    train_alias_scalars.update({f"TRAIN/{key}": value for key, value in grad_group_norms.items()})
                    self._log_scalars(train_alias_scalars, step=start_steps)

                    self.global_step = start_steps

        # # gather the stats from all processes
        # metric_logger.synchronize_between_processes()
        # print("Averaged stats:", metric_logger)

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @staticmethod
    def _loss_items(loss):
        if isinstance(loss, (list, tuple)):
            return [item for item in loss if item is not None]
        return [loss]

    @staticmethod
    def _sum_loss_items(loss_items):
        total = None
        for loss_item in loss_items:
            total = loss_item if total is None else total + loss_item
        if total is None:
            raise RuntimeError("No loss tensors were produced for this training step.")
        return total

    def _backward_loss_items(self, loss_items):
        for loss_item in loss_items:
            self.accelerator.backward(loss_item)


    def log_all(self, output, step, prefix=""):        
        if 'log_keys' in output:
            log_keys = output.log_keys
        else:
            log_keys = list(output.keys())

        log_scaler = {}
        log_img = {}
        for k in log_keys:
            v = output[k]
            if np.isscalar(v):
                log_scaler[prefix+'/'+k] = v
                if prefix in ("train", "val"):
                    log_scaler[prefix.upper()+'/'+k] = v
                continue
            if Image.isImageType(v):
                log_img[prefix+'/'+k] = v
                if prefix in ("train", "val"):
                    log_img[prefix.upper()+'/'+k] = v

        self._log_scalars(log_scaler, step)
        self._log_images(log_img, step)

    def _init_direct_tensorboard(self):
        tb_dir = self.cfg.log.get("tensorboard_dir", None) or self.cfg.log.output_dir
        os.makedirs(tb_dir, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter

        self._tb_writer = SummaryWriter(log_dir=tb_dir)
        self.log_info(f"TensorBoard events are written directly to {tb_dir}")

    def _log_scalars(self, scalars, step):
        if not scalars:
            return
        if self._direct_tensorboard and not self.accelerator.is_main_process:
            return
        if self._tb_writer is None:
            self.accelerator.log(scalars, step=step)
            return
        for tag, value in scalars.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().mean().item()
            self._tb_writer.add_scalar(tag, float(value), int(step))
        self._tb_writer.flush()

    def _log_images(self, images, step):
        if not images:
            return
        if self._direct_tensorboard and not self.accelerator.is_main_process:
            return
        if self._tb_writer is None:
            for tracker in self.accelerator.trackers:
                tracker.log_images(images, step)
            return
        for tag, image in images.items():
            if Image.isImageType(image):
                self._tb_writer.add_image(tag, np.asarray(image), int(step), dataformats="HWC")
        self._tb_writer.flush()

    def forward_batch(self, batch, mode='train'):
        output = self.model(batch)
        assert isinstance(output, EasyDict)
        return output

    def calculate_loss(self, output, batch, mode='train'):
        pass

    def build_accelerator(self):
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.cfg.log.output_dir,
            logging_dir=self.cfg.log.output_dir,
            total_limit=4,      # self.cfg.save_total_limit = 4
            # automatic_checkpoint_naming=True,
        )

        # Initialize the Environment variables throught MPI run
        init_distributed_mode(
            self.cfg.train, init_pytorch_ddp=False
        )  # set `init_pytorch_ddp` to False, since the accelerate will do later

        self._direct_tensorboard = bool(
            self.cfg.log.get("direct_tensorboard", self.cfg.log.get("use_tensorboard", False))
        )

        if self.cfg.log.use_wandb:
            log_with = 'wandb'
        elif self.cfg.log.use_tensorboard and not self._direct_tensorboard:
            log_with = 'tensorboard'
        else:
            log_with = None

        mixed_precision = 'no' if self.cfg.train.model_dtype not in ['fp8', 'fp16', 'bf16'] else self.cfg.train.model_dtype

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # dynamic complie
        if self.cfg.train.get("dynamo_backend"):
            if isinstance(self.cfg.train.dynamo_backend, str) and hasattr(
                DynamoBackend, self.cfg.train.dynamo_backend.upper()
            ):
                dynamo_backend = getattr(DynamoBackend, self.cfg.train.dynamo_backend.upper())
            elif isinstance(self.cfg.train.dynamo_backend, DynamoBackend):
                dynamo_backend = self.cfg.train.dynamo_backend
            else:
                print(
                    f"Invalid dynamo_backend {self.cfg.train.dynamo_backend}, using default. Please refer to "
                    "https://huggingface.co/docs/accelerate/v1.2.1/en/package_reference/utilities#accelerate.utils.DynamoBackend for available names."
                )
        else:
            dynamo_backend = DynamoBackend.NO

        print(f"Using dynamo backend: {dynamo_backend}")

        torch._inductor.config.reorder_for_compute_comm_overlap = True

        dynamo_plugin = TorchDynamoPlugin(
            backend=dynamo_backend,
            mode="max-autotune-no-cudagraphs",
            dynamic=self.cfg.train.get("dynamic_compile", True),
        )
        
        accelerate_config = dict(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=log_with,
            project_config=accelerator_project_config,
            dataloader_config=DataLoaderConfiguration(
                non_blocking=True,
                split_batches=False,
                dispatch_batches=None,
                even_batches=True,
                use_seedable_sampler=False,
            ),
            step_scheduler_with_optimizer=False,             # not to step n_gpus times per step.
            dynamo_plugin=dynamo_plugin,
        )

        # fsdp
        if self.cfg.get("fsdp_plugin"):
            fsdp_plugin_kwargs = {}
            fsdp_plugin_kwargs[
                "mixed_precision_policy"
            ] = torch.distributed.fsdp.MixedPrecision(
                param_dtype=self.weight_dtype,
                reduce_dtype=self.weight_dtype,
                buffer_dtype=self.weight_dtype,
                cast_forward_inputs=True,
                cast_root_forward_inputs=True,
            )

            fsdp_plugin = hydra.utils.instantiate(self.cfg.fsdp_plugin)(**fsdp_plugin_kwargs)
            accelerate_config["fsdp_plugin"] = fsdp_plugin
        else:
            ddp_kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=self.cfg.train.find_unused_parameters,
                static_graph=bool(self.cfg.train.get("static_graph", False)),
            )
            accelerate_config['kwargs_handlers'] = [ddp_kwargs]

        accelerator = Accelerator(**accelerate_config)

        self.logger = get_logger(self.cfg, os.path.basename(__file__))

        # To block the print on non main process
        setup_for_distributed(accelerator.is_main_process)

        # self.logger.rank_zero_only = False
        self.log_info(accelerator.state)
        # self.logger.rank_zero_only = True
        
        if self.cfg.random_seed is not None:
            set_seed(self.cfg.random_seed, device_specific=True)

        self.device = accelerator.device

        self.accelerator = accelerator

    def auto_resume(self):
        if self.cfg.train.resume:
            path = self.cfg.train.resume
        elif os.path.exists(self.cfg.log.ckpt_dir):
            # Get the most recent checkpoint
            dirs = os.listdir(self.cfg.log.ckpt_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint_")]
            dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            if path is not None:
                path = os.path.join(self.cfg.log.ckpt_dir, path)
        else:
            path = None

        if path is None:
            self.log_info("Checkpoint does not exist. Starting a new training run.")
            
            start_epoch = 0
        else:
            self.log_info(f"Resuming from checkpoint {path}")
            self.accelerator.load_state(
                # os.path.join(self.cfg.log.ckpt_dir, path)
                path
            )
            # Extract epoch number from checkpoint path
            # Handles both "checkpoint_N" and "best_model" formats
            if "checkpoint_" in path:
                # Extract epoch from "checkpoint_N" format
                checkpoint_name = path.rstrip('/').split('/')[-1]
                # checkpoint_N is saved after finishing epoch N, so resume from N+1.
                start_epoch = int(checkpoint_name.split("checkpoint_")[-1]) + 1
            else:
                # For "best_model" or other formats, start from epoch 0
                # This is correct for stage transitions where we want to reset the epoch counter
                start_epoch = 0
            self._extend_resumed_scheduler_if_needed(path, start_epoch)

        return start_epoch

    def _base_lr_scheduler(self):
        return getattr(self.lr_scheduler, "scheduler", self.lr_scheduler)

    def _extend_resumed_scheduler_if_needed(self, checkpoint_path, start_epoch):
        if start_epoch <= 0:
            return

        scheduler = self._base_lr_scheduler()
        if scheduler is None or not hasattr(scheduler, "total_steps"):
            return

        try:
            resumed_total_steps = int(scheduler.total_steps)
            target_total_steps = int(self.cfg.train.num_epoch) * int(self.iters_per_epoch)
        except Exception:
            self.log_info(
                f"Skip resumed scheduler total_steps check for {checkpoint_path}: "
                f"cannot parse scheduler.total_steps={getattr(scheduler, 'total_steps', None)}"
            )
            return

        if target_total_steps <= resumed_total_steps:
            return

        scheduler_name = scheduler.__class__.__name__
        if scheduler_name != "OneCycleLR":
            self.log_info(
                f"Resume target has more train steps ({target_total_steps}) than loaded "
                f"{scheduler_name} total_steps ({resumed_total_steps}), but automatic "
                "extension is only implemented for OneCycleLR."
            )
            return

        pct_start = float(self.cfg.train.lr_scheduler.get("pct_start", 0.3))
        three_phase = bool(self.cfg.train.lr_scheduler.get("three_phase", False))
        if three_phase:
            schedule_phases = [
                {
                    "end_step": float(pct_start * target_total_steps) - 1,
                    "start_lr": "initial_lr",
                    "end_lr": "max_lr",
                    "start_momentum": "max_momentum",
                    "end_momentum": "base_momentum",
                },
                {
                    "end_step": float(2 * pct_start * target_total_steps) - 2,
                    "start_lr": "max_lr",
                    "end_lr": "initial_lr",
                    "start_momentum": "base_momentum",
                    "end_momentum": "max_momentum",
                },
                {
                    "end_step": target_total_steps - 1,
                    "start_lr": "initial_lr",
                    "end_lr": "min_lr",
                    "start_momentum": "max_momentum",
                    "end_momentum": "max_momentum",
                },
            ]
        else:
            schedule_phases = [
                {
                    "end_step": float(pct_start * target_total_steps) - 1,
                    "start_lr": "initial_lr",
                    "end_lr": "max_lr",
                    "start_momentum": "max_momentum",
                    "end_momentum": "base_momentum",
                },
                {
                    "end_step": target_total_steps - 1,
                    "start_lr": "max_lr",
                    "end_lr": "min_lr",
                    "start_momentum": "base_momentum",
                    "end_momentum": "max_momentum",
                },
            ]

        scheduler.total_steps = target_total_steps
        scheduler._schedule_phases = schedule_phases
        self.log_info(
            f"Extended resumed OneCycleLR total_steps from {resumed_total_steps} "
            f"to {target_total_steps} for checkpoint {checkpoint_path}; "
            f"last_epoch={getattr(scheduler, 'last_epoch', None)}, start_epoch={start_epoch}."
        )

    def log_info(self, info):
        if is_logging_process():
            self.logger.info(info)
