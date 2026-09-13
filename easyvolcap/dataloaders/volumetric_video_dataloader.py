# This file defines typical data loaders to be constructed in the runner
# Dataloader
#     Dataset
#     Collator
import cv2
import torch
import os
import numpy as np
from typing import List
from torch.utils.data import DataLoader, get_worker_info

from easyvolcap.engine import cfg, args  # automatically register all components when importing this
from easyvolcap.engine import DATASETS, DATASAMPLERS, DATALOADERS
from easyvolcap.dataloaders.datasets.volumetric_video_dataset import VolumetricVideoDataset
from easyvolcap.dataloaders.datasamplers import BatchSampler, RandomSampler, IterationBasedBatchSamplerWrapper
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict  # we use dotdict as the standard dictionary
from easyvolcap.utils.data_utils import collate_fn
from easyvolcap.utils.net_utils import setup_deterministic
from easyvolcap.utils.data_utils import default_collate, default_convert

# https://github.com/pytorch/pytorch/issues/11201
import torch.multiprocessing
_sharing_strategy = os.environ.get('EVC_MP_SHARING_STRATEGY', 'file_system').strip() or 'file_system'
try:
    torch.multiprocessing.set_sharing_strategy(_sharing_strategy)
except RuntimeError:
    torch.multiprocessing.set_sharing_strategy('file_system')
# https://pytorch.org/docs/stable/multiprocessing.html#file-system-file-system
# File system - file_system
# This strategy will use file names given to shm_open to identify the shared memory regions. This has a benefit of not requiring the implementation to cache the file descriptors obtained from it, but at the same time is prone to shared memory leaks. The file can’t be deleted right after its creation, because other processes need to access it to open their views. If the processes fatally crash, or are killed, and don’t call the storage destructors, the files will remain in the system. This is very serious, because they keep using up the memory until the system is restarted, or they’re freed manually.
# To counter the problem of shared memory file leaks, torch.multiprocessing will spawn a daemon named torch_shm_manager that will isolate itself from the current process group, and will keep track of all shared memory allocations. Once all processes connected to it exit, it will wait a moment to ensure there will be no new connections, and will iterate over all shared memory files allocated by the group. If it finds that any of them still exist, they will be deallocated. We’ve tested this method and it proved to be robust to various failures. Still, if your system has high enough limits, and file_descriptor is a supported strategy, we do not recommend switching to this one.
# torch.multiprocessing.set_start_method('spawn')  # https://pytorch.org/docs/stable/data.html#platform-specific-behaviors


def worker_init_fn(worker_id, fix_random, allow_tf32, deterministic, benchmark):
    cv2.setNumThreads(1)  # only 1 thread for opencv undistortion, high cpu, not faster
    # torch.set_num_threads(1)  # only 1 thread for torch, high cpu, not faster
    setup_deterministic(fix_random, allow_tf32, deterministic, benchmark, worker_id)

    worker_info = get_worker_info()
    dataset = worker_info.dataset


@DATALOADERS.register_module()
class VolumetricVideoDataloader(DataLoader):
    # NOTE: order for arguments: constructed objects, default configurations, default arguments
    def __init__(self,
                 # Dataloader configs
                 num_workers: int = 8,  # heavy on cpu, reduce memory usage
                 prefetch_factor: int = 3,  # heavy on cpu, reduce memory usage
                 pin_memory: bool = True,  # heavy on memory
                 max_iter: int = cfg.runner_cfg.ep_iter * cfg.runner_cfg.epochs,  # MARK: global config

                 # Per-process parameters
                 fix_random: bool = cfg.fix_random,
                 allow_tf32: bool = cfg.allow_tf32,
                 deterministic: bool = cfg.deterministic,  # for debug use only, # MARK: global config
                 benchmark: bool = cfg.benchmark,  # for debug use only, # MARK: global config

                 # Dataset configs
                 dataset_cfg: dotdict = dotdict(type=VolumetricVideoDataset.__name__),
                 sampler_cfg: dotdict = dotdict(type=RandomSampler.__name__),  # plain sampler
                 batch_sampler_cfg: dotdict = dotdict(type=BatchSampler.__name__),  # plain sampler
                 ):
        # Preparing objects for dataloader
        if batch_sampler_cfg.batch_size == -1: batch_sampler_cfg.batch_size = len(dataset)
        dataset: VolumetricVideoDataset = DATASETS.build(dataset_cfg)
        sampler: RandomSampler = DATASAMPLERS.build(sampler_cfg, dataset=dataset)
        batch_sampler: BatchSampler = DATASAMPLERS.build(batch_sampler_cfg, sampler=sampler)  # exposing config, not the best practice?
        if max_iter != -1: batch_sampler = IterationBasedBatchSamplerWrapper(batch_sampler, max_iter)

        # GUI related special config
        if benchmark == 'train': benchmark = args.type == 'train'  # for static sized input

        # Initialization of dataloader object
        super().__init__(dataset=dataset,
                         batch_sampler=batch_sampler,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         collate_fn=collate_fn,
                         worker_init_fn=partial(worker_init_fn, fix_random=fix_random, allow_tf32=allow_tf32, deterministic=deterministic, benchmark=benchmark),
                         prefetch_factor=prefetch_factor if num_workers > 0 else None if torch.__version__[0] >= '2' else 2,
                         persistent_workers=num_workers > 0,
                         )

        # Only for annotation
        self.dataset: VolumetricVideoDataset
