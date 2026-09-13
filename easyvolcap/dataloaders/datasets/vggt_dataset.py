"""Deprecated. Please directly use `MultiviewPointDataset`"""

import torch
import random
import numpy as np
from typing import List, Dict, Union

from easyvolcap.engine import cfg, args, DATASETS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.dataloaders.datasets.multiview_point_dataset import MultiviewPointDataset
from easyvolcap.dataloaders.datasets.volumetric_video_dataset import VolumetricVideoDataset

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import collate_fn


@DATASETS.register_module()
class VGGTDataset(MultiviewPointDataset):
    def __init__(self,
                 n_frames_batch: int = 48,
                 inside_random: bool = True,  # outside __getitem__() is fixed
                 **kwargs,
                 ):
        # Ignore things, since this will serve as a base class of classes supporting *args and **kwargs
        # The inspection of registration and config system only goes down one layer
        # Otherwise it would be to inefficient
        call_from_cfg(super().__init__, kwargs)

        # Fixed number of frames in a batch, actually batch * n_srcs
        self.n_frames_batch = n_frames_batch
        self.inside_random = inside_random

        # TODO: remove this file after 2026-03-01
        log(red("VGGTDataset is deprecated. Please use `MultiviewPointDataset` instead."))

    def get_metadata(self, index: dotdict):
        # TODO: remove this file after 2026-03-01
        log(red("VGGTDataset is deprecated. Please use `MultiviewPointDataset` instead."))

        # `VGGTDataset` requires `ImageBasedBatchSampler`,
        # for more elegant random seed control
        index, n_srcs, aspect_ratio = index.index, index.n_srcs, index.aspect_ratio
        n_srcs = min(n_srcs, len(self) - 1)

        # Number of batches to sample
        n_batchs = max(self.n_frames_batch // (n_srcs + 1), 1)

        if self.inside_random:
            i_batchs = random.choices(
                population=range(len(self)),
                k=n_batchs,
            )
        else:
            # Randomly sample `n_batchs - 1` indices
            # No need to exclude the current index, since it serves as the
            # target view for each batch
            i_batchs = [index] + random.choices(
                population=range(len(self)),
                k=n_batchs-1
            )

        # Total output placeholder
        output = collate_fn([
            MultiviewPointDataset.get_metadata(
                self,
                dotdict(
                    index=idx,
                    n_srcs=n_srcs,
                    aspect_ratio=aspect_ratio
                )
            ) for idx in i_batchs
        ])

        return output
