import abc
import torch
import inspect
import random
from typing import Callable
import warnings
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.cam_utils import compute_camera_dist_rot_similarity, compute_camera_similarity
from easyvolcap.utils.math_utils import affine_padding, affine_inverse
from easyvolcap.dataloaders.datasets.src_view_selector.base import DistRotSrcviewSelector, RandomSrcviewSelector, SequentialSrcviewSelector
from easyvolcap.dataloaders.datasets.src_view_selector.multicam import MultiviewSeqSrcviewSelector, PairwiseGraphSrcviewSelector

class UniformSrcviewSelector(abc.ABC):
    def __init__(self, src_exts: torch.Tensor, random_ap: int, source_cfg: dotdict, is_train: bool):
        self.source_cfg = source_cfg
        self.source_type = source_cfg.get("type", "RANDOM")
        self.closest_using_t = source_cfg.get("closest_using_t", False)
        if self.source_type == "RANDOM":
            self.selector = RandomSrcviewSelector(src_exts, random_ap, is_train)
        elif self.source_type == "DISTANCEROT":
            self.selector = DistRotSrcviewSelector(src_exts, random_ap, is_train)
        elif self.source_type == "SEQUENTIAL":
            assert self.closest_using_t, "SequentialSrcviewSelector requires closest_using_t to be True"
            self.selector = SequentialSrcviewSelector(src_exts, random_ap, is_train)
        elif self.source_type == "MULTIVIEWSEQ":
            assert self.closest_using_t, "MultiviewSeqSrcviewSelector requires closest_using_t to be True"
            self.selector = MultiviewSeqSrcviewSelector(src_exts, random_ap, is_train)
        elif self.source_type == "PAIRS":
            assert self.closest_using_t, "PairwiseGraphSrcviewSelector requires closest_using_t to be True"
            self.selector = PairwiseGraphSrcviewSelector(src_exts, random_ap, is_train, source_cfg)
        else:
            raise NotImplementedError

    def __call__(self, target_index, extra_index, n_srcs, output: dotdict = None):
        inds, extra_inds = self.selector(target_index, extra_index, n_srcs, output=output)
        return inds, extra_inds
