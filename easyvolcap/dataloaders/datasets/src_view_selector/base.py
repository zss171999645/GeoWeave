import abc
import inspect
import random
from typing import Callable
import warnings
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.cam_utils import compute_camera_dist_rot_similarity, compute_camera_similarity
from easyvolcap.utils.math_utils import affine_padding, affine_inverse
import torch


class BaseSelector(abc.ABC):
    def __init__(self, src_exts: torch.Tensor, random_ap: int, is_train: bool):
        """
        Base class for source view selectors.
        Args:
            src_exts: [N, Ncam, 4, 4]
            random_ap: the number of additional source views to sample
        """
        self.src_exts = src_exts.clone()
        self.n_views = src_exts.shape[0]
        self.n_cams = src_exts.shape[1]
        if len(self.src_exts.shape) != 4:
            raise ValueError("src_exts should be a 4D tensor")
        if self.n_views <= 1:
            raise ValueError("src_exts should have at least 2 views")
        
        self.random_ap = random_ap
        self.is_train = is_train
    
    def sample_n(self, inds, n_srcs):
        if self.random_ap:
            if n_srcs < len(inds): inds = random.sample(inds, n_srcs)  # (S,), no replacement
            else: inds = random.choices(inds, k=n_srcs)  # (S,), with replacement
        else:
            assert len(inds) == n_srcs, "The number of source views should be equal to n_srcs"
        return inds

    @abc.abstractmethod
    def __call__(self, target_index, extra_index, n_srcs, output: dotdict = None):
        raise NotImplementedError


class RandomSrcviewSelector(BaseSelector):
    def __init__(self, src_exts: torch.Tensor, random_ap: int, is_train: bool, source_cfg: dotdict = None):
        super().__init__(src_exts, random_ap, is_train)

    def __call__(self, target_index, extra_index, n_srcs, output: dotdict = None):
        if not self.is_train:
            warnings.warn("warning: random src view selector is used in eval mode")

        inds = list(range(self.n_views))
        inds.remove(target_index)
        inds = self.sample_n(inds, n_srcs)

        return inds, extra_index


class DistRotSrcviewSelector(BaseSelector):
    def __init__(self, src_exts: torch.Tensor, random_ap: int, is_train: bool, source_cfg: dotdict = None):
        super().__init__(src_exts, random_ap, is_train)
        if not self.is_train:
            assert self.random_ap == 0, "DistRotSrcviewSelector: random_ap should be 0 in eval mode"
    
    def __call__(self, target_index, extra_index, n_srcs, output: dotdict = None):
        # NOTE that this selector only select views from the same camera, not the whole scene
        tar_c2ws = self.src_exts[target_index, extra_index][None]  # (1, 4, 4)
        src_c2ws = affine_inverse(self.src_exts[:, extra_index])  # (N, 4, 4)
        _, src_inds = compute_camera_dist_rot_similarity(tar_c2ws, src_c2ws)  # (1, N), (1, N)
        inds = src_inds[0, 1:1 + n_srcs + self.random_ap].numpy().tolist()  # always exclude the target view, (S + random_ap,)
        inds = self.sample_n(inds, n_srcs)
        return inds, extra_index


class SequentialSrcviewSelector(BaseSelector):
    def __init__(self, src_exts: torch.Tensor, random_ap: int, is_train: bool, source_cfg: dotdict = None):
        super().__init__(src_exts, random_ap, is_train)
        if not self.is_train:
            assert self.random_ap == 0, "SequentialSrcviewSelector: random_ap should be 0 in eval mode"
    
    def __call__(self, target_index, extra_index, n_srcs, output: dotdict = None):
        sidx = max(0, target_index - (n_srcs + self.random_ap) // 2)
        inds = list(range(sidx, min(sidx + n_srcs + self.random_ap + 1, self.n_views)))  # (S + random_ap + 1,)
        inds.remove(target_index)  # (S + random_ap,), remove itself
        
        if len(inds) < n_srcs:
            warnings.warn(f"SequentialSrcviewSelector: The number of source views is less than n_srcs, got {len(inds)}")
        else:
            assert len(inds) == n_srcs, f"The number of source views should be equal to n_srcs, got {len(inds)}"
        inds = self.sample_n(inds, n_srcs)
        return inds, extra_index
