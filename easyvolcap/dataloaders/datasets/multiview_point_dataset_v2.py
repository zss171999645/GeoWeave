""" This is a simplified/refactored version of the original MultiviewPointDataset. Ask Xiaoyang Guo for details.

"""


import cv2
import torch
import random
import pickle
import numpy as np
from functools import lru_cache
from typing import List, Dict, Literal, Tuple, Union

from easyvolcap.engine import DATASETS
from easyvolcap.engine import cfg, args
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.dataloaders.datasets.volumetric_video_dataset import VolumetricVideoDataset

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.ray_utils import get_rays, get_plucker
from easyvolcap.utils.aug_utils import get_image_augmentation
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.math_utils import affine_padding, affine_inverse
from easyvolcap.utils.data_utils import DataSplit, pin_memory, to_tensor, as_torch_func, sparse_resize
from easyvolcap.utils.image_utils import crop_nhwc_image, rotate_90_degree, fill_nhwc_image
from easyvolcap.utils.depth_utils import Depth, Scale, convert_linear_depth, convert_inverse_depth
from easyvolcap.utils.cam_utils import (Sourcing, 
                                        encode_camera_params,
                                        add_noise_to_poses)
from easyvolcap.dataloaders.datasets.src_view_selector.uniform import UniformSrcviewSelector

# We have a tricky situation here:
# There are datasets that stores all images in a single folder
# We want to sample closest camera from all those images with ease
# While maintaining the ability to use different kinds of dataloading techniques
# It's essentially a dataset transpose problem
# Maybe the best solution is to just store all images in a separated folder?
# For now, we need to add these dirty supports for
# 1. Distributed training -> only convert latent_index (which might be used for view selection)
# 2. Training dataloading -> there's a definition of latent and view index, so it's quite easy
# 3. Inference (especially gui) dataloading -> no definition of so called view_index -> tricky implementation of selection logic

# For distributed training, should not perform datasharding since all source views are needed
# Or we could implement some dataset based sharding technology cleverly?

@DATASETS.register_module()
class MultiviewPointDatasetV2(VolumetricVideoDataset):
    def __init__(self,
                 n_srcs_list: List[int] = [16],  # MARK: repeated global configuration
                 n_srcs_prob: List[float] = [1.0],  # MARK: repeated global configuration
                 closest_using_t: bool = False,  # find the closest view using the temporal dimension
                 source_cfg: dotdict = dotdict(
                     type="DISTANCEROT",
                     extra_src_pool=0,
                     extra_src_pool_ratio=-1,  # ratio of extra source pool relative to the number of source views
                 ),  # Sourcing.DISTANCE or Sourcing.ZIGZAG or Sourcing.SEQUENTIAL or Sourcing.RANDOM or Sourcing.DISTANCEROT
                 supply_decoded: bool = False,
                 barebone: bool = False,
                 disk_dataset: bool = False,

                 src_view_sample: List = [0, None, 1],  # use these as input source views
                 force_sparse_view: bool = True,  # The user will be responsible for setting up the correct view count

                 use_z_depth: bool = True,  # use z-depth for normalization
                 min_depth: float = 0.0,  # minimum valid depth value for supervision, valid only if it is set to > 0
                 max_depth: float = 1e9,  # maximum valid depth value for supervision, valid only if it is set to > 0
                 min_depth_quantile: float = 0.01,  # minimum valid depth value for supervision
                 max_depth_quantile: float = 0.99,  # maximum valid depth value for supervision
                 scale: float = 1e-2,  # fixed global scale for scene normalization
                 scale_type: str = Scale.DIST.name,  # depth type, Depth.LINEAR or Depth.INVERSE
                 use_world_coord: bool = True,  # use world coordinate or camera coordinate for supervision
                 infer: bool = False,  # evaluation mode, no ground truth depth

                 depth_type: str = Depth.LINEAR.name,  # depth type, Depth.LINEAR or Depth.INVERSE
                 depth_scale: float = 1.0,  # scaling factor for input raw ground truth depth
                 depth_near: float = 0.02,  # near clipping plane for inverse depth conversion
                 depth_far: float = 1e5,  # far clipping plane for inverse depth conversion

                 proc_max_size: int = -1,  # processing time maximum size, used for resizing
                 proc_align_size: int = 1,  # processing time align size, used for resizing

                 portrait_check: bool = False,  # check if landscape or portrait
                 portrait_ratio: float = 1.25,  # threshold for landscape or portrait
                 rot90: Optional[Literal["clockwise", "counterclockwise"]] = None,  # rotate the image 90 degrees clockwise or counter-clockwise
                 safe_bound: int = 4,  # safe bound for the image size
                 center_crop: bool = True,  # center crop the image to the target aspect ratio
                 limit_max_aspect_ratio_as_original_ratio: bool = False,  # limit the maximum aspect ratio as the original image ratio

                 aug_scale: float = None,  # image rescale augmentation
                 aug_scale_ratio: float = 1.0,  # probability of applying rescale augmentation
                 aug_image: bool = False,  # whether to apply image augmentation
                 aug_image_cfg: dotdict = dotdict(
                     color_jitter=None,
                     gray_scale=True,
                     gau_blur=False,
                 ),
                 aug_image_cojit: bool = False,  # whether to apply color jitter augmentation
                 aug_image_cojit_ratio: float = 0.3,  # probability of applying color jitter augmentation individually
                 shared_aug_scale_value: bool = False,
                 sort_view_inds: bool = False,
                 **kwargs,
                 ):
        # Ignore things, since this will serve as a base class of classes supporting *args and **kwargs
        # The inspection of registration and config system only goes down one layer
        # Otherwise it would be to inefficient
        call_from_cfg(super().__init__, kwargs, disk_dataset=disk_dataset, use_z_depth=use_z_depth)

        self.closest_using_t = closest_using_t
        self.src_view_sample = src_view_sample
        assert not self.closest_using_t or self.frame_sample == [0, None, 1] or force_sparse_view, "Should use default frame_sample [0, None, 1] for ibr dataset with `closest_using_t`. Control sampling through sampler.frame_sample and src_view_sample"
        assert self.view_sample == [0, None, 1] or force_sparse_view, "Should use default view_sample [0, None, 1] for ibr dataset. Control sampling through sampler.view_sample and src_view_sample"
        assert not (self.cache_raw and not supply_decoded), "Will always supply decoded source images when cache_raw is enabled for faster sampling, set cache_raw to False to supply jpeg streams"

        self.source_cfg = source_cfg
        self.source_cfg["closest_using_t"] = self.closest_using_t
        self.source_type = Sourcing[source_cfg["type"]]
        # Views are selected and loaded
        # Frames are selected and loaded
        self.load_source_params()
        # Need to build all possible view selections (distance of c2w)
        # - Dot product of v_front - euclidian distance of center
        # NOTE: Only compute when disk_dataset is False to avoid large scale matrix OOM
        assert self.disk_dataset, "Only support disk_dataset for now"

        self.n_srcs_list = n_srcs_list if n_srcs_list is None or len(n_srcs_list) != 1 or n_srcs_list[0] != 0 else [self.n_views]
        self.n_srcs_prob = n_srcs_prob
        self.extra_src_pool = source_cfg.get("extra_src_pool", 0)
        self.extra_src_pool_ratio = source_cfg.get("extra_src_pool_ratio", -1)
        assert not (self.extra_src_pool > 0 and self.extra_src_pool_ratio > 0), "Cannot set both extra_src_pool and extra_src_pool_ratio, choose one of them"

        # src_inps will come in as decoded bytes instead of jpegs
        self.supply_decoded = supply_decoded
        self.barebone = barebone

        # Global scale for validation and testing pose normalization
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_depth_quantile = min_depth_quantile
        self.max_depth_quantile = max_depth_quantile
        self.scale = scale
        self.scale_type = Scale[scale_type]
        self.use_world_coord = use_world_coord
        self.infer = infer

        # Depth related configurations
        self.depth_type = Depth[depth_type]
        self.depth_cfg = dotdict(scale=depth_scale, near=depth_near, far=depth_far)

        # Processing related configurations
        self.proc_max_size = proc_max_size
        self.proc_align_size = proc_align_size

        # Check if landscape or portrait
        self.portrait_check = portrait_check
        self.portrait_ratio = portrait_ratio
        # Image post-processing
        self.rot90 = rot90
        self.safe_bound = safe_bound
        self.center_crop = center_crop
        self.limit_max_aspect_ratio_as_original_ratio = limit_max_aspect_ratio_as_original_ratio

        # Augmentation related configurations
        self.aug_scale = aug_scale
        self.aug_scale_ratio = aug_scale_ratio
        self.aug_image = aug_image
        self.aug_image_cfg = aug_image_cfg
        self.aug_image_cojit = aug_image_cojit
        self.aug_image_cojit_ratio = aug_image_cojit_ratio
        # Maybe create the image augmentor
        self.augmentor = None
        if self.aug_image:
            self.augmentor = get_image_augmentation(
                **self.aug_image_cfg
            )
        self.shared_aug_scale_value = shared_aug_scale_value
        self.sort_view_inds = sort_view_inds

    def load_source_params(self):
        # Perform view selection first
        view_inds = self.frame_inds if self.closest_using_t else self.view_inds
        view_inds = torch.arange(0, len(view_inds))
        if len(self.src_view_sample) != 3: view_inds = view_inds[self.src_view_sample]  # this is a list of indices
        else: view_inds = view_inds[self.src_view_sample[0]:self.src_view_sample[1]:self.src_view_sample[2]]  # begin, start, end
        self.src_view_inds = view_inds
        if len(view_inds) == 1: view_inds = [view_inds]  # MARK: pytorch indexing bug, when length is 1, will reduce a dim

        # For getting the actual data (renaming w2c and K)
        if self.closest_using_t:  # this checks whether the view selection is performed on the frame or view dim
            self.src_ixts = self.Ks[:, view_inds]  # N, L, 4, 4
            self.src_exts = affine_padding(self.w2cs[:, view_inds])  # N, L, 4, 4
            self.src_ixts = self.src_ixts.permute(1, 0, 2, 3)  # L, N, 4, 4 # MARK: transpose
            self.src_exts = self.src_exts.permute(1, 0, 2, 3)  # L, N, 4, 4 # MARK: transpose
            if self.use_3ddr:
                self.src_exts_3ddr = affine_padding(self.w2cs_3ddr[:, view_inds])  # N, L, 4, 4
                self.src_exts_3ddr = self.src_exts_3ddr.permute(1, 0, 2, 3)  # L, N, 4, 4 # MARK: transpose
            self.src_Hs = self.Hs[:, view_inds].permute(1, 0)  # L, N
            self.src_Ws = self.Ws[:, view_inds].permute(1, 0)  # L, N

        else:
            self.src_ixts = self.Ks[view_inds]  # N, L, 4, 4
            self.src_exts = affine_padding(self.w2cs[view_inds])  # N, L, 4, 4
            self.src_Hs = self.Hs[view_inds]  # N, L
            self.src_Ws = self.Ws[view_inds]  # N, L

    def get_source_views(self, src_exts: np.ndarray, target_index: int, extra_index: int, n_srcs: int, output: dotdict):
        if self.extra_src_pool_ratio > 0: random_ap = int(self.extra_src_pool_ratio * n_srcs)
        else: random_ap = self.extra_src_pool  # training -> randomly sample more
        is_train = self.split == DataSplit.TRAIN

        selector = UniformSrcviewSelector(src_exts, random_ap, self.source_cfg, is_train=is_train)
        inds, extra_inds = selector(target_index, extra_index, n_srcs, output=output)

        if isinstance(extra_inds, int):
            extra_inds = [extra_inds] * n_srcs
        
        # Always include the target index
        inds = torch.as_tensor([target_index] + inds)  # (1 + S,)
        extra_inds = torch.as_tensor([extra_index] + extra_inds)  # (1 + S,)

        # sort inds and extra_inds
        if self.sort_view_inds:
            extra_inds = extra_inds[torch.argsort(inds)]
            inds = inds[torch.argsort(inds)]

        if "covis_matrix" not in output:
            # covis_matrix: (S, S) bool, sparse attn mask
            # covis_pairs: (P, 2) int, pairs of source views
            # covis_indices: (P,) float, confidence of the pairs
            output.covis_matrix = torch.abs(inds[:, None] - inds[None, :]) <= 5  # temporal setup
            output.covis_pairs = output.covis_matrix.nonzero(as_tuple=False).contiguous().view(-1, 2)  # (P, 2)
            output.covis_indices = torch.ones(output.covis_pairs.shape[0], dtype=torch.float32)  # (P,)
            output.covis_dist = torch.abs(inds[output.covis_pairs[:, 0]] - inds[output.covis_pairs[:, 1]] - 0.01).float().unsqueeze(1).repeat(1, 2)

        return inds, extra_inds

    def get_metadata(self, index: dotdict):
        assert isinstance(index, dotdict), f"Index should be a dotdict from VGGTBatchSampler, got {type(index)}"
        index, n_srcs, aspect_ratio = index.index, index.n_srcs, index.get('aspect_ratio', -1)
        
        if self.source_type in (Sourcing.MULTIVIEWSEQ, Sourcing.MULTIVIEWSEQV2):
            # modify index to camera 00, n_cam * n_latents -> camrea_00 index
            # src imgs/ixts shape: [n_cam, n_latents, ...], n_latents = n_frames per camera
            index = index % self.n_latents

        # Load target view related stuff
        # Get `H, W, K, D, n, f, w2c, c2w, view_index, camera_index, latent_index, frame_index` in `output`
        output = VolumetricVideoDataset.get_metadata(self, index)  # target view camera matrices

        # Load source view related stuff
        if self.closest_using_t:  # selecting closest view along temporal dimension # MARK: transpose
            target_index = output.latent_index
            extra_index = output.view_index
        else:
            target_index = output.view_index
            extra_index = output.latent_index

        # Clone, clone and clone...
        src_exts = self.src_exts.clone()
        src_ixts = self.src_ixts.clone()

        if self.use_3ddr:
            src_exts_3ddr = self.src_exts_3ddr.clone()

        # Select the source views
        inds, extra_index = self.get_source_views(src_exts, target_index, extra_index, n_srcs, output=output)

        output.t_inds = extra_index
        output.meta.t_inds = extra_index
        output.w2cs = src_exts[inds, extra_index]  # (S, 4, 4)
        output.ixts = src_ixts[inds, extra_index]  # (S, 3, 3)
        if self.use_3ddr:
            output.w2cs_3ddr = src_exts_3ddr[inds, extra_index]  # (S, 4, 4)
            if self.aug_3ddr:
                output.w2cs_3ddr = add_noise_to_poses(output.w2cs_3ddr.clone(), trans_sigma=0.05, rot_sigma_deg=2.0)

        # Other bookkeepings
        inds = self.src_view_inds.gather(-1, inds)  # (S,) -> (T, S, L) -> (T, S, L)
        output.inds = inds  # as tensors
        output.meta.inds = inds  # as tensors
        output.meta.use_world_coord = self.use_world_coord
        output.meta.aspect_ratio = aspect_ratio  # record aspect ratio for resizing

        source_index = inds.detach().cpu().numpy().tolist()
        if isinstance(extra_index, torch.Tensor):
            extra_index = extra_index.detach().cpu().numpy().tolist()
        if self.closest_using_t:  # selecting closest view along temporal dimension # MARK: transpose
            latent_index = source_index
            view_index = extra_index
        else:
            latent_index = extra_index
            view_index = source_index

        # Load rgb, msk, wet, dpt, bkg, norm for all the selected views in the local window
        output = self.get_sources(latent_index, view_index, output)

        # Get the normalized xyz, c2ws and plucker embedding using normalized camera poses
        output = self.get_xyz(output)

        return output

    def get_sources(self, latent_index: Union[List[int], int], view_index: Union[List[int], int], output: dotdict):
        # Most of the time we asynchronously load images for training, thus no need to decode them using nvjpeg
        rgb, msk, dynamic_msk, seg, wet, dpt, bkg, norm, H, W, K = zip(*parallel_execution(view_index, latent_index, output=output, action=self.get_image, sequential=True, num_workers=self.dataloading_workers))
        # NOTE: Extremely important to clone the tensor in the list to avoid modifying the original tensor
        clone = lambda x: [i.clone() if isinstance(i, torch.Tensor) else i for i in x]
        rgb, msk, dynamic_msk, seg, wet, dpt, bkg, norm, H, W, K = clone(rgb), clone(msk), clone(dynamic_msk), clone(seg), clone(wet), clone(dpt), clone(bkg), clone(norm), clone(H), clone(W), clone(K)

        # TODO: Maybe find a better way to do the moderate resizing?
        # Only consider the moderate resizing for now
        # All the views in the source list should have the same height and width
        if (len(self.render_ratio.shape) and  # avoid length of 0-d tensor error, check length of shape
                self.render_ratio[output.view_index] != 1.0) or \
                self.render_ratio != 1.0:
            render_ratio = self.render_ratio[output.view_index] if len(self.render_ratio.shape) else self.render_ratio

            # Adjust the image height and width
            h, w = int(H[0] * render_ratio), int(W[0] * render_ratio)
            ratio_h, ratio_w = h / H[0], w / W[0]
            output.H, output.W = h, w
            output.meta.H, output.meta.W = h, w

            # Adjust the camera intrinsics
            for i in range(len(K)):
                K[i][0:1] *= ratio_w
                K[i][1:2] *= ratio_h

            # Resize all the existing images in the source list
            rgb = [as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_AREA))(i) for i in rgb]
            if msk[0] is not None: msk = [as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_NEAREST))(i) for i in msk][..., None]
            if wet[0] is not None: wet = [as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_NEAREST))(i) for i in wet][..., None]
            if dpt[0] is not None: 
                if self.sparse_interpolate: dpt = [as_torch_func(partial(sparse_resize, H=h, W=w))(i) for i in dpt]
                else: dpt = [as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_NEAREST))(i) for i in dpt][..., None]
            if bkg[0] is not None: bkg = [as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_AREA))(i) for i in bkg]
            if norm[0] is not None: norm = [as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_AREA))(i) for i in norm]
            if dynamic_msk[0] is not None: dynamic_msk = [as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_NEAREST))(i) for i in dynamic_msk][..., None]
            if seg[0] is not None: seg = [as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_NEAREST))(i) for i in seg][..., None]
        if self.rot90 is not None:
            clockwise = self.rot90 == 'clockwise'
            for i in range(len(rgb)):
                H[i], W[i] = W[i], H[i]  # remember to swap the height and width
                rgb[i], K[i], output.w2cs[i] = rotate_90_degree(rgb[i], K[i], output.w2cs[i].clone(), clockwise=clockwise)
                if msk[i] is not None: msk[i], _, _ = rotate_90_degree(msk[i], K[i], output.w2cs[i].clone(), clockwise=clockwise)
                if wet[i] is not None: wet[i], _, _ = rotate_90_degree(wet[i], K[i], output.w2cs[i].clone(), clockwise=clockwise)
                if dpt[i] is not None: dpt[i], _, _ = rotate_90_degree(dpt[i], K[i], output.w2cs[i].clone(), clockwise=clockwise)
                if bkg[i] is not None: bkg[i], _, _ = rotate_90_degree(bkg[i], K[i], output.w2cs[i].clone(), clockwise=clockwise)
                if norm[i] is not None: norm[i], _, _ = rotate_90_degree(norm[i], K[i], output.w2cs[i].clone(), clockwise=clockwise)
                if dynamic_msk[i] is not None: dynamic_msk[i], _, _ = rotate_90_degree(dynamic_msk[i], K[i], output.w2cs[i].clone(), clockwise=clockwise)
                if seg[i] is not None: seg[i], _, _ = rotate_90_degree(seg[i], K[i], output.w2cs[i].clone(), clockwise=clockwise)
        # Use the original aspect ratio if not specified
        if output.meta.aspect_ratio < 0:
            output.meta.aspect_ratio = H[0] / W[0]
        # Limit the maximum aspect ratio as the original image ratio if specified
        if self.limit_max_aspect_ratio_as_original_ratio:
            output.meta.aspect_ratio = min(H[0] / W[0], output.meta.aspect_ratio)
        # Record the original aspect ratio
        original_aspect_ratio = H[0] / W[0]

        # Post-process images in the source list
        shared_aug_scale = 0
        if self.aug_scale and self.shared_aug_scale_value and random.random() < self.aug_scale_ratio:
            shared_aug_scale = np.random.triangular(0, 0, self.aug_scale)
        for i in range(len(rgb)):
            # Aspect ratio based resizing
            if output.meta.aspect_ratio > 0:
                if self.proc_max_size > 0: h, w = int(output.meta.aspect_ratio * self.proc_max_size), self.proc_max_size
                else: h, w = int(output.meta.aspect_ratio * max(H[i], W[i])), max(H[i], W[i])
            else: h, w = int(H[i]), int(W[i])
            # Check if the image is in portrait mode
            if self.portrait_check and H[i] > W[i] * self.portrait_ratio:
                if h != w and random.random() < 0.5: h, w, rotate = w, h, True  # maybe swap the height and width
                else: h, w, rotate = h, w, False
            else: h, w, rotate = h, w, False
            # Align the image size to get the final target size
            ht, wt = h // self.proc_align_size * self.proc_align_size, w // self.proc_align_size * self.proc_align_size

            do_perimg_aug_scale = self.aug_scale is not None and not self.shared_aug_scale_value and random.random() < self.aug_scale_ratio
            do_shared_aug_scale = self.aug_scale is not None and  self.shared_aug_scale_value
            
            if self.split == DataSplit.TRAIN and do_perimg_aug_scale:
                safe_bound = self.safe_bound + np.random.triangular(0, 0, self.aug_scale) * max(ht, wt)
            elif self.split == DataSplit.TRAIN and do_shared_aug_scale:
                safe_bound = self.safe_bound + shared_aug_scale * max(ht, wt)
            else:
                safe_bound = self.safe_bound
                
            # Need to resize the image to be larger than the target size in a safe bound
            ratio = max((ht + safe_bound) / H[i], (wt + safe_bound) / W[i])
            h, w = int(H[i] * ratio), int(W[i] * ratio)

            # Resize the image to the target size and update the intrinsic matrix
            if h != H[i] or w != W[i]:
                K[i][0:1] *= w / W[i]
                K[i][1:2] *= h / H[i]
                rgb[i] = as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_AREA))(rgb[i])
                if msk[i] is not None: msk[i] = as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_NEAREST))(msk[i])[..., None]
                if wet[i] is not None: wet[i] = as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_NEAREST))(wet[i])[..., None]
                if dpt[i] is not None: 
                    if self.sparse_interpolate: dpt[i] = as_torch_func(partial(sparse_resize, H=h, W=w))(dpt[i])[..., None]
                    else: dpt[i] = as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_NEAREST))(dpt[i])[..., None]
                if bkg[i] is not None: bkg[i] = as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_AREA))(bkg[i])
                if norm[i] is not None: norm[i] = as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_AREA))(norm[i])
                if dynamic_msk[i] is not None: dynamic_msk[i] = as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_NEAREST))(dynamic_msk[i])[..., None]
                if seg[i] is not None: seg[i] = as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_NEAREST))(seg[i])[..., None]

            # Rotate the image if specified
            if rotate:
                # Rotate the image 90 degrees clockwise or counter-clockwise
                h, w = w, h  # swap the height and width
                ht, wt = wt, ht  # swap the target height and width
                clockwise = random.random() < 0.5
                rgb[i], K[i], output.w2cs[i] = rotate_90_degree(rgb[i], K[i], output.w2cs[i].clone(), clockwise=clockwise)
                if msk[i] is not None: msk[i], _, _ = rotate_90_degree(msk[i], K[i], output.w2cs[i].clone(), clockwise=clockwise)
                if wet[i] is not None: wet[i], _, _ = rotate_90_degree(wet[i], K[i], output.w2cs[i].clone(), clockwise=clockwise)
                if dpt[i] is not None: dpt[i], _, _ = rotate_90_degree(dpt[i], K[i], output.w2cs[i].clone(), clockwise=clockwise)
                if bkg[i] is not None: bkg[i], _, _ = rotate_90_degree(bkg[i], K[i], output.w2cs[i].clone(), clockwise=clockwise)
                if norm[i] is not None: norm[i], _, _ = rotate_90_degree(norm[i], K[i], output.w2cs[i].clone(), clockwise=clockwise)
                if dynamic_msk[i] is not None: dynamic_msk[i], _, _ = rotate_90_degree(dynamic_msk[i], K[i], output.w2cs[i].clone(), clockwise=clockwise)
                if seg[i] is not None: seg[i], _, _ = rotate_90_degree(seg[i], K[i], output.w2cs[i].clone(), clockwise=clockwise)

            # Crop the image to the target size
            rgb[i], crop_h, crop_w = crop_nhwc_image(rgb[i], size=(ht, wt), center=False, strict_center=self.center_crop, K=K[i], return_offset=True)
            if msk[i] is not None: msk[i], _, _ = crop_nhwc_image(msk[i], size=(ht, wt), center=False, strict_center=self.center_crop, K=K[i], return_offset=True)
            if wet[i] is not None: wet[i], _, _ = crop_nhwc_image(wet[i], size=(ht, wt), center=False, strict_center=self.center_crop, K=K[i], return_offset=True)
            if dpt[i] is not None: dpt[i], _, _ = crop_nhwc_image(dpt[i], size=(ht, wt), center=False, strict_center=self.center_crop, K=K[i], return_offset=True)
            if bkg[i] is not None: bkg[i], _, _ = crop_nhwc_image(bkg[i], size=(ht, wt), center=False, strict_center=self.center_crop, K=K[i], return_offset=True)
            if norm[i] is not None: norm[i], _, _ = crop_nhwc_image(norm[i], size=(ht, wt), center=False, strict_center=self.center_crop, K=K[i], return_offset=True)
            if dynamic_msk[i] is not None: dynamic_msk[i], _, _ = crop_nhwc_image(dynamic_msk[i], size=(ht, wt), center=False, strict_center=self.center_crop, K=K[i], return_offset=True)
            if seg[i] is not None: seg[i], _, _ = crop_nhwc_image(seg[i], size=(ht, wt), center=False, strict_center=self.center_crop, K=K[i], return_offset=True)
            # Adjust the camera intrinsic
            K[i][0, 2] -= crop_w
            K[i][1, 2] -= crop_h

            # Update the meta data
            H[i], W[i] = ht, wt
            meta = dotdict(H=H[i], W=W[i], K=K[i])
            output.update(meta)
            output.meta.update(meta)
            output.meta.data_root = self.data_root
            output.meta.dataset_name = self.dataset_name
            output.meta.original_aspect_ratio = original_aspect_ratio

        # Pad the all the images to the same largest size in the source list
        if len(set(H)) > 1 or len(set(W)) > 1:
            # Get the largest height and width
            h, w = max(H), max(W)
            for i in range(len(rgb)):
                if H[i] != h or W[i] != w:
                    # Update the camera intrinsic
                    K[i][0, 2] += (w - W[i]) // 2
                    K[i][1, 2] += (h - H[i]) // 2
                    # Pad the image to the largest size
                    rgb[i] = fill_nhwc_image(rgb[i], size=(h, w), value=0.0, center=True)  # (H, W, 3)
                    if msk[0] is not None: msk[i] = fill_nhwc_image(msk[i], size=(h, w), value=0.0, center=True)  # (H, W, 1)
                    if wet[0] is not None: wet[i] = fill_nhwc_image(wet[i], size=(h, w), value=0.0, center=True)  # (H, W, 1)
                    if dpt[0] is not None: dpt[i] = fill_nhwc_image(dpt[i], size=(h, w), value=0.0, center=True)  # (H, W, 1)
                    if bkg[0] is not None: bkg[i] = fill_nhwc_image(bkg[i], size=(h, w), value=0.0, center=True)  # (H, W, 3)
                    if norm[0] is not None: norm[i] = fill_nhwc_image(norm[i], size=(h, w), value=0.0, center=True)  # (H, W, 3)
                    if dynamic_msk[0] is not None: dynamic_msk[i] = fill_nhwc_image(dynamic_msk[i], size=(h, w), value=0.0, center=True)  # (H, W, 1)
                    if seg[0] is not None: seg[i] = fill_nhwc_image(seg[i], size=(h, w), value=0.0, center=True)  # (H, W, 1)
                    # Update the meta data
                    meta = dotdict(H=h, W=w, K=K[i])
                    output.update(meta)
                    output.meta.update(meta)
                else:
                    continue

        # Maybe perform image augmentation
        if self.split == DataSplit.TRAIN and self.aug_image:
            if self.aug_image_cojit and random.random() > self.aug_image_cojit_ratio:
                rgb = self.augmentor(
                    torch.stack(rgb, dim=0).permute(0, 3, 1, 2),  # (S, H, W, 3) -> (S, 3, H, W)
                ).permute(0, 2, 3, 1)  # (S, 3, H, W) -> (S, H, W, 3), the same augmentation for all the views
            else:
                rgb = [self.augmentor(
                    i.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
                ).permute(1, 2, 0) for i in rgb]  # each view individually

        output.rgb = torch.stack([i.reshape(-1, 3) for i in rgb], dim=0)  # (S, H * W, 3)
        if msk[0] is not None: output.msk = torch.stack([i.reshape(-1, 1) for i in msk], dim=0)  # (S, H * W, 1)
        if wet[0] is not None: output.wet = torch.stack([i.reshape(-1, 1) for i in wet], dim=0)  # (S, H * W, 1)
        if dpt[0] is not None: output.dpt = torch.stack([i.reshape(-1, 1) for i in dpt], dim=0)  # (S, H * W, 1)
        if bkg[0] is not None: output.bkg = torch.stack([i.reshape(-1, 3) for i in bkg], dim=0)  # (S, H * W, 3)
        if norm[0] is not None: output.norm = torch.stack([i.reshape(-1, 3) for i in norm], dim=0)  # (S, H * W, 3)
        if dynamic_msk[0] is not None: output.dynamic_msk = torch.stack([i.reshape(-1, 1) for i in dynamic_msk], dim=0)  # (S, H * W, 1)
        if seg[0] is not None: output.seg = torch.stack([i.reshape(-1, 1) for i in seg], dim=0)  # (S, H * W, 1)
        output.ixts = torch.stack([i for i in K], dim=0)  # (S, 3, 3), always store the modified intrinsics
        return output

    def get_xyz(self, output: dotdict):
        # Preserve pre-normalization camera poses so exporters can recover the
        # original dataset world frame from canonical VGGT coordinates.
        output.w2cs_raw = output.w2cs.clone()
        output.c2ws_raw = affine_inverse(output.w2cs_raw)

        # Set the first camera pose to identity and transform the rest accordingly
        # Maybe setting the first camera pose to identity make it easier for training?
        c2ws = affine_inverse(output.w2cs)  # (S, 4, 4)
        c2ws = output.w2cs[:1] @ c2ws  # (S, 4, 4)
        w2cs = affine_inverse(c2ws)  # (S, 4, 4)
        # Shape things
        S = c2ws.shape[0]

        # If there is no ground truth depth, use the pre-defined
        # global scale for validation and testing
        if self.infer:
            scale = self.scale  # FIXME: maybe a better way to handle this?

        # Compute the global scale according to the ground truth depth
        else:
            # Compute the original ray origins and directions before normalization
            ray_o, ray_d = get_rays(output.H, output.W,  # (S, H, W, 3), (S, H, W, 3)
                                    output.ixts,
                                    w2cs[..., :3, :3],
                                    w2cs[..., :3, 3:],
                                    z_depth=self.use_z_depth,
                                    correct_pix=self.correct_pix)  # `self.use_z_depth = True`, without normalization
            ray_o = ray_o.reshape(S, -1, 3)  # (S, H * W, 3)
            ray_d = ray_d.reshape(S, -1, 3)  # (S, H * W, 3)

            # Convert the depth if necessary
            if self.depth_type == Depth.LINEAR:
                dpt = convert_linear_depth(output.dpt, scale=self.depth_cfg.scale)
            elif self.depth_type == Depth.INVERSE:
                dpt = convert_inverse_depth(output.dpt, **self.depth_cfg)
            else:
                raise NotImplementedError

            # Find any nan or inf values in the depth and fill them with 0
            dpt[torch.isinf(dpt) | torch.isnan(dpt)] = 0.0  # (S, H * W, 1)

            # Get the valid mask for depth smaller than the maximum depth
            # This is used to filter out invalid depth values that may influence the normalization
            if self.min_depth > 0: dpt[dpt < self.min_depth] = 0.0  # (S, H * W, 1)
            if self.max_depth > 0: dpt[dpt > self.max_depth] = 0.0  # (S, H * W, 1)

            # Fill the loaded depth mask with zero
            dpt[~(output.msk > 0)] = 0.0  # (S, H * W, 1)

            # Zero out the depth values that are smaller than the minimum depth quantiled
            if self.min_depth_quantile > 0:
                dpt_min = torch.quantile(dpt, self.min_depth_quantile)  # scalar
                if dpt_min > 0: dpt[dpt < dpt_min] = 0.0

            # Zero out the depth values that are larger than the maximum depth quantiled
            if self.max_depth_quantile > 0 and self.max_depth_quantile < 1:
                dpt_max = torch.quantile(dpt, self.max_depth_quantile)  # scalar
                if dpt_max > 0: dpt[dpt > dpt_max] = 0.0

            # Get the final valid mask
            msk = dpt > 0  # (S, H * W, 1)

            # Compute the xyz points for all the views
            xyz = ray_o + ray_d * dpt  # (S, H * W, 3)
            if self.use_world_coord: xyz = xyz
            else: xyz = xyz @ w2cs[..., :3, :3].mT + w2cs[..., :3, 3:].mT  # (S, H * W, 3)

            # Calculate the global scaling factor
            if self.scale_type == Scale.DIST:
                dists = torch.norm(xyz, dim=-1)  # (S, H * W)
                scale = ((dists * msk[..., 0]).sum() / (msk.sum() + 1e-3)).clamp(min=1e-6, max=1e6)  # scalar
            elif self.scale_type == Scale.XYZ:
                scale = torch.max(torch.abs(xyz[msk[..., 0]]))  # scalar
            elif self.scale_type == Scale.DEPTH:
                scale = min(dpt_max, self.max_depth)  # scalar
            elif self.scale_type == Scale.MANUAL:
                scale = torch.tensor(self.scale, device=xyz.device)
            else:
                raise NotImplementedError

            # Normalize the xyz and corresponding depth
            output.xyz = xyz / scale  # (S, H * W, 3)
            output.dpt = dpt / scale  # (S, H * W, 1)

        # Record the mask and the global scale
        output.msk = msk  # (S, H * W, 1)
        output.scale = scale

        # Normalize the camera poses using the global scale
        c2ws[..., :3, 3:] = c2ws[..., :3, 3:] / scale
        w2cs = affine_inverse(c2ws)
        output.w2cs = w2cs  # (S, 4, 4)
        output.c2ws = c2ws  # (S, 4, 4)

        # Encode the camera parameters, follow VGGT format
        # https://github.com/facebookresearch/vggt/blob/main/vggt/utils/pose_enc.py
        cam = encode_camera_params(w2cs, output.ixts, output.H, output.W)
        output.cam = cam  # (S, 9)
        
        if self.use_3ddr:
            # Set the first camera pose to identity and transform the rest accordingly
            c2ws_3ddr = affine_inverse(output.w2cs_3ddr)
            c2ws_3ddr = output.w2cs_3ddr[:1] @ c2ws_3ddr
            w2cs_3ddr = affine_inverse(c2ws_3ddr)
            # Normalize the camera poses using the global scale
            c2ws_3ddr[..., :3, 3:] = c2ws_3ddr[..., :3, 3:] / scale
            w2cs_3ddr = affine_inverse(c2ws_3ddr)
            output.w2cs_3ddr = w2cs_3ddr  # (S, 4, 4)
            output.c2ws_3ddr = c2ws_3ddr  # (S, 4, 4)
            # Encode the camera parameters, follow VGGT format
            cam_3ddr = encode_camera_params(w2cs_3ddr, output.ixts, output.H, output.W)
            output.cam_3ddr = cam_3ddr  # (S, 9)

        # only for relative pose
        if "covis_pairs" in output:
            # intrin
            output.focal = cam[:, 7:]  # (S, 2)
            covis_pairs = output.covis_pairs.view(-1)
            rel_focal_1 = cam[covis_pairs[::2], 7:9]
            rel_focal_2 = cam[covis_pairs[1::2], 7:9]
            rel_c2cs = w2cs[covis_pairs]
            # rel pose: w2c0 @ w2c1^-1 = c1_2_c0
            rel_c2cs = rel_c2cs[::2] @ rel_c2cs[1::2].inverse()            
            rel_cam = encode_camera_params(rel_c2cs, None, None, None, "abs_quat")
            rel_cam = torch.cat([rel_cam, rel_focal_1, rel_focal_2], dim=-1)
            output.rel_cam = rel_cam  # (S_pairs, 7)

        # Calculate the ray origins and directions after normalization
        ray_o, ray_d = get_rays(output.H, output.W,
                                output.ixts,
                                output.w2cs[..., :3, :3],
                                output.w2cs[..., :3, 3:],
                                z_depth=self.use_z_depth,
                                correct_pix=self.correct_pix)
        output.ray_o = ray_o.reshape(S, -1, 3)  # (S, H * W, 3)
        output.ray_d = ray_d.reshape(S, -1, 3)  # (S, H * W, 3)

        # Compute the ray map, plucker embedding for now
        output.ray_m = get_plucker(output.ray_o, output.ray_d)  # (S, H * W, 6)

        return output

    def get_ground_truth(self, index):
        # Load actual images, mask
        output = self.get_metadata(index)
        return output
