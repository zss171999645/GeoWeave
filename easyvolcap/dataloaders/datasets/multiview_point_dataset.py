import cv2
import torch
import random
import numpy as np
from functools import lru_cache
from typing import List, Dict, Literal, Union
from scipy.spatial.transform import Rotation

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
from easyvolcap.utils.depth_utils import (
    Depth,
    Scale,
    convert_linear_depth,
    convert_inverse_depth,
    depth_specific_quantile,
)
from easyvolcap.utils.cam_utils import (Sourcing, 
                                        compute_camera_similarity,
                                        compute_camera_zigzag_similarity, 
                                        compute_camera_sequential_similarity, 
                                        compute_camera_dist_rot_similarity, 
                                        encode_camera_params,
                                        add_noise_to_poses)
from easyvolcap.utils.multiview_utils import compute_mvseq_inds, compute_mvseq_inds_with_loopclose, compute_mvseqv2_inds

MVSEQ_TYPES = (Sourcing.SEQUENTIAL, Sourcing.MULTIVIEWSEQ, Sourcing.MULTIVIEWSEQV2, Sourcing.MULTIVIEWSEQLC)
MVSEQ_ONLY_TYPES = (Sourcing.MULTIVIEWSEQ, Sourcing.MULTIVIEWSEQV2, Sourcing.MULTIVIEWSEQLC)


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
class MultiviewPointDataset(VolumetricVideoDataset):
    def __init__(self,
                 n_srcs_list: List[int] = [16],  # MARK: repeated global configuration
                 n_srcs_prob: List[float] = [1.0],  # MARK: repeated global configuration
                 extra_src_pool: int = 0,
                 extra_src_pool_ratio: float = -1,  # ratio of extra source pool relative to the number of source views
                 nearby_sampling: bool = False,  # random sample then constrain to a temporal neighborhood
                 nearby_expand_ratio: float = -1.0,  # window size = (n_srcs + 1) * nearby_expand_ratio
                 nearby_expand_range: int = -1,  # fixed window size, overrides nearby_expand_ratio if > 0
                 nearby_allow_duplicate: bool = True,  # allow duplicate frames when sampling nearby
                 sequential_stride: int = 1,  # temporal stride used by SEQUENTIAL source selection
                 ref_view_type: Optional[Literal["random_ref_view", "front_ref_view", "random_front_ref_view"]] = "front_ref_view",
                 closest_using_t: bool = False,  # find the closest view using the temporal dimension
                 source_type: str = Sourcing.DISTANCEROT.name,  # Sourcing.DISTANCE or Sourcing.ZIGZAG or Sourcing.SEQUENTIAL or Sourcing.RANDOM or Sourcing.DISTANCEROT
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
                 # VGGT official evaluation preprocessing (facebookresearch/vggt)
                 # See: vggt/utils/load_fn.py::load_and_preprocess_images(mode="crop"/"pad")
                 vggt_official_preprocess: bool = False,
                 vggt_official_preprocess_mode: Literal["crop", "pad"] = "crop",
                 vggt_official_target_size: int = 518,

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
                 **kwargs,
                 ):
        # Ignore things, since this will serve as a base class of classes supporting *args and **kwargs
        # The inspection of registration and config system only goes down one layer
        # Otherwise it would be to inefficient
        # Drop meta-only keys before initializing the base dataset to avoid unused-cfg warnings.
        kwargs.pop("prob", None)
        kwargs.pop("dataset_idx", None)
        call_from_cfg(super().__init__, kwargs, disk_dataset=disk_dataset, use_z_depth=use_z_depth)

        self.closest_using_t = closest_using_t
        self.src_view_sample = src_view_sample
        assert not self.closest_using_t or self.frame_sample == [0, None, 1] or force_sparse_view, "Should use default frame_sample [0, None, 1] for ibr dataset with `closest_using_t`. Control sampling through sampler.frame_sample and src_view_sample"
        assert self.view_sample == [0, None, 1] or force_sparse_view, "Should use default view_sample [0, None, 1] for ibr dataset. Control sampling through sampler.view_sample and src_view_sample"
        assert not (self.cache_raw and not supply_decoded), "Will always supply decoded source images when cache_raw is enabled for faster sampling, set cache_raw to False to supply jpeg streams"

        self.source_type = Sourcing[source_type]
        # Views are selected and loaded
        # Frames are selected and loaded
        self.load_source_params()
        self.select_source_function()
        # Need to build all possible view selections (distance of c2w)
        # - Dot product of v_front - euclidian distance of center
        # NOTE: Only compute when disk_dataset is False to avoid large scale matrix OOM
        if not self.disk_dataset:
            self.load_source_indices()

        self.n_srcs_list = n_srcs_list if n_srcs_list is None or len(n_srcs_list) != 1 or n_srcs_list[0] != 0 else [self.n_views]
        self.n_srcs_prob = n_srcs_prob
        self.extra_src_pool = extra_src_pool
        self.extra_src_pool_ratio = extra_src_pool_ratio
        self.nearby_sampling = nearby_sampling
        self.nearby_expand_ratio = nearby_expand_ratio
        self.nearby_expand_range = nearby_expand_range
        self.nearby_allow_duplicate = nearby_allow_duplicate
        self.sequential_stride = max(int(sequential_stride), 1)
        assert not (self.extra_src_pool > 0 and self.extra_src_pool_ratio > 0), "Cannot set both extra_src_pool and extra_src_pool_ratio, choose one of them"
        self.ref_view_type = ref_view_type
        
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
        self.vggt_official_preprocess = vggt_official_preprocess
        self.vggt_official_preprocess_mode = vggt_official_preprocess_mode
        self.vggt_official_target_size = int(vggt_official_target_size)

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
        self._warned_missing_depth = False

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
                self.src_exts_3ddr_inv = affine_padding(self.c2ws_3ddr[:, view_inds])
                self.src_exts_3ddr_inv = self.src_exts_3ddr_inv.permute(1, 0, 2, 3)
        else:
            self.src_ixts = self.Ks[view_inds]  # N, L, 4, 4
            self.src_exts = affine_padding(self.w2cs[view_inds])  # N, L, 4, 4

    def select_source_function(self):
        # Set the default source similarity and indices to None
        self.src_sims, self.src_inds = None, None

        # Select the source function
        if self.source_type == Sourcing.DISTANCE:
            self.source_func = compute_camera_similarity
        elif self.source_type == Sourcing.DISTANCEROT:
            self.source_func = compute_camera_dist_rot_similarity
        elif self.source_type == Sourcing.ZIGZAG:
            self.source_func = compute_camera_zigzag_similarity
        elif self.source_type in MVSEQ_TYPES:
            assert self.closest_using_t, f"Multi-view/Single-view sequential selection only works when closest_using_t is True"
            self.source_func = compute_camera_sequential_similarity  
        elif self.source_type == Sourcing.RANDOM:
            self.source_func = lambda *args, **kwargs: (None, None)
        else:
            raise NotImplementedError

    def load_source_indices(self):
        # Get the target views and source views
        tar_c2ws = self.c2ws.permute(1, 0, 2, 3) if self.closest_using_t else self.c2ws  # MARK: transpose
        src_c2ws = affine_inverse(self.src_exts)

        # Source view index and there similarity
        self.src_sims, self.src_inds = self.source_func(tar_c2ws, src_c2ws)

    def get_metadata(self, index: dotdict):
        scene_pool = None
        if isinstance(index, dotdict):
            scene_pool = index.get('scene_pool', None)
            index, n_srcs, aspect_ratio = index.index, index.n_srcs, index.get('aspect_ratio', -1)
        else:
            n_srcs, aspect_ratio = random.choices(self.n_srcs_list, self.n_srcs_prob)[0], -1

        if self.source_type in MVSEQ_ONLY_TYPES:
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
            src_exts_3ddr_inv = self.src_exts_3ddr_inv.clone()

        # Sample the whole local window, including the target view itself
        # NOTE: DO NOT distinguish between the target and source view here, that's IBR thing, here is MVS
        # Sample the source views according to the similarity
        if scene_pool is not None and len(scene_pool):
            # Paper protocol: the wrapper dataset may provide a fixed per-scene pool of indices.
            # Select sources deterministically from that pool (exclude the target which is always included).
            pool = []
            for i in scene_pool:
                try:
                    ii = int(i)
                except Exception:
                    continue
                if 0 <= ii < src_exts.shape[0] and ii != target_index:
                    pool.append(ii)

            # Match requested view count (n_srcs + 1). If pool is short, fill from the remaining indices.
            if len(pool) < n_srcs:
                rest = [i for i in range(src_exts.shape[0]) if i != target_index and i not in pool]
                pool = pool + rest[:max(0, n_srcs - len(pool))]
            if len(pool) < n_srcs:
                pool = pool + [pool[-1] if pool else target_index] * (n_srcs - len(pool))

            inds = torch.as_tensor([target_index] + pool[:n_srcs])  # (1 + S,)

        elif self.nearby_sampling:
            if not self.closest_using_t:
                raise ValueError("nearby_sampling requires closest_using_t=True")
            full_len = src_exts.shape[0]
            total_imgs = n_srcs + 1
            if self.nearby_expand_range > 0:
                expand_range = int(self.nearby_expand_range)
            else:
                expand_ratio = self.nearby_expand_ratio if self.nearby_expand_ratio > 0 else 2.0
                expand_range = int(total_imgs * expand_ratio)
            low = max(0, target_index - expand_range)
            high = min(full_len, target_index + expand_range)
            pool = list(range(low, high))
            if target_index in pool:
                pool.remove(target_index)
            if not pool:
                inds = []
            elif self.nearby_allow_duplicate or n_srcs >= len(pool):
                inds = random.choices(pool, k=n_srcs)
            else:
                inds = random.sample(pool, n_srcs)
        elif self.source_type != Sourcing.RANDOM:
            if self.extra_src_pool_ratio > 0: random_ap = int(self.extra_src_pool_ratio * n_srcs)
            else: random_ap = self.extra_src_pool  # training -> randomly sample more
            if not self.disk_dataset:
                inds = self.src_inds[target_index, 1:1 + n_srcs + random_ap, extra_index].numpy().tolist()  # always exclude the target view, (S + random_ap,)
            else:
                if self.source_type in MVSEQ_ONLY_TYPES:
                    # TODO: find a better way to do this, it is not elegant...
                    # Select the closest view along the temporal dimension centered at the target view
                    n_srcs_sample = 0  # use random_ap to sample temporal inds
                    sidx = max(0, target_index - (n_srcs_sample + random_ap) // 2)
                    inds = list(range(sidx, min(sidx + n_srcs_sample + random_ap + 1, len(src_exts))))  # (random_ap + 1,)
                    inds.remove(target_index)  # (S + random_ap,), remove itself
                elif self.source_type != Sourcing.SEQUENTIAL:
                    tar_c2ws = self.c2ws[extra_index, target_index][None] if self.closest_using_t else self.c2ws[target_index, extra_index][None]  # (1, 4, 4)
                    src_c2ws = affine_inverse(src_exts[:, extra_index])  # (N, 4, 4)
                    src_sims, src_inds = self.source_func(tar_c2ws, src_c2ws)  # (1, N), (1, N)
                    
                    inds = src_inds[0, 1:1 + n_srcs + random_ap].numpy().tolist()  # always exclude the target view, (S + random_ap,)
                else:
                    # Select a centered temporal window and optionally enlarge the
                    # spacing between adjacent source frames for long-range sweeps.
                    total_count = n_srcs + random_ap + 1
                    sparse_pool = list(range(0, len(src_exts), self.sequential_stride))
                    if target_index not in sparse_pool:
                        sparse_pool.append(target_index)
                        sparse_pool.sort()
                    target_rank = sparse_pool.index(target_index)
                    start_rank = max(0, target_rank - (total_count - 1) // 2)
                    end_rank = min(len(sparse_pool), start_rank + total_count)
                    start_rank = max(0, end_rank - total_count)
                    inds = sparse_pool[start_rank:end_rank]
                    if len(inds) < total_count:
                        rest = [i for i in range(len(src_exts)) if i not in inds]
                        inds.extend(rest[:max(0, total_count - len(inds))])
                    inds.remove(target_index)  # (S + random_ap,), remove itself
                    
            if random_ap:
                if self.source_type in MVSEQ_ONLY_TYPES:
                    n_cam = len(self) // self.n_latents
                    src_inds = inds.copy()
                else:
                    if n_srcs < len(src_exts): inds = random.sample(inds, n_srcs)  # (S,), no replacement
                    else: inds = random.choices(inds, k=n_srcs)  # (S,), with replacement
        # No similarity, just randomly sample
        else:
            # Need to exclude the current index from the sampling to avoid sampling the same index multiple times
            pool = list(range(src_exts.shape[0]))
            pool.remove(target_index)
            if n_srcs < len(pool): inds = random.sample(pool, n_srcs)  # (S,), no replacement
            else: inds = random.choices(pool, k=n_srcs)  # (S,), with replacement
        # Always include the target index
        if not isinstance(inds, torch.Tensor):
            inds = torch.as_tensor([target_index] + inds)  # (1 + S,)

        if scene_pool is None and self.source_type == Sourcing.MULTIVIEWSEQ:
            extra_index, inds = compute_mvseq_inds(n_cam, src_inds, target_index, n_srcs, self.split == DataSplit.TRAIN, ref_view_type=self.ref_view_type)
        elif scene_pool is None and self.source_type == Sourcing.MULTIVIEWSEQV2:
            extra_index, inds = compute_mvseqv2_inds(n_cam, src_inds, target_index, n_srcs, self.split == DataSplit.TRAIN)
        elif scene_pool is None and self.source_type == Sourcing.MULTIVIEWSEQLC:
            assert self.use_3ddr, "Loop closure only works with 3DDR dataset"
            if self.use_3ddr:
                extra_index, inds = compute_mvseq_inds_with_loopclose(n_cam, src_inds, target_index, n_srcs, src_exts_3ddr_inv, self.split == DataSplit.TRAIN, ref_view_type=self.ref_view_type)

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
        packed = parallel_execution(view_index, latent_index, output=output, action=self.get_image, sequential=True, num_workers=self.dataloading_workers)
        if len(packed) == 0:
            raise RuntimeError("get_sources received empty samples from get_image")

        # Keep compatibility with both signatures:
        # - legacy: (im, mk, dynamic_mk, seg, wt, dp, bg, nm, H, W, K)
        # - newer:  (im, mk, dynamic_mk, seg, wt, dp, bg, nm, H, W, K, im_ts)
        n_fields = len(packed[0])
        if n_fields == 11:
            rgb, msk, dynamic_msk, seg, wet, dpt, bkg, norm, H, W, K = zip(*packed)
            im_ts = None
        elif n_fields == 12:
            rgb, msk, dynamic_msk, seg, wet, dpt, bkg, norm, H, W, K, im_ts = zip(*packed)
        else:
            raise ValueError(f"Unexpected get_image return length: {n_fields}, expected 11 or 12")

        # NOTE: Extremely important to clone the tensor in the list to avoid modifying the original tensor
        clone = lambda x: [i.clone() if isinstance(i, torch.Tensor) else i for i in x]
        rgb, msk, dynamic_msk, seg, wet, dpt, bkg, norm, H, W, K = clone(rgb), clone(msk), clone(dynamic_msk), clone(seg), clone(wet), clone(dpt), clone(bkg), clone(norm), clone(H), clone(W), clone(K)
        if im_ts is not None:
            im_ts = clone(im_ts)

        def _sparse_resize_depth(depth, target_h: int, target_w: int):
            # Some scenes can return depth as nested python lists; normalize to ndarray first.
            if isinstance(depth, torch.Tensor):
                depth_np = depth.detach().cpu().numpy()
            else:
                depth_np = np.asarray(depth)
            if isinstance(depth_np, np.ndarray) and depth_np.ndim >= 3 and depth_np.shape[-1] == 1:
                depth_np = depth_np[..., 0]
            depth_resized = sparse_resize(depth_np, H=target_h, W=target_w)
            # Use direct tensor conversion here to avoid torchvision/PIL helper paths
            # returning nested Python lists for sparse depth maps.
            depth_t = torch.as_tensor(np.asarray(depth_resized))
            if depth_t.ndim == 2:
                depth_t = depth_t[..., None]
            return depth_t

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
            if msk[0] is not None:
                msk = [as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_NEAREST))(i) for i in msk]
                msk = [i[..., None] if i.ndim == 2 else i for i in msk]
            if wet[0] is not None:
                wet = [as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_NEAREST))(i) for i in wet]
                wet = [i[..., None] if i.ndim == 2 else i for i in wet]
            if dpt[0] is not None: 
                if self.sparse_interpolate:
                    dpt = [_sparse_resize_depth(i, h, w) for i in dpt]
                else:
                    dpt = [as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_NEAREST))(i) for i in dpt]
                dpt = [i[..., None] if i.ndim == 2 else i for i in dpt]
            if bkg[0] is not None: bkg = [as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_AREA))(i) for i in bkg]
            if norm[0] is not None: norm = [as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_AREA))(i) for i in norm]
            if dynamic_msk[0] is not None:
                dynamic_msk = [as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_NEAREST))(i) for i in dynamic_msk]
                dynamic_msk = [i[..., None] if i.ndim == 2 else i for i in dynamic_msk]
            if seg[0] is not None:
                seg = [as_torch_func(partial(cv2.resize, dsize=(w, h), interpolation=cv2.INTER_NEAREST))(i) for i in seg]
                seg = [i[..., None] if i.ndim == 2 else i for i in seg]
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

        # VGGT official eval preprocessing (crop/pad + white padding).
        # This is used to reproduce VGGT paper Table 1 results on RE10K/CO3Dv2.
        if self.vggt_official_preprocess:
            mode = (self.vggt_official_preprocess_mode or "crop").strip().lower()
            if mode not in {"crop", "pad"}:
                raise ValueError(f"Unknown vggt_official_preprocess_mode={self.vggt_official_preprocess_mode!r}, expected crop|pad.")

            target = int(self.vggt_official_target_size)
            div = 14  # VGGT uses patch_size=14
            original_aspect_ratio = float(H[0]) / float(W[0]) if int(W[0]) > 0 else -1.0

            def _resize_nhwc(im: torch.Tensor, new_h: int, new_w: int, interp: int):
                return as_torch_func(partial(cv2.resize, dsize=(new_w, new_h), interpolation=interp))(im)

            # Per-image resize (+ optional crop/pad)
            for i in range(len(rgb)):
                h0, w0 = int(H[i]), int(W[i])
                if h0 <= 0 or w0 <= 0:
                    raise ValueError(f"Invalid image size: H={h0}, W={w0}, view={i}, root={self.data_root}")

                if mode == "pad":
                    if w0 >= h0:
                        new_w = target
                        new_h = round((h0 * (new_w / w0)) / div) * div
                    else:
                        new_h = target
                        new_w = round((w0 * (new_h / h0)) / div) * div
                else:
                    new_w = target
                    new_h = round((h0 * (new_w / w0)) / div) * div

                new_h = max(div, int(new_h))
                new_w = max(div, int(new_w))

                if new_h != h0 or new_w != w0:
                    K[i][0:1] *= new_w / w0
                    K[i][1:2] *= new_h / h0
                    rgb[i] = _resize_nhwc(rgb[i], new_h, new_w, cv2.INTER_CUBIC)
                    if msk[i] is not None: msk[i] = _resize_nhwc(msk[i], new_h, new_w, cv2.INTER_NEAREST)[..., None]
                    if wet[i] is not None: wet[i] = _resize_nhwc(wet[i], new_h, new_w, cv2.INTER_NEAREST)[..., None]
                    if dpt[i] is not None: dpt[i] = _resize_nhwc(dpt[i], new_h, new_w, cv2.INTER_NEAREST)[..., None]
                    if bkg[i] is not None: bkg[i] = _resize_nhwc(bkg[i], new_h, new_w, cv2.INTER_CUBIC)
                    if norm[i] is not None: norm[i] = _resize_nhwc(norm[i], new_h, new_w, cv2.INTER_CUBIC)
                    if dynamic_msk[i] is not None: dynamic_msk[i] = _resize_nhwc(dynamic_msk[i], new_h, new_w, cv2.INTER_NEAREST)[..., None]
                    if seg[i] is not None: seg[i] = _resize_nhwc(seg[i], new_h, new_w, cv2.INTER_NEAREST)[..., None]

                # Center crop height if needed (crop mode)
                if mode == "crop" and new_h > target:
                    start_y = (new_h - target) // 2
                    rgb[i] = rgb[i][start_y:start_y + target, :, :]
                    if msk[i] is not None: msk[i] = msk[i][start_y:start_y + target, :, :]
                    if wet[i] is not None: wet[i] = wet[i][start_y:start_y + target, :, :]
                    if dpt[i] is not None: dpt[i] = dpt[i][start_y:start_y + target, :, :]
                    if bkg[i] is not None: bkg[i] = bkg[i][start_y:start_y + target, :, :]
                    if norm[i] is not None: norm[i] = norm[i][start_y:start_y + target, :, :]
                    if dynamic_msk[i] is not None: dynamic_msk[i] = dynamic_msk[i][start_y:start_y + target, :, :]
                    if seg[i] is not None: seg[i] = seg[i][start_y:start_y + target, :, :]
                    K[i][1, 2] -= start_y
                    new_h = target

                # Square padding (pad mode)
                if mode == "pad" and (new_h != target or new_w != target):
                    pad_top = (target - new_h) // 2
                    pad_left = (target - new_w) // 2
                    K[i][0, 2] += pad_left
                    K[i][1, 2] += pad_top
                    rgb[i] = fill_nhwc_image(rgb[i], size=(target, target), value=1.0, center=True)
                    if msk[i] is not None: msk[i] = fill_nhwc_image(msk[i], size=(target, target), value=0.0, center=True)
                    if wet[i] is not None: wet[i] = fill_nhwc_image(wet[i], size=(target, target), value=0.0, center=True)
                    if dpt[i] is not None: dpt[i] = fill_nhwc_image(dpt[i], size=(target, target), value=0.0, center=True)
                    if bkg[i] is not None: bkg[i] = fill_nhwc_image(bkg[i], size=(target, target), value=1.0, center=True)
                    if norm[i] is not None: norm[i] = fill_nhwc_image(norm[i], size=(target, target), value=0.0, center=True)
                    if dynamic_msk[i] is not None: dynamic_msk[i] = fill_nhwc_image(dynamic_msk[i], size=(target, target), value=0.0, center=True)
                    if seg[i] is not None: seg[i] = fill_nhwc_image(seg[i], size=(target, target), value=0.0, center=True)
                    new_h, new_w = target, target

                H[i], W[i] = int(new_h), int(new_w)

            # Batch padding to max size (crop mode, multiple shapes)
            max_h, max_w = max(H), max(W)
            if len(set(H)) > 1 or len(set(W)) > 1:
                for i in range(len(rgb)):
                    if H[i] == max_h and W[i] == max_w:
                        continue
                    pad_top = (max_h - H[i]) // 2
                    pad_left = (max_w - W[i]) // 2
                    K[i][0, 2] += pad_left
                    K[i][1, 2] += pad_top
                    rgb[i] = fill_nhwc_image(rgb[i], size=(max_h, max_w), value=1.0, center=True)
                    if msk[i] is not None: msk[i] = fill_nhwc_image(msk[i], size=(max_h, max_w), value=0.0, center=True)
                    if wet[i] is not None: wet[i] = fill_nhwc_image(wet[i], size=(max_h, max_w), value=0.0, center=True)
                    if dpt[i] is not None: dpt[i] = fill_nhwc_image(dpt[i], size=(max_h, max_w), value=0.0, center=True)
                    if bkg[i] is not None: bkg[i] = fill_nhwc_image(bkg[i], size=(max_h, max_w), value=1.0, center=True)
                    if norm[i] is not None: norm[i] = fill_nhwc_image(norm[i], size=(max_h, max_w), value=0.0, center=True)
                    if dynamic_msk[i] is not None: dynamic_msk[i] = fill_nhwc_image(dynamic_msk[i], size=(max_h, max_w), value=0.0, center=True)
                    if seg[i] is not None: seg[i] = fill_nhwc_image(seg[i], size=(max_h, max_w), value=0.0, center=True)
                    H[i], W[i] = max_h, max_w

            # Update meta
            output.H, output.W = int(max_h), int(max_w)
            output.meta.H, output.meta.W = int(max_h), int(max_w)
            output.meta.data_root = self.data_root
            output.meta.dataset_name = self.dataset_name
            output.meta.original_aspect_ratio = original_aspect_ratio

            output.rgb = torch.stack([im.reshape(-1, 3) for im in rgb], dim=0)
            if msk[0] is not None: output.msk = torch.stack([im.reshape(-1, 1) for im in msk], dim=0)
            if wet[0] is not None: output.wet = torch.stack([im.reshape(-1, 1) for im in wet], dim=0)
            if dpt[0] is not None: output.dpt = torch.stack([im.reshape(-1, 1) for im in dpt], dim=0)
            if bkg[0] is not None: output.bkg = torch.stack([im.reshape(-1, 3) for im in bkg], dim=0)
            if norm[0] is not None: output.norm = torch.stack([im.reshape(-1, 3) for im in norm], dim=0)
            if dynamic_msk[0] is not None: output.dynamic_msk = torch.stack([im.reshape(-1, 1) for im in dynamic_msk], dim=0)
            if seg[0] is not None: output.seg = torch.stack([im.reshape(-1, 1) for im in seg], dim=0)
            output.ixts = torch.stack([ixt for ixt in K], dim=0)
            return output
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
                    if self.sparse_interpolate: dpt[i] = _sparse_resize_depth(dpt[i], h, w)
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
        if im_ts is not None:
            output.im_ts = torch.stack([i if isinstance(i, torch.Tensor) else torch.as_tensor(i) for i in im_ts], dim=0)
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
        has_depth = hasattr(output, 'dpt') and output.dpt is not None

        # For inference-only or depth-free datasets (e.g. pose-only benchmarks),
        # fall back to a fixed global scale and keep a valid mask for camera metrics.
        if self.infer or not has_depth:
            scale = torch.as_tensor(self.scale, device=c2ws.device, dtype=c2ws.dtype)
            if hasattr(output, 'msk') and output.msk is not None:
                msk = output.msk > 0
                if not self.use_masks and not torch.any(msk):
                    if not hasattr(self, '_warned_empty_unmasked_output_msk'):
                        self._warned_empty_unmasked_output_msk = False
                    if not self._warned_empty_unmasked_output_msk:
                        log(yellow(
                            'MultiviewPointDataset.get_xyz: output.msk is empty while use_masks=False; '
                            'fallback to all-ones mask for camera-only normalization.'
                        ))
                        self._warned_empty_unmasked_output_msk = True
                    msk = torch.ones_like(msk, dtype=torch.bool)
            else:
                n_pix = int(output.H) * int(output.W)
                msk = torch.ones((S, n_pix, 1), dtype=torch.bool, device=c2ws.device)

            if not has_depth and not self._warned_missing_depth:
                log(yellow('MultiviewPointDataset.get_xyz: missing depth, fallback to camera-only normalization.'))
                self._warned_missing_depth = True

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

            # Fill the loaded depth mask with zero.
            # Some evaluation-only pipelines intentionally disable mask loading
            # to avoid dataset-side unreadable mask files. In that case, use an
            # all-ones mask fallback so depth normalization still works.
            if hasattr(output, 'msk') and output.msk is not None:
                dpt[~(output.msk > 0)] = 0.0  # (S, H * W, 1)
            else:
                if not hasattr(self, '_warned_missing_mask_for_depth_norm'):
                    self._warned_missing_mask_for_depth_norm = False
                if not self._warned_missing_mask_for_depth_norm:
                    log(yellow(
                        'MultiviewPointDataset.get_xyz: output.msk missing while depth is enabled; '
                        'fallback to depth-only valid mask.'
                    ))
                    self._warned_missing_mask_for_depth_norm = True

            dpt_valid = dpt[dpt > 0]
            if dpt_valid.numel() == 0:
                pass
            else:
                # Zero out the depth values that are smaller than the minimum depth quantiled
                dpt_min = depth_specific_quantile(dpt_valid, self.min_depth_quantile)  # scalar
                if self.min_depth_quantile > 0 and dpt_min > 0: dpt[dpt < dpt_min] = 0.0

                # Zero out the depth values that are larger than the maximum depth quantiled
                dpt_max = depth_specific_quantile(dpt_valid, self.max_depth_quantile)  # scalar
                if self.max_depth_quantile > 0 and dpt_max > 0: dpt[dpt > dpt_max] = 0.0

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
