from typing_extensions import deprecated
import os
import torch
import random
import numpy as np
import collections

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
from easyvolcap.engine import DATASAMPLERS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.dist_utils import get_rank, get_distributed, get_world_size
from torch.utils.data.sampler import RandomSampler, BatchSampler, SequentialSampler, Sampler
from easyvolcap.dataloaders.datasets.volumetric_video_dataset import VolumetricVideoDataset
from easyvolcap.dataloaders.datasets.volumetric_video_inference_dataset import VolumetricVideoInferenceDataset


def _sampler_log_enabled() -> bool:
    value = os.getenv("EVC_SAMPLER_LOG", "")
    if not value:
        return False
    return value.lower() in ("1", "true", "yes", "y", "on")


@DATASAMPLERS.register_module()
class BatchSampler(BatchSampler):
    def __init__(self,
                 sampler: Sampler,
                 batch_size: int = 8,
                 drop_last: bool = False,  # when training, will use random sampler, this doesn't matter anymore
                 *arg,
                 **kwargs,
                 ):
        super().__init__(sampler,
                         batch_size,
                         drop_last)  # strange naming


@deprecated("Use VGGTBatchSampler instead in training.")
@DATASAMPLERS.register_module()
class ImageBasedBatchSampler(BatchSampler):
    # WARNING: Deprecated, use VGGTBatchSampler instead
    # VGGTBatchSampler is a more advanced version of this sampler, which is more flexible
    # 
    # This datasampler geneartes a new 'n_imgs' in keys passed into the dataset
    # making sure all dataset instances across different processes returns the same number of
    # source images for collating together a batched input, while maintaining the ability
    # to introduce per-iter randomness in the number of images sampled
    def __init__(self,
                 sampler: Union[Sampler[int], Iterable[int]],
                 batch_size: int = 8,
                 drop_last: bool = False,
                 n_srcs_list: List[int] = [2, 3, 4],
                 n_srcs_prob: List[int] = [0.2, 0.6, 0.2],
                 aspect_ratio_range: Optional[Union[List[float], float, int]] = -1,
                 fix_random: bool = False,
                 seed: Optional[int] = 42,
                 ) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.n_srcs_list = n_srcs_list
        self.n_srcs_prob = n_srcs_prob
        self.aspect_ratio_range = aspect_ratio_range
        self.fix_random = fix_random

        # Maybe fix the random seed?
        if fix_random:
            if _sampler_log_enabled():
                log(yellow(f"batchsampler: rank {get_rank()}: fixing ImageBasedBatchSampler random seed to {seed} (nsrc & aspect ratio)"))
            self.rng = random.Random(seed)
        else:
            if _sampler_log_enabled():
                log(yellow(f"batchsampler: rank {get_rank()}: no random seed for ImageBasedBatchSampler (nsrc & aspect ratio)"))
            self.rng = random.Random()

    def __iter__(self):
        # Use shared number of images for batching
        iterator = super().__iter__()
        for batch in iterator:
            n_srcs = self.rng.choices(self.n_srcs_list, self.n_srcs_prob)[0]  # may allow this random among different rank
            if isinstance(self.aspect_ratio_range, (float, int)): aspect_ratio = self.aspect_ratio_range
            else: aspect_ratio = round(self.rng.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1]), 2)
            batch = [dotdict(index=i, n_srcs=n_srcs, aspect_ratio=aspect_ratio) for i in batch]  # expand indices to dotdict
            yield batch


@DATASAMPLERS.register_module()
class VGGTBatchSampler(BatchSampler):
    # This datasampler geneartes a new 'n_imgs' in keys passed into the dataset
    # making sure all dataset instances across different processes returns the same number of
    # source images for collating together a batched input, while maintaining the ability
    # to introduce per-iter randomness in the number of images sampled
    # Please use this sampler only for training only;
    def __init__(self,
                 sampler: Union[Sampler[int], Iterable[int]],
                 batch_size: int = 8,
                 drop_last: bool = False,
                 n_frames_batch: Optional[int] = None,
                 inside_random: bool = False,
                 n_srcs_list: List[int] = [2, 3, 4],
                 n_srcs_prob: List[int] = [0.2, 0.6, 0.2],
                 aspect_ratio_range: Optional[Union[List[float], float, int]] = -1,
                 fix_random_n_srcs: bool = False,
                 fix_random_aspect: bool = True,
                 seed: Optional[int] = 42,
                 ) -> None:
        super().__init__(sampler, batch_size, drop_last)
        # We are sampling a total of `n_frames_batch` frames, namely
        # `batch_size * n_srcs` frames, where `n_srcs` is the number of source images
        if n_frames_batch is None:
            if not n_srcs_list:
                raise ValueError("n_srcs_list cannot be empty when inferring n_frames_batch")
            n_frames_batch = max(n_srcs_list) + 1
        self.n_frames_batch = int(n_frames_batch)
        self.inside_random = inside_random

        # source view and aspect ratio sampling
        self.n_srcs_list = n_srcs_list
        self.n_srcs_prob = n_srcs_prob
        self.aspect_ratio_range = aspect_ratio_range

        # Maybe fix the random seed across different ranks
        self.fix_random_n_srcs = fix_random_n_srcs
        self.fix_random_aspect = fix_random_aspect

        if fix_random_aspect or fix_random_n_srcs:
            if _sampler_log_enabled():
                log(yellow(f"batchsampler: set random seed for aspect/nsrc in VGGTBatchSampler {seed}"))
            self.rng = random.Random(seed)
        else:
            if _sampler_log_enabled():
                log(yellow(f"batchsampler: no random seed for aspect/nsrc in VGGTBatchSampler"))
            self.rng = random.Random()

        if _sampler_log_enabled():
            log(yellow('VGGTBatchSampler\'s iterator will be recreated every iteration. Please do not use this in validation/testing. It will be fixed soon'))

    def __iter__(self):
        # Get the iterator of the sampler
        sampler_iter = iter(self.sampler)

        # Randomly sample the number of source across the batch
        if self.fix_random_n_srcs: n_srcs = self.rng.choices(self.n_srcs_list, self.n_srcs_prob)[0]
        else: n_srcs = random.choices(self.n_srcs_list, self.n_srcs_prob)[0]  # may allow this random among different rank
        # Calculate the number of batches
        n_batchs = max(np.floor(self.n_frames_batch // (n_srcs + 1)).astype(int), 1)

        # Randomly sample the aspect ratio across the batch
        if isinstance(self.aspect_ratio_range, (float, int)): aspect_ratio = self.aspect_ratio_range
        elif self.fix_random_aspect: aspect_ratio = self.rng.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1])
        else: aspect_ratio = round(random.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1]), 2)

        batch = []
        # Iterate over the number of batches
        for _ in range(n_batchs):
            if self.inside_random:
                index = random.randint(0, len(sampler_iter) - 1)
            else:
                try:
                    index = next(sampler_iter)
                except StopIteration:
                    sampler_iter = iter(self.sampler)
                    index = next(sampler_iter)
            batch.append(dotdict(index=index, n_srcs=n_srcs, aspect_ratio=aspect_ratio))
        yield batch


class IterationBasedBatchSamplerWrapper(BatchSampler):
    """ This is a wrapper for the BatchSampler;

    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler: BatchSampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter
        self.sampler = self.batch_sampler.sampler

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


@DATASAMPLERS.register_module()
class IterationBasedBatchSampler(IterationBasedBatchSamplerWrapper):
    """Backward-compatible alias for IterationBasedBatchSamplerWrapper."""
    pass


def get_inds(dataset: VolumetricVideoDataset,
             frame_sample: List[int] = [0, None, 1],
             view_sample: List[int] = [0, None, 1],
             force_manual_frame_selection: bool = False,
             force_manual_view_selection: bool = False,
             ith_latent: int = 0,):
    inds = torch.arange(0, len(dataset))
    nl = 1 if isinstance(dataset, VolumetricVideoInferenceDataset) else max(1, dataset.n_latents)
    nv = len(dataset) // nl

    # Perform view selection
    view_inds = torch.arange(nv)
    if len(view_sample) != 3 or force_manual_view_selection: view_inds = view_inds[view_sample]  # this is a list of indices
    else: view_inds = view_inds[view_sample[0]:view_sample[1]:view_sample[2]]  # begin, start, end
    if len(view_inds) == 1: view_inds = [view_inds]  # MARK: pytorch indexing bug, when length is 1, will reduce a dim

    # Perform frame selection
    frame_inds = torch.arange(nl)
    if len(frame_sample) != 3 or force_manual_frame_selection: frame_inds = frame_inds[frame_sample]
    else: frame_inds = frame_inds[frame_sample[0]:frame_sample[1]:frame_sample[2]]
    if len(frame_inds) == 1: frame_inds = [frame_inds]  # MARK: pytorch indexing bug, when length is 1, will reduce a dim

    # Actual sampler selection
    inds = inds.reshape(nv, nl)[view_inds][:, frame_inds][:, ith_latent:]
    return inds


# Are there better ways to add existing modules?
@DATASAMPLERS.register_module()
class RandomSampler(RandomSampler):  # cannot use function here, recursion
    """ Training-only sampler, randomly sample from the datasets. """
    def __init__(self,
                 dataset: VolumetricVideoDataset,
                 frame_sample: List[int] = [0, None, 1],
                 view_sample: List[int] = [0, None, 1],
                 ith_latent: int = 0,
                 fix_random: bool = False,
                 seed: Optional[int] = 42,
                 *arg, **kwargs):
        self.dataset = dataset
        self.inds = get_inds(dataset, frame_sample, view_sample, ith_latent, *arg, **kwargs).ravel().numpy().tolist()
        # Create a generator if fix_random is True
        generator = None
        if fix_random:
            generator = torch.Generator()
            if _sampler_log_enabled():
                log(yellow(f"sampler: rank {get_rank()}: fixing RandomSampler random seed to {seed}"))
            generator.manual_seed(seed)
        else:
            if _sampler_log_enabled():
                log(yellow(f"sampler: no seed for RandomSampler"))
        super().__init__(data_source=self.inds, generator=generator)  # strange naming


@DATASAMPLERS.register_module()
class SegmentProbabilitySampler(Sampler[int]):
    """ Training-only sampler, sample from the datasets with probability n_prob_list. """
    def __init__(
        self,
        dataset: VolumetricVideoDataset,
        replacement: bool = True,
        fix_random: bool = False,
        seed: Optional[int] = 42,
        *arg, **kwargs
    ):
        if isinstance(dataset, VolumetricVideoDataset):
            if dataset.split.name not in ["TRAIN"]:
                raise RuntimeError("SegmentProbabilitySampler is only supported in training and validation")

        # Check the `GeneralizableDataset` for more detals
        self.n_lens_list = dataset.n_lens_list
        self.n_prob_list = dataset.n_prob_list
        # Assertions for better debugging
        assert self.n_lens_list.ndim == self.n_prob_list.ndim == 1
        assert len(self.n_lens_list) == len(self.n_prob_list), \
            "n_lens_list must have the same length as n_prob_list"

        # Initialize the sampler
        self._num_samples = self.n_lens_list.sum().item()
        self.cumsum_samples = torch.as_tensor([0] + torch.cumsum(self.n_lens_list, 0).tolist())
        self.replacement = replacement

        # Maybe fix the random seed?
        generator = None
        if fix_random:
            generator = torch.Generator()
            if _sampler_log_enabled():
                log(yellow(f"sampler: rank {get_rank()}: fixing SegmentProbabilitySampler random seed to {seed}"))
            generator.manual_seed(seed)
        else:
            if _sampler_log_enabled():
                log(yellow("sampler: no seed for SegmentProbabilitySampler"))
        self.generator = generator

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        # Generator
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # `multinomial` sampling
        counter = collections.Counter(
            torch.multinomial(
                self.n_prob_list,
                self.num_samples,
                replacement=self.replacement,
                generator=generator,
            ).tolist()
        )

        inds = []
        for idx, count in counter.items():
            offset = int(self.cumsum_samples[idx].item())  # segment start index
            length = int(self.n_lens_list[idx])  # segment length

            # Segment length smaller than count,
            # fallback to sample with replacement
            if count > length:
                inds.extend(
                    (offset + torch.randint(
                        length, (count,), generator=generator
                    )).tolist()
                )
            # Sample without replacement
            else:
                inds.extend(
                    (offset + torch.randperm(
                        length, generator=generator
                    )[:count]).tolist()
                )

        # Global random shuffle
        inds = torch.as_tensor(inds)
        inds = inds[torch.randperm(len(inds), generator=generator)].tolist()
        return iter(inds)

    def __len__(self) -> int:
        return self.num_samples


@DATASAMPLERS.register_module()
class SequentialSampler(SequentialSampler):
    """ Test-only sampler, sample from the datasets sequentially. """
    def __init__(self,
                 dataset: VolumetricVideoDataset,
                 frame_sample: List[int] = [0, None, 1],
                 view_sample: List[int] = [0, None, 1],
                 ith_latent: int = 0,
                 *arg, **kwargs):
        # Check if distributed training is initialized
        if get_distributed() and type(self) is SequentialSampler:
            raise RuntimeError("SequentialSampler is not supported in distributed training")
        if isinstance(dataset, VolumetricVideoDataset):
            if dataset.split.name not in ["VAL", "TEST"]:
                raise RuntimeError("SequentialSampler is only supported in validation and testing")

        self.dataset = dataset
        self.inds = get_inds(dataset, frame_sample, view_sample, ith_latent, *arg, **kwargs).ravel().numpy().tolist()

        # Filter images
        if (self._get_img_filter_file(dataset) is not None) or \
            (hasattr(dataset, 'datasets') and any(
                self._get_img_filter_file(d) is not None
                for d in dataset.datasets
            )):
            self.inds = self._filter_by_img_list(self.inds)
            if not self.inds:
                raise ValueError("target/img list filtering matched no samples; check target_img_list_file paths")

        super().__init__(data_source=self.inds)  # strange naming

    def __iter__(self) -> Iterator[int]:
        yield from [self.inds[i] for i in super().__iter__()]
    
    def _filter_by_img_list(self, inds: List[int]) -> List[int]:
        filter_inds = []
        valid_inds = set(inds)

        # GeneralizableDataset
        if hasattr(self.dataset, 'datasets') and hasattr(self.dataset, 'lengths'):
            lengths = self.dataset.lengths.tolist()
            base_index = 0
            for dataset_ind, subdataset in enumerate(self.dataset.datasets):
                img_filter_file = self._get_img_filter_file(subdataset)
                if img_filter_file is None:
                    base_index += lengths[dataset_ind]
                    continue

                fast_inds = self._fast_filter_subdataset_by_img_list(
                    subdataset=subdataset,
                    img_filter_file=img_filter_file,
                    base_index=base_index,
                    valid_inds=valid_inds,
                )
                if fast_inds is not None:
                    filter_inds.extend(fast_inds)
                    base_index += lengths[dataset_ind]
                    continue

                realpaths = self._load_img_filter_realpaths(img_filter_file)
                for view_ind, frames in enumerate(subdataset.ims):
                    for frame_ind, frame in enumerate(frames):
                        if os.path.realpath(frame) in realpaths:
                            final_id = base_index + view_ind * subdataset.n_latents + frame_ind
                            if final_id in valid_inds:
                                filter_inds.append(final_id)
                base_index += lengths[dataset_ind]

        # MultiviewPointDataset
        else:
            img_filter_file = self._get_img_filter_file(self.dataset)
            if img_filter_file is None:
                return filter_inds
            fast_inds = self._fast_filter_subdataset_by_img_list(
                subdataset=self.dataset,
                img_filter_file=img_filter_file,
                base_index=0,
                valid_inds=valid_inds,
            )
            if fast_inds is not None:
                return fast_inds
            realpaths = self._load_img_filter_realpaths(img_filter_file)
            for view_ind, frames in enumerate(self.dataset.ims):
                for frame_ind, frame in enumerate(frames):
                    if os.path.realpath(frame) in realpaths:
                        final_id = view_ind * self.dataset.n_latents + frame_ind
                        if final_id in valid_inds:
                            filter_inds.append(final_id)

        return filter_inds

    @staticmethod
    def _get_img_filter_file(dataset):
        return getattr(dataset, 'target_img_list_file', None) or getattr(dataset, 'img_list_file', None)

    @staticmethod
    def _load_img_filter_realpaths(img_filter_file: str):
        with open(img_filter_file, 'r') as f:
            return {os.path.realpath(line.strip()) for line in f if line.strip()}

    @staticmethod
    def _load_img_filter_realpaths_ordered(img_filter_file: str):
        with open(img_filter_file, 'r') as f:
            return [os.path.realpath(line.strip()) for line in f if line.strip()]

    @staticmethod
    def _load_img_filter_paths_ordered(img_filter_file: str):
        with open(img_filter_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def _fast_filter_subdataset_by_img_list(subdataset, img_filter_file: str, base_index: int, valid_inds: set):
        try:
            raw_data_root = os.path.abspath(subdataset.data_root)
            data_root = os.path.realpath(raw_data_root)
            images_dir = getattr(subdataset, 'images_dir', 'images')
            n_latents = int(subdataset.n_latents)
            ims = subdataset.ims
        except Exception:
            return None

        filter_inds = []
        raw_data_parent = os.path.dirname(raw_data_root)
        data_parent = os.path.dirname(data_root)
        parsed_any = False
        parse_dirs = [images_dir]
        if images_dir != 'images':
            parse_dirs.append('images')
        camera_names = list(getattr(subdataset, 'camera_names', []))
        for path in SequentialSampler._load_img_filter_paths_ordered(img_filter_file):
            parsed_target = None
            for parse_dir in parse_dirs:
                parsed_target = SequentialSampler._parse_img_path(path, parse_dir)
                if parsed_target is not None:
                    break
            if parsed_target is None:
                continue
            parsed_any = True
            target_parent, camera_name, frame_name = parsed_target
            realpath = os.path.realpath(path)
            if (
                path != raw_data_root
                and not path.startswith(raw_data_root + os.sep)
                and realpath != data_root
                and not realpath.startswith(data_root + os.sep)
                and target_parent not in (raw_data_parent, data_parent)
            ):
                continue
            if camera_name.isdigit():
                view_ind = int(camera_name)
            elif camera_name in camera_names:
                view_ind = camera_names.index(camera_name)
            else:
                continue
            if view_ind < 0 or view_ind >= len(ims):
                continue
            frame_paths = ims[view_ind]
            frame_ind = None
            if frame_name.isdigit():
                candidate_ind = int(frame_name)
                if 0 <= candidate_ind < len(frame_paths):
                    candidate_name = os.path.splitext(os.path.basename(frame_paths[candidate_ind]))[0]
                    if candidate_name == frame_name:
                        frame_ind = candidate_ind
            if frame_ind is None:
                for candidate_ind, candidate_path in enumerate(frame_paths):
                    if os.path.splitext(os.path.basename(candidate_path))[0] == frame_name:
                        frame_ind = candidate_ind
                        break
            if frame_ind is None:
                continue
            final_id = base_index + view_ind * n_latents + frame_ind
            if final_id in valid_inds:
                filter_inds.append(final_id)
        return filter_inds if parsed_any else None

    @staticmethod
    def _parse_img_path(path: str, images_dir: str):
        parts = path.split(os.sep)
        try:
            reverse_index = parts[::-1].index(images_dir)
        except ValueError:
            return None
        images_index = len(parts) - reverse_index - 1
        if images_index + 2 >= len(parts):
            return None
        sequence_root = os.sep.join(parts[:images_index]) or os.sep
        scene_parent = os.path.dirname(sequence_root)
        camera_name = parts[images_index + 1]
        frame_name = os.path.splitext(parts[images_index + 2])[0]
        return scene_parent, camera_name, frame_name


@DATASAMPLERS.register_module()
class DistributedSequentialSampler(SequentialSampler):
    """ Test-only sampler, sample from the datasets sequentially. """
    def __init__(self,
                 dataset: VolumetricVideoDataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 drop_last: bool = False,
                 *arg, **kwargs):
        # Check if distributed training is initialized
        if not get_distributed():
            raise RuntimeError("DDP must be initialized first")
        if isinstance(dataset, VolumetricVideoDataset):
            if dataset.split.name not in ["VAL", "TEST"]:
                raise RuntimeError("DistributedSequentialSampler is only supported in validation and testing")

        if num_replicas is None: num_replicas = get_world_size()
        if rank is None: rank = get_rank()
        if _sampler_log_enabled():
            log(f"rank {rank}: initialize DistributedSequentialSampler with num_replicas={num_replicas}, rank={rank}")

        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        # Inherit from the `EasyVolCap` SequentialSampler
        super().__init__(dataset=dataset, *arg, **kwargs)

        # Maybe drop last
        self.num_samples = len(self.inds)
        if self.drop_last: self.num_samples = self.num_samples // self.num_replicas
        else: self.num_samples = (self.num_samples + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self):
        return self.num_samples

    def get_num_total_valid_samples(self):
        return len(self.inds)

    def __iter__(self):
        # Drop last samples if necessary
        if self.drop_last:
            inds = self.inds[:self.total_size]
        else:
            inds = self.inds + self.inds[:self.total_size - len(self.inds)]
        # Sharding
        inds = inds[self.rank::self.num_replicas]
        return iter(inds)


@DATASAMPLERS.register_module()
class StreamSampler(SequentialSampler):
    # This is indeed the original sequential sampler
    # Since streaming circumstances only need the sample action, but do not care about the sample content
    # The original sequential sampler and random sampler are both wrapped above, with adiitional
    # `frame_sample` and `view_sample` arguments, which we do not need in streaming
    # So, the best way maybe to wrap the original sequential sampler again?
    def __init__(self,
                 dataset: VolumetricVideoDataset,
                 *arg, **kwargs):
        super().__init__(dataset=dataset, *arg, **kwargs)
