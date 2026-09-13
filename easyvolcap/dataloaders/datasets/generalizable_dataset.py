# This dataset creates instances of volumetric_video_dataset or image_based_dataset for all specified scenes
# And samples one of them during training of the generalizable model

import os
import copy
import torch
import random
import pickle
import hashlib
from collections import OrderedDict
from glob import glob
from os.path import join, exists, dirname, relpath
from typing import List, Union

from easyvolcap.engine import DATASETS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import DataSplit
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.dataloaders.datasets.volumetric_video_dataset import VolumetricVideoDataset
from easyvolcap.utils.dist_utils import get_local_rank, get_rank, get_world_size, get_distributed, get_local_size


def _runtime_cache_evict_log_enabled() -> bool:
    value = os.getenv("EVC_RUNTIME_CACHE_EVICT_LOG", "")
    if not value:
        return False
    return value.lower() in ("1", "true", "yes", "y", "on")


@DATASETS.register_module()
class GeneralizableDataset(VolumetricVideoDataset):
    _SCENE_CFG_EXCLUDED_KEYS = {
        "type",
        "dataset_cfgs",
        "metaset_cfgs",
        "data_roots",
        "data_roots_file",
        "meta_roots",
        "meta_file",
        "write_data_roots",
        "runtime_build",
        "delete_after_runtime_build",
        "runtime_dataset_cache_size",
        "dataset_indexing",
        "parallel_loading",
        "dataloading_workers",
        "local_sharding",
        "global_sharding",
        "samples_per_scene",
        "fixed_target_index",
        "scene_pool_size",
        "scene_pool_sorted",
        "scene_pool_contiguous",
    }
    _SCENE_CFG_DEFAULTS = {
        "meta_file": "__meta.pkl",
        "intri_file": "intri.yml",
        "extri_file": "extri.yml",
        "images_dir": "images",
        "cameras_dir": "cameras",
    }

    def __init__(self,
                 # Dataset sharding configurations
                 local_sharding: bool = False,
                 global_sharding: bool = False,

                 # Lazy loading configurations
                 runtime_build: bool = False,  # sometimes the meta datasets are extremely large, and we want to load them in a lazy way
                 delete_after_runtime_build: bool = False, # delete the dataset after runtime build
                 runtime_dataset_cache_size: int = 0,  # 0 means unbounded cache (legacy behavior)
                 dataset_indexing: bool = False,  # whether to sample the index in the dataset pool or view pool

                 # Parallel loading configurations
                 parallel_loading: bool = False,  # this will use the parallel execution to load the datasets
                 dataloading_workers: int = 64,

                 # Consideration, should we expose all configurable entries through a multi-level network like design
                 # Or just leave everything as default?

                 dataset_cfgs: List[dotdict] = [
                    dotdict(type=VolumetricVideoDataset.__name__),  # the last dataset here will be used as default during dataloading
                 ],
                 # The final design:
                 # User could pass in addtional arguments through dataset configs
                 # Each entry in `metaset_cfgs` corresponds to all datasets in a specific meta_root
                 metaset_cfgs: List[dotdict] = [
                     dotdict(type=VolumetricVideoDataset.__name__),  # the last dataset here will be used as default during dataloading
                 ],

                 data_roots=[],  # this will overwrite the dataset configs
                 data_roots_file: str = 'data_roots.txt',  # this will be used to load the data_roots
                 meta_roots=['data/dtu'],  # this will overwrite thet data_roots configs,
                 meta_file: str = None,  # xiaoyang: meta_data = {"intri": ..., "extri": ..., "n_view_total": ..., "n_frames_total": ...}
                 write_data_roots: bool = False,  # avoid writing data_roots.txt under read-only datasets

                 fix_random: bool = False,  # useful when performing deterministic testing with dataset_indexing=True
                 seed: int = 0,  # random seed for reproducibility
                 samples_per_scene: int = 1,  # for dataset_indexing=True: sample K frames per scene deterministically
                 fixed_target_index: int = -1,  # for dataset_indexing=True: force every scene to use a specific target frame index
                 scene_pool_size: int = 0,  # for dataset_indexing=True: pass a fixed pool of indices to downstream dataset (paper protocol)
                 scene_pool_sorted: bool = False,  # for dataset_indexing=True: sort the scene_pool indices (e.g. temporal order)
                 scene_pool_contiguous: bool = False,  # for dataset_indexing=True: sample a contiguous window (video protocol)
                 **kwargs,
                 ):
        # Dataset configs
        self.split = DataSplit[kwargs.get('split', 'TRAIN')]

        default_dataset_cfgs = []
        meta_data = self._load_meta_data(meta_file)

        # Get the default dataset config
        # This will be used to create the dataset if no dataset_cfg is provided
        default_dataset_cfg = self._select_default_dataset_cfg(
            dataset_cfgs=dataset_cfgs,
            metaset_cfgs=metaset_cfgs,
            meta_roots=meta_roots,
        )
        default_dataset_cfg.update(self._scene_dataset_kwargs(kwargs))

        # Prepare for dataset config
        if meta_roots and data_roots:
            log(yellow(f'data_roots entries will be replace by meta_roots: {meta_roots}'))

        if meta_roots:
            # Add more assertions here
            assert len(meta_roots) == len(metaset_cfgs) - 1, \
                f'Length of meta_roots must equals to length of metaset_cfgs - 1'

            data_roots, default_dataset_cfgs = self._expand_meta_roots(
                meta_roots=meta_roots,
                metaset_cfgs=metaset_cfgs,
                data_roots_file=data_roots_file,
                write_data_roots=write_data_roots,
                default_dataset_cfg=default_dataset_cfg,
            )

        # if data_roots and len(default_dataset_cfgs) > 1:
        #     log(yellow(f'dataset_cfgs entries will be replace by data_roots: {data_roots}'))

        if data_roots:
            if not default_dataset_cfgs and not meta_roots and metaset_cfgs and len(metaset_cfgs) > 1:
                # Explicit data_roots overrides may still rely on metaset_cfgs[0] to provide
                # dataset-specific patterns such as DTU/ETH3D mask/depth filename templates.
                scene_dataset_cfg = copy.deepcopy(default_dataset_cfg)
                scene_dataset_cfg.update(metaset_cfgs[0])
                default_dataset_cfgs = [copy.deepcopy(scene_dataset_cfg) for _ in data_roots]
            dataset_cfgs = self._build_dataset_cfgs(
                data_roots=data_roots,
                default_dataset_cfgs=default_dataset_cfgs,
                default_dataset_cfg=default_dataset_cfg,
                meta_data=meta_data,
            )

        # Perform explict dataset sharding here to avoid read conflict
        rank = get_rank() if global_sharding else get_local_rank() if local_sharding else 0
        num_replicas = get_world_size() if global_sharding else get_local_size() if local_sharding else 1
        dataset_cfgs = dataset_cfgs[rank::num_replicas]
        self.dataset_cfgs = dataset_cfgs

        # Warning
        if parallel_loading and not dataset_cfgs[0].get('disk_dataset', False):
            log(yellow(
                f'The nested parallel loading in the GeneralizableDataset will conflict with the disk_dataset=False setting in the VolumetricVideoDataset'
            ))

        # Assertion
        assert not (not dataset_indexing and runtime_build), \
            'Cannot use runtime build mode with dataset_indexing=False, since the dataset index will not be known until the dataset is built'

        # Build the datasets
        if runtime_build:
            # Maybe lazy build
            self.datasets: List[VolumetricVideoDataset] = [None] * len(dataset_cfgs)
        else:
            if parallel_loading:
                # Parallelly build
                self.datasets: List[VolumetricVideoDataset] = parallel_execution(
                    self.dataset_cfgs,
                    action=lambda cfg: DATASETS.build(cfg),
                    num_workers=dataloading_workers,
                    sequential=False,
                    print_progress=True if not get_rank() else False
                )
            else:
                self.datasets: List[VolumetricVideoDataset] = [
                    DATASETS.build(dataset_cfg) for dataset_cfg in self.dataset_cfgs
                ]

        # Compute the lengths and accumulated lengths
        if dataset_indexing:
            self.samples_per_scene = max(1, int(samples_per_scene))
            self.lengths = torch.full((len(self.dataset_cfgs),), self.samples_per_scene, dtype=torch.int64)
        else:
            self.samples_per_scene = 1
            self.lengths = torch.as_tensor([len(d) for d in self.datasets])
        self.accum_lengths = self.lengths.cumsum(dim=-1)
        self.scene_pool_size = max(0, int(scene_pool_size)) if dataset_indexing else 0
        self.scene_pool_sorted = bool(scene_pool_sorted) if dataset_indexing else False
        self.scene_pool_contiguous = bool(scene_pool_contiguous) if dataset_indexing else False
        self.fixed_target_index = int(fixed_target_index) if dataset_indexing else -1

        try:
            total = int(self.lengths.sum().item()) if len(self.lengths) else 0
            log(yellow(
                f"[gds-debug] dataset_indexing={dataset_indexing} scenes={len(self.dataset_cfgs)} total_len={total} "
                f"min_len={int(self.lengths.min().item()) if len(self.lengths) else 0} "
                f"max_len={int(self.lengths.max().item()) if len(self.lengths) else 0}"
            ))
            for i, d in enumerate(self.datasets[:32]):
                root = getattr(d, "data_root", "<unknown>")
                n_views = getattr(d, "n_views", -1)
                n_latents = getattr(d, "n_latents", -1)
                log(yellow(
                    f"[gds-debug] idx={i} len={len(d)} n_views={n_views} n_latents={n_latents} root={root}"
                ))
        except Exception as e:
            log(yellow(f"[gds-debug] failed to dump dataset stats: {e}"))

        # Bookkeepings
        self.local_sharding = local_sharding
        self.global_sharding = global_sharding
        self.runtime_build = runtime_build
        self.delete_after_runtime_build = delete_after_runtime_build
        self.runtime_dataset_cache_size = max(0, int(runtime_dataset_cache_size))
        self.dataset_indexing = dataset_indexing  # whether to sample the index in the dataset pool or view pool
        self.parallel_loading = parallel_loading
        self.dataloading_workers = dataloading_workers
        self.seed = int(seed)
        self._scene_frame_choices = {}  # dataset_index -> List[int]
        self._scene_pool_choices = {}  # dataset_index -> List[int]
        self._runtime_dataset_lru = OrderedDict()  # dataset_index -> None
        self._runtime_dataset_evictions = 0
        if self.runtime_build and not self.delete_after_runtime_build and self.runtime_dataset_cache_size <= 0:
            log(yellow(
                "GeneralizableDataset runtime cache is unbounded. "
                "Set runtime_dataset_cache_size>0 to avoid host-memory growth."
            ))

        # Calculate dataset lengths and sampling probabilities
        self.calculate_dataset_statistics()

        if fix_random:
            log(yellow(f"rank {get_rank()}: fixing GeneralizableDataset random seed to {seed}"))
            self.rng = random.Random(seed)
        else: self.rng = random.Random()

    def _stable_scene_seed(self, dataset_index: int):
        # Use the leaf directory name as a stable scene key (avoid salted python hash).
        try:
            data_root = self.dataset_cfgs[dataset_index].data_root
            parent = os.path.basename(os.path.dirname(data_root))
            leaf = os.path.basename(data_root)
            scene_key = f"{parent}/{leaf}" if parent else leaf
        except Exception:
            scene_key = str(dataset_index)
        payload = f"{self.seed}:{scene_key}".encode("utf-8", errors="ignore")
        digest = hashlib.sha1(payload).digest()[:8]
        return int.from_bytes(digest, byteorder="little", signed=False)

    def _get_scene_frame_choices(self, dataset_index: int, dataset: VolumetricVideoDataset):
        cached = self._scene_frame_choices.get(dataset_index)
        if cached is not None:
            return cached

        n = len(dataset)
        if n <= 0:
            choices = [0]
        else:
            k = min(self.samples_per_scene, n)
            rng = random.Random(self._stable_scene_seed(dataset_index))
            if k < n:
                choices = sorted(rng.sample(range(n), k))
            else:
                choices = list(range(n))

        self._scene_frame_choices[dataset_index] = choices
        return choices

    def _get_scene_pool_choices(self, dataset_index: int, dataset: VolumetricVideoDataset):
        # A fixed pool (e.g. paper protocol) used by downstream datasets to pick src views deterministically.
        if self.scene_pool_size <= 0:
            return None
        cached = self._scene_pool_choices.get(dataset_index)
        if cached is not None:
            return cached

        n = len(dataset)
        if n <= 0:
            choices = [0]
        else:
            k = min(self.scene_pool_size, n)
            rng = random.Random(self._stable_scene_seed(dataset_index))
            if k < n:
                if self.scene_pool_contiguous:
                    # Sample a random starting point and take a contiguous window.
                    # Useful for video benchmarks that use "random clips" instead of random sets.
                    start = rng.randint(0, n - k)
                    choices = list(range(start, start + k))
                else:
                    # Keep the sampled order instead of sorting so the "first view" (canonical frame)
                    # matches the original "random 10 frames" paper protocol more closely.
                    choices = rng.sample(range(n), k)
            else:
                choices = list(range(n))
                rng.shuffle(choices)

        if self.scene_pool_sorted:
            choices = sorted(choices)

        self._scene_pool_choices[dataset_index] = choices
        return choices

    @staticmethod
    def _scene_dataset_kwargs(kwargs):
        return dotdict({
            k: v for k, v in kwargs.items()
            if k not in GeneralizableDataset._SCENE_CFG_EXCLUDED_KEYS
        })

    @staticmethod
    def _load_meta_data(meta_file):
        if meta_file is None:
            return None
        with open(meta_file, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _select_default_dataset_cfg(dataset_cfgs, metaset_cfgs, meta_roots):
        # When explicit data_roots overrides are used, meta_roots may be cleared while
        # metaset_cfgs still carries the actual scene dataset type (e.g. MultiviewPointDataset).
        # In that case, keep using the trailing metaset cfg as the default scene dataset.
        if metaset_cfgs:
            if meta_roots and len(metaset_cfgs) > len(meta_roots):
                return metaset_cfgs[-1]
            if not meta_roots and len(metaset_cfgs) > 1:
                return metaset_cfgs[-1]
        return dataset_cfgs[-1]

    @staticmethod
    def _expand_meta_roots(meta_roots, metaset_cfgs, data_roots_file, write_data_roots, default_dataset_cfg):
        data_roots = []
        default_dataset_cfgs = []

        for i, meta_root in enumerate(meta_roots):
            data_roots_path = join(meta_root, data_roots_file)
            meta_data_roots = None
            if exists(data_roots_path):
                try:
                    with open(data_roots_path, 'r') as f:
                        meta_data_roots = [join(meta_root, line.strip()) for line in f.readlines()]
                except OSError as e:
                    log(yellow(f'failed reading cached {data_roots_file} under {meta_root}: {e}; falling back to directory scan'))

            if meta_data_roots is None:
                meta_data_roots = sorted(glob(join(meta_root, '**', 'images'), recursive=True))
                meta_data_roots = [dirname(data_root) for data_root in meta_data_roots]
                if write_data_roots:
                    try:
                        with open(data_roots_path, 'w') as f:
                            for data_root in meta_data_roots:
                                f.write(f'{relpath(data_root, meta_root)}\n')
                    except OSError as e:
                        log(yellow(f'failed writing cached {data_roots_file} under {meta_root}: {e}; keeping scanned scene list only'))
                else:
                    log(yellow(f'skip writing {data_roots_file} under read-only meta_root: {meta_root}'))
            data_roots.extend(meta_data_roots)

            dataset_cfg = copy.deepcopy(default_dataset_cfg)
            if len(metaset_cfgs) > i:
                dataset_cfg.update(metaset_cfgs[i])
            dataset_cfg.dataset_idx = i
            default_dataset_cfgs.extend([dataset_cfg] * len(meta_data_roots))

        return data_roots, default_dataset_cfgs

    @staticmethod
    def _build_dataset_cfgs(data_roots, default_dataset_cfgs, default_dataset_cfg, meta_data):
        dataset_cfgs = []
        for i, data_root in enumerate(data_roots):
            if len(default_dataset_cfgs):
                dataset_cfg = copy.deepcopy(default_dataset_cfgs[i])
            else:
                dataset_cfg = copy.deepcopy(default_dataset_cfg)
            for k, v in GeneralizableDataset._SCENE_CFG_DEFAULTS.items():
                dataset_cfg.setdefault(k, v)
            dataset_cfg.data_root = data_root
            if meta_data is not None and data_root in meta_data:
                dataset_cfg.meta_data = meta_data[data_root]
            dataset_cfgs.append(dataset_cfg)
        return dataset_cfgs

    @property
    def render_ratio(self):
        return self.datasets[0].render_ratio_shared.item()

    @render_ratio.setter
    def render_ratio(self, value: float):
        for dataset in self.datasets:
            dataset.render_ratio_shared.fill_(value)

    def calculate_dataset_statistics(self):
        """Calculate dataset lengths and sampling probabilities for grouped datasets."""
        # Group datasets by their dataset_idx
        dataset_groups = {}
        for i, cfg in enumerate(self.dataset_cfgs):
            # Explicit data_roots overrides may bypass _expand_meta_roots and thus
            # not attach dataset_idx. Treat such cases as a single metaset.
            dataset_idx = cfg.pop('dataset_idx', 0)
            probability = cfg.pop("prob", 1.0)  # probability, default to 1.0
            # Create a new group if it doesn't exist
            if dataset_idx not in dataset_groups:
                dataset_groups[dataset_idx] = {
                    'configs': [],
                    'indices': [],
                    'prob': probability
                }
            # Append the dataset config and index to the group
            dataset_groups[dataset_idx]['configs'].append(cfg)
            dataset_groups[dataset_idx]['indices'].append(i)

        dataset_lengths = []
        dataset_probs = []
        # Calculate length and probability for each dataset group
        for dataset_idx in sorted(dataset_groups.keys()):
            group = dataset_groups[dataset_idx]
            # Calculate total length for this dataset group
            if self.dataset_indexing:
                group_length = len(group['indices'])  # number of scenes
            else:
                group_length = sum(self.lengths[i] for i in group['indices'])  # lengths of all scenes
            # Append the length and probability to the list
            dataset_lengths.append(group_length)
            dataset_probs.append(group['prob'])

        # Store the results
        self.n_lens_list = torch.as_tensor(dataset_lengths, dtype=torch.int64)
        self.n_prob_list = torch.as_tensor(dataset_probs, dtype=torch.float32)

        # Assertion to ensure the lengths and probabilities match
        assert len(dataset_lengths) == len(dataset_probs), \
            "Dataset lengths and probabilities must match in length"

    def _touch_runtime_dataset_cache(self, dataset_index: int):
        if not self.runtime_build or self.delete_after_runtime_build:
            return
        if self.runtime_dataset_cache_size <= 0:
            return

        # Keep a per-worker LRU list of initialized scene datasets.
        if dataset_index in self._runtime_dataset_lru:
            self._runtime_dataset_lru.move_to_end(dataset_index, last=True)
        else:
            self._runtime_dataset_lru[dataset_index] = None

        while len(self._runtime_dataset_lru) > self.runtime_dataset_cache_size:
            evict_index, _ = self._runtime_dataset_lru.popitem(last=False)
            if evict_index == dataset_index:
                continue
            self.datasets[evict_index] = None
            self._scene_frame_choices.pop(evict_index, None)
            self._scene_pool_choices.pop(evict_index, None)
            self._runtime_dataset_evictions += 1
            if _runtime_cache_evict_log_enabled() and (
                self._runtime_dataset_evictions <= 3 or self._runtime_dataset_evictions % 200 == 0
            ):
                log(yellow(
                    f"GeneralizableDataset runtime cache evict: index={evict_index}, "
                    f"keep={self.runtime_dataset_cache_size}, evictions={self._runtime_dataset_evictions}"
                ))

    def extract_dataset_index(self, index: Union[dotdict, int]):
        # Maybe think of a better way to update input of __getitem__
        # If we've already built all the datasets, we compute the dataset index and sampler index accordingly
        if not self.dataset_indexing:
            if isinstance(index, dotdict): sampler_index, n_srcs = index.index, index.n_srcs
            else: sampler_index = index

            # Dataset index will indicate the sample to use
            dataset_index = torch.searchsorted(self.accum_lengths, sampler_index, right=True)  # 2 will not be inserted as 2
            sampler_index = sampler_index - (self.accum_lengths[dataset_index - 1] if dataset_index > 0 else 0)  # maybe -1
            # MARK: This is nasty, pytorch inconsistency of conversion
            sampler_index = sampler_index.item() if isinstance(sampler_index, torch.Tensor) else sampler_index  # convert to int

        # If we are in runtime build mode, we need to build the dataset first if it is not built yet
        else:
            if isinstance(index, dotdict):
                global_index, n_srcs = index.index, index.n_srcs
            else:
                global_index = index

            global_index = int(global_index)
            if self.samples_per_scene > 1:
                dataset_index = global_index // self.samples_per_scene
                scene_sample_rank = global_index % self.samples_per_scene
            else:
                dataset_index = global_index
                scene_sample_rank = 0

            # Build the dataset if it is not built yet
            if self.datasets[dataset_index] is None:
                last_error = None
                for _ in range(100):  # maximum it will hang 10 minutes
                    try:
                        self.datasets[dataset_index] = DATASETS.build(self.dataset_cfgs[dataset_index])
                        last_error = None
                        break
                    except Exception as e:
                        last_error = e
                        log(red(f"Error building dataset {self.dataset_cfgs[dataset_index].data_root}, hostname {hostname()}, error: {e}, retrying in 1.0s"))
                        time.sleep(5.0)  # lower io frequency

            dataset = self.datasets[dataset_index]
            if dataset is None:
                raise RuntimeError(
                    f"Failed to build dataset after retries: data_root={self.dataset_cfgs[dataset_index].data_root}, "
                    f"dataset_index={dataset_index}, runtime_build={self.runtime_build}, "
                    f"delete_after_runtime_build={self.delete_after_runtime_build}, last_error={last_error}"
                )
            self._touch_runtime_dataset_cache(dataset_index)
            scene_pool = self._get_scene_pool_choices(dataset_index, dataset)
            if isinstance(index, dotdict) and scene_pool is not None:
                index.scene_pool = scene_pool

            if scene_pool is not None:
                # Pick a target index from the pool (paper protocol uses samples_per_scene=1).
                sampler_index = scene_pool[scene_sample_rank % len(scene_pool)]
            elif self.fixed_target_index >= 0:
                sampler_index = min(self.fixed_target_index, max(0, len(dataset) - 1))
            elif self.samples_per_scene > 1 and getattr(self, "seed", 0) is not None:
                choices = self._get_scene_frame_choices(dataset_index, dataset)
                sampler_index = choices[scene_sample_rank % len(choices)]
            else:
                # Backward-compatible: one random frame per scene.
                sampler_index = self.rng.sample(range(len(dataset)), 1)[0]

        if isinstance(index, dotdict): index.index = sampler_index
        else: index = sampler_index

        return dataset_index, index

    @property
    def n_views(self):
        return 1

    @property
    def n_latents(self):
        return sum(self.lengths)  # for samplers

    def __getitem__(self, index: Union[dotdict, int]):
        t0 = time.time()
        dataset_index, index = self.extract_dataset_index(index)
        t1 = time.time()
        dataset = self.datasets[dataset_index]  # get the dataset to sample from
        data = dataset.__getitem__(index)
        if self.runtime_build and self.delete_after_runtime_build:
            self.datasets[dataset_index] = None
            self._runtime_dataset_lru.pop(dataset_index, None)
            self._scene_frame_choices.pop(dataset_index, None)
            self._scene_pool_choices.pop(dataset_index, None)
        t2 = time.time()
        # MARK: temporal code for debugging, will be removed later
        if t2 - t0 > 24:
            log(f"(rank {get_rank()}) __getitem__: {t2 - t0}s, load_datasets: {t1 - t0}s, getitem: {t2 - t1}s, root: {dataset.data_root}")
        return data
