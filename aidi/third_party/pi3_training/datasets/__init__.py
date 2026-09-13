from .base.transforms import *

from utils.misc import get_world_size, get_rank
from torch.utils.data import DataLoader
import hydra
from datasets.base.base_dataset import sample_resolutions, unified_collate_fn
from datasets.base.batched_sampler import DynamicBatchSampler, DynamicDistributedSampler

__HIGH_QUALITY_DATASETS__ = ['BlinkVision', 'Game', 'GameNew', 'DynamicStereo', 'FlyingThings3D', 'GTA-sfm', 'Hypersim', 'MatrixCity', 'MidAir', 'Monkaa', 'PointOdyssey', 'Sintel', 'Spring', 'TarTanAir', 'Unreal4k', 'VirtualKitti', 'Habitat']
__MIDDLE_QUALITY_DATASETS__ = [
    'BlendedMVG', 'BlendedMVS', 'DTU', 'ETH3D', 'ScanNet', 'Scannetpp', 'Taskonomy',
    'Replica', 'ASE', 'ADT', 'MegaDepth', 'WildRGBD', 'MapFree', 'CO3Dv2', 'Mapillary', 'DL3DV',
    'BusinessDriving', 'BusinessParking', 'BusinessParkingMechanical', 'BusinessParkingMechanicalFisheye',
]
__INDOOR_DATASETS__ = ['Hypersim', 'ScanNet', 'Scannetpp', 'Taskonomy', 'ARKitScenes', 'Habitat']

def create_dataloader(cfg, mode):
    data_loader = DataLoader

    # pytorch dataset
    if mode == 'train':
        cfg_dataset = cfg.train_dataset
        cfg_dataloader = cfg.train_dataloader
        batch_size = cfg.train.batch_size
        num_workers = cfg.train.num_workers
    else:
        cfg_dataset = cfg.test_dataset
        cfg_dataloader = cfg.test_dataloader
        batch_size = cfg.test.batch_size
        num_workers = cfg.test.num_workers

    if isinstance(cfg_dataset, str):
        dataset = eval(cfg_dataset) 
    elif 'weights' in cfg_dataset:
        weights = cfg_dataset.weights
        if 'length' in cfg_dataset:
            dataset_length = cfg_dataset.length
            weight_sum = sum([v for k, v in weights.items()])
            new_weights = {}
            for dataset_name, weight in weights.items():
                new_weights[dataset_name] = max(int(weight / weight_sum * dataset_length), 1)
            weights = new_weights
            print(f'New weights for dataset (adjusting to dataset length {dataset_length}): {new_weights}')
        resize_weighted_dataset = mode == 'train' or 'length' in cfg_dataset

        datasets_all = []

        num_resolution = cfg.train.num_resolution if 'num_resolution' in cfg.train and mode == 'train' else 1
        if mode == 'train' and 'random_reslution' in cfg.train and cfg.train.random_reslution:
            seed = 777 + 0
            resolutions = sample_resolutions(aspect_ratio_range=cfg.train.aspect_ratio_range, pixel_count_range=cfg.train.pixel_count_range, patch_size=cfg.train.patch_size, num_resolutions=num_resolution, seed=seed)
            print('Initialized resolution', resolutions)
            num_resolution = len(resolutions)
            for dataset_name, weight in weights.items():
                dataset_i = hydra.utils.instantiate(cfg_dataset[dataset_name], resolution=resolutions)
                dataset_i.convert_attributes()
                datasets_all.append((weight @ dataset_i) if resize_weighted_dataset else dataset_i)
        elif 'resolution' in cfg.train:
            resolutions = cfg.train.resolution
            print('Setting dataset resolution', resolutions)
            for dataset_name, weight in weights.items():
                dataset_i = hydra.utils.instantiate(cfg_dataset[dataset_name], resolution=resolutions)
                dataset_i.convert_attributes()
                datasets_all.append((weight @ dataset_i) if resize_weighted_dataset else dataset_i)
        else:
            for dataset_name, weight in weights.items():
                dataset_i = hydra.utils.instantiate(cfg_dataset[dataset_name])
                dataset_i.convert_attributes()
                datasets_all.append((weight @ dataset_i) if resize_weighted_dataset else dataset_i)
        dataset = datasets_all[0]
        for dataset_ in datasets_all[1:]:
            dataset += dataset_
    else:
        dataset = hydra.utils.instantiate(cfg_dataset)
        dataset.convert_attributes()
    world_size = get_world_size()
    rank = get_rank()

    if mode == 'train':
        image_num_range = cfg.train.image_num_range
    else:
        image_num_range = cfg.test.image_num_range if 'image_num_range' in cfg.test else [8, 8]
    print(f'Sampling frame number range from {image_num_range}')
    # adapte from vggt
    max_img_per_gpu = cfg.train.max_img_per_gpu if 'max_img_per_gpu' in cfg.train else image_num_range[0]
    min_image_num = max(1, int(image_num_range[0]))
    max_img_per_gpu = max(int(max_img_per_gpu), min_image_num)
    print(f'Max frame number per rank {max_img_per_gpu}')
    if mode == 'train' and cfg.train.iters_per_epoch > 0:
        dataset_len_per_rank = len(dataset) // world_size
        # Keep the original strict "<" rule while avoiding brittle hard failure.
        available_samples_per_epoch = max(1, dataset_len_per_rank - 1)
        batch_factor = max(1, max_img_per_gpu // min_image_num)
        needed_samples_per_epoch = batch_factor * cfg.train.iters_per_epoch

        print('Needed batch number per epoch (per rank):', needed_samples_per_epoch)
        print('Dataset length per rank:', dataset_len_per_rank)

        if needed_samples_per_epoch >= dataset_len_per_rank:
            safe_batch_factor = max(1, available_samples_per_epoch // cfg.train.iters_per_epoch)
            safe_max_img_per_gpu = max(min_image_num, safe_batch_factor * min_image_num)
            if safe_max_img_per_gpu < max_img_per_gpu:
                print(
                    f"[WARN] Adjust train.max_img_per_gpu from {max_img_per_gpu} to {safe_max_img_per_gpu} "
                    f"to satisfy sampler constraint ({needed_samples_per_epoch} >= {dataset_len_per_rank})."
                )
                max_img_per_gpu = safe_max_img_per_gpu
                batch_factor = max(1, max_img_per_gpu // min_image_num)
                needed_samples_per_epoch = batch_factor * cfg.train.iters_per_epoch
                print('Adjusted needed batch number per epoch (per rank):', needed_samples_per_epoch)

        if needed_samples_per_epoch >= dataset_len_per_rank:
            max_safe_iters = max(1, available_samples_per_epoch // max(1, max_img_per_gpu // min_image_num))
            raise AssertionError(
                f"Sampler constraint unsatisfied: needed={needed_samples_per_epoch}, "
                f"dataset_len_per_rank={dataset_len_per_rank}. "
                f"Please reduce train.iters_per_epoch <= {max_safe_iters} or increase train_dataset.length."
            )

    sampler = DynamicDistributedSampler(dataset, seed=cfg.train.base_seed, shuffle=cfg_dataloader.shuffle, rank=rank, drop_last=cfg_dataloader.drop_last)
    batch_sampler = DynamicBatchSampler(
        sampler, 
        num_resolution, 
        image_num_range, 
        seed=cfg.train.base_seed,
        max_img_per_gpu=max_img_per_gpu,
        rank=rank
    )

    pin_memory = cfg_dataloader.pin_memory if 'pin_memory' in cfg_dataloader else True
    persistent_workers = cfg_dataloader.persistent_workers if 'persistent_workers' in cfg_dataloader else True
    prefetch_factor = cfg_dataloader.prefetch_factor if 'prefetch_factor' in cfg_dataloader else 2
    loader_kwargs = dict(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        collate_fn=unified_collate_fn,
    )
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return data_loader(**loader_kwargs)
