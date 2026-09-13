from trainers.base_trainer_accelerate import BaseTrainer
from easydict import EasyDict
import torch
from datasets.base.base_dataset import sample_resolutions
import hydra


def _is_pi3_indexer_param_name(name: str) -> bool:
    return ".attn.indexer." in name or ".indexer." in name or name.startswith("indexer.")


def _split_pi3_optimizer_named_parameters(named_params):
    groups = {"encoder": [], "indexer": [], "other": []}
    for name, param in named_params:
        if _is_pi3_indexer_param_name(name):
            groups["indexer"].append((name, param))
        elif name.startswith("encoder.") or ".encoder." in name:
            groups["encoder"].append((name, param))
        else:
            groups["other"].append((name, param))
    return groups


def _sum_losses(losses):
    total = None
    for loss in losses:
        total = loss if total is None else total + loss
    if total is None:
        raise RuntimeError("No indexer loss tensors were produced.")
    return total


class Pi3Trainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_loss = hydra.utils.instantiate(cfg.loss.train_loss)
        self.test_loss = hydra.utils.instantiate(cfg.loss.test_loss)

    def build_optimizer(self, cfg_optimizer, model):
        def param_group_fn(model_):
            grouped_params = _split_pi3_optimizer_named_parameters(model_.named_parameters())
            encoder_params = grouped_params["encoder"]
            indexer_params = grouped_params["indexer"]
            other_params = grouped_params["other"]

            print(f'Number of trainable encoder parameters:', sum(p.numel() for _, p in encoder_params if p.requires_grad))
            print(f'Number of trainable indexer parameters:', sum(p.numel() for _, p in indexer_params if p.requires_grad))
            print(f'Length of trainable others:', sum(p.numel() for _, p in other_params if p.requires_grad))

            def handle_weight_decay(params, weight_decay, lr, group_prefix, is_indexer=False):
                decay = []
                no_decay = []
                for name, param in params:
                    if not param.requires_grad:
                        continue

                    if param.ndim <= 1 or name.endswith(".bias"):
                        no_decay.append(param)
                    else:
                        decay.append(param)

                groups = []
                if no_decay:
                    groups.append({
                        "params": no_decay,
                        "weight_decay": 0.0,
                        "lr": lr,
                        "group_name": f"{group_prefix}_no_decay",
                        "is_indexer": is_indexer,
                    })
                if decay:
                    groups.append({
                        "params": decay,
                        "weight_decay": weight_decay,
                        "lr": lr,
                        "group_name": f"{group_prefix}_decay",
                        "is_indexer": is_indexer,
                    })
                return groups

            indexer_lr = getattr(cfg_optimizer, "indexer_lr", None)
            if indexer_lr is None:
                indexer_cfg = getattr(getattr(self.cfg, "model", None), "indexer_cfg", {})
                if hasattr(indexer_cfg, "get"):
                    indexer_lr = indexer_cfg.get("sparse_lr", None)
            if indexer_lr is None:
                indexer_lr = cfg_optimizer.lr

            res = []
            res.extend(handle_weight_decay(encoder_params, cfg_optimizer.weight_decay, cfg_optimizer.encoder_lr, "encoder"))
            res.extend(handle_weight_decay(indexer_params, cfg_optimizer.weight_decay, indexer_lr, "indexer", is_indexer=True))
            res.extend(handle_weight_decay(other_params, cfg_optimizer.weight_decay, cfg_optimizer.lr, "other"))

            return res
        
        return super().build_optimizer(cfg_optimizer, model, param_group_fn=param_group_fn)

    def before_epoch(self, epoch):
        if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, 'set_epoch'):
            self.train_loader.dataset.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.train_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'batch_sampler') and hasattr(self.train_loader.batch_sampler.batch_sampler, 'sampler') and hasattr(self.train_loader.batch_sampler.batch_sampler.sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.train_loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.train_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.train_loader.batch_sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        

        if hasattr(self.test_loader, 'dataset') and hasattr(self.test_loader.dataset, 'set_epoch'):
            self.test_loader.dataset.set_epoch(0, base_seed=self.cfg.train.base_seed)
        if hasattr(self.test_loader, 'batch_sampler') and hasattr(self.test_loader.batch_sampler, 'batch_sampler') and hasattr(self.test_loader.batch_sampler.batch_sampler, 'sampler') and hasattr(self.test_loader.batch_sampler.batch_sampler.sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.test_loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.test_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.test_loader.batch_sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)

        if 'random_reslution' in self.cfg.train and self.cfg.train.random_reslution and self.cfg.train.num_resolution > 0:
            seed = epoch + self.cfg.train.base_seed
            resolutions = sample_resolutions(aspect_ratio_range=self.cfg.train.aspect_ratio_range, pixel_count_range=self.cfg.train.pixel_count_range, patch_size=self.cfg.train.patch_size, num_resolutions=self.cfg.train.num_resolution, seed=seed)
            print('[Pi3 Trainer] Sampled new resolutions:', resolutions)
            datasets = []
            recursive_get_dataset(self.train_loader.dataset, datasets)
            for dataset in datasets:
                dataset._set_resolutions(resolutions)

    def _unwrap_model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self.model)
        return self.model

    def _apply_indexer_stage_lr(self, state):
        if state is None or not hasattr(self, "optimizer"):
            return
        indexer_cfg = getattr(self.cfg.model, "indexer_cfg", {})
        indexer_groups = [
            group for group in self.optimizer.param_groups
            if group.get("is_indexer", False)
        ]
        if state.get("warmup", False):
            target_lr = indexer_cfg.get("warmup_lr", None)
            if target_lr is None:
                return
            target_lr = float(target_lr)
            for group in indexer_groups:
                group["lr"] = target_lr
            return

        if state.get("sparse", False):
            sparse_lr = indexer_cfg.get("sparse_lr", None)
            if sparse_lr is None or not indexer_groups:
                return

            decay_mode = str(indexer_cfg.get("sparse_lr_decay", "scheduler")).lower()
            sparse_lr = float(sparse_lr)
            sparse_min_lr = indexer_cfg.get("sparse_min_lr", None)
            sparse_min_lr = None if sparse_min_lr is None else float(sparse_min_lr)
            for group in indexer_groups:
                fallback_lr = float(group.get("lr", sparse_lr))
                if decay_mode in ("scheduler", "sched", "follow_scheduler", "decay"):
                    target_lr = fallback_lr
                elif decay_mode in ("constant", "none"):
                    target_lr = sparse_lr
                elif decay_mode in ("cap", "clamp"):
                    target_lr = min(sparse_lr, fallback_lr)
                else:
                    raise ValueError(
                        f"Unsupported sparse_lr_decay={decay_mode}, "
                        "choose from: scheduler, constant, cap"
                    )
                if sparse_min_lr is not None:
                    target_lr = max(sparse_min_lr, target_lr)
                group["lr"] = target_lr
            return

    def _sync_indexer_state(self, mode='train'):
        model = self._unwrap_model()
        if not hasattr(model, "set_indexer_state_by_step"):
            return None
        step = int(getattr(self, "global_step", 0))
        state = model.set_indexer_state_by_step(step, training=(mode == 'train'))
        self._apply_indexer_stage_lr(state)
        return state

    def _use_warmup_only_indexer_loss(self, state, mode='train'):
        if mode != 'train' or not state:
            return False
        model = self._unwrap_model()
        indexer_cfg = getattr(model, "indexer_cfg", {})
        return bool(state.get("warmup", False) and indexer_cfg.get("warmup_only_indexer_loss", True))

    @staticmethod
    def _stack_optional_view_tensor(batch, key):
        if not batch or key not in batch[0] or batch[0][key] is None:
            return None
        values = [view.get(key, None) for view in batch]
        if any(value is None for value in values):
            return None
        return torch.stack(values, dim=1)
            
    def forward_batch(self, batch, mode='train'):
        state = self._sync_indexer_state(mode=mode)
        warmup_only_indexer_loss = self._use_warmup_only_indexer_loss(state, mode=mode)
        imgs = torch.stack([view['img'] for view in batch], dim=1)
        pose_prior = self._stack_optional_view_tensor(batch, 'pose_prior')
        pose_prior_intrinsics = self._stack_optional_view_tensor(batch, 'camera_intrinsics') if pose_prior is not None else None
        pred = self.model(
            imgs,
            warmup_only_indexer_loss=warmup_only_indexer_loss,
            pose_prior=pose_prior,
            pose_prior_intrinsics=pose_prior_intrinsics,
        )

        return [pred, batch, warmup_only_indexer_loss]
    
    def calculate_loss(self, output, batch, mode='train'):
        output, batch, warmup_only_indexer_loss = output

        if warmup_only_indexer_loss:
            loss = None
            details = {}
        elif mode == 'train':
            loss, details = self.train_loss(output, batch)
        else:
            loss, details = self.test_loss(output, batch)

        model = self._unwrap_model()
        indexer_loss = getattr(model, "indexer_loss", None)
        if indexer_loss is not None:
            state = model.get_indexer_state() if hasattr(model, "get_indexer_state") else {}
            if isinstance(indexer_loss, (list, tuple)):
                indexer_loss_total = _sum_losses(indexer_loss)
                details["indexer_loss"] = indexer_loss_total.detach()
            else:
                indexer_loss_total = indexer_loss
                details["indexer_loss"] = indexer_loss.detach()
            details["indexer_sparse"] = float(state.get("sparse", False))
            details["indexer_warmup"] = float(state.get("warmup", False))
            if mode == 'train':
                if state.get("warmup", False) and model.indexer_cfg.get("warmup_only_indexer_loss", True):
                    loss = list(indexer_loss) if isinstance(indexer_loss, (list, tuple)) else indexer_loss
                else:
                    loss = loss + indexer_loss_total
        elif warmup_only_indexer_loss:
            raise RuntimeError("warmup_only_indexer_loss=True requires model.indexer_loss to be populated")

        return EasyDict(
            loss=loss,
            **details
        )


def recursive_get_dataset(dataset, res=[]):
    if hasattr(dataset, 'datasets'):
        for ds in dataset.datasets:
            recursive_get_dataset(ds, res)
    else:
        if hasattr(dataset, 'dataset'):
            res.append(dataset.dataset)
    return res
