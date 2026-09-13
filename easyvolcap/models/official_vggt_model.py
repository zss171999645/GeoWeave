import math
import os
import torch
from torch import nn

from easyvolcap.engine import MODELS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.official_vggt.models.vggt import VGGT
from easyvolcap.official_vggt.training.loss import MultitaskLoss
from easyvolcap.official_vggt.train_utils.freeze import freeze_modules
from easyvolcap.official_vggt.layers import attention as vggt_attention
from easyvolcap.official_vggt.utils.epipolar_selector import augment_indexer_state_with_camera
from easyvolcap.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
from easyvolcap.utils.console_utils import blue, log, yellow
from easyvolcap.utils.ray_utils import get_rays


@MODELS.register_module()
class OfficialVGGTModel(nn.Module):
    def __init__(
        self,
        vggt_cfg: dotdict = dotdict(),
        loss_cfg: dotdict = dotdict(),
        frozen_module_names: list = None,
        pretrained_path: str = "",
        agg_ckpt: str = "",
        cam_ckpt: str = "",
        xyz_ckpt: str = "",
        dpt_ckpt: str = "",
        tra_ckpt: str = "",
        pretrained_strict: bool = True,
        dtype: str = "float",
        depth_xyz_use_gt_camera: bool = False,
    ):
        super().__init__()
        memory_cfg = dotdict(vggt_cfg.get("memory_cfg", {}))
        vggt_attention.set_xformers_enabled(memory_cfg.get("use_xformers", False))
        self.vggt = VGGT(**vggt_cfg)
        self.indexer_cfg = dotdict(vggt_cfg.get("indexer_cfg", {}))
        if not self.indexer_cfg.get("enabled", False) and hasattr(self.vggt, "aggregator"):
            agg_cfg = getattr(self.vggt.aggregator, "indexer_cfg", None)
            if isinstance(agg_cfg, (dict, dotdict)) and agg_cfg.get("enabled", False):
                self.indexer_cfg = dotdict(agg_cfg)
        self.loss_cfg = dotdict(loss_cfg or {})
        self.loss_fn = MultitaskLoss(**self.loss_cfg) if self.loss_cfg else None
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.depth_xyz_use_gt_camera = bool(depth_xyz_use_gt_camera)

        if frozen_module_names:
            lock_train = not self.indexer_cfg.get("enabled", False)
            freeze_modules(self.vggt, frozen_module_names, lock_train=lock_train)
        self._base_requires_grad = {name: param.requires_grad for name, param in self.vggt.named_parameters()}
        self._indexer_stage = "none"
        self._warmup_saved = False
        self._pending_warmup_ckpt_step = None
        self._initial_stage_lr_bootstrapped = False
        self._indexer_init_loaded = False

        if self.loss_fn is not None:
            if hasattr(self.vggt, "depth_head") and hasattr(self.vggt.depth_head, "set_loss_cfg"):
                self.vggt.depth_head.set_loss_cfg(self.loss_cfg.get("depth", {}))
            if hasattr(self.vggt, "point_head") and hasattr(self.vggt.point_head, "set_loss_cfg"):
                self.vggt.point_head.set_loss_cfg(self.loss_cfg.get("point", {}))

        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model", checkpoint)
            state_dict, _ = self._remap_special_token_keys(state_dict)
            missing, unexpected = self.vggt.load_state_dict(state_dict, strict=False)
            if pretrained_strict:
                missing, unexpected = self._filter_optional_state_dict_diffs(missing, unexpected)
                if missing or unexpected:
                    raise RuntimeError(
                        f"Error(s) in loading state_dict for vggt: missing={missing}, unexpected={unexpected}"
                    )
        else:
            self._load_module_ckpt(self.vggt.aggregator, agg_ckpt, pretrained_strict, "aggregator")
            self._load_module_ckpt(self.vggt.camera_head, cam_ckpt, pretrained_strict, "camera")
            self._load_module_ckpt(self.vggt.point_head, xyz_ckpt, pretrained_strict, "point")
            self._load_module_ckpt(self.vggt.depth_head, dpt_ckpt, pretrained_strict, "depth")
            self._load_module_ckpt(self.vggt.track_head, tra_ckpt, pretrained_strict, "track")

    @staticmethod
    def _remap_special_token_keys(state_dict: dict) -> tuple[dict, bool]:
        if state_dict is None:
            return state_dict, False
        remapped = False
        state_dict = dict(state_dict)

        def _remap_suffix(old_suffix: str, new_suffix: str) -> None:
            nonlocal remapped
            for key in list(state_dict.keys()):
                if not key.endswith(old_suffix):
                    continue
                if key.endswith(new_suffix):
                    continue
                new_key = key[:-len(old_suffix)] + new_suffix
                if new_key in state_dict:
                    continue
                state_dict[new_key] = state_dict.pop(key)
                remapped = True

        _remap_suffix("camera_token", "special_tokens.camera_token")
        _remap_suffix("register_token", "special_tokens.register_token")
        _remap_suffix("patch_embed.cls_token", "patch_embed.special_tokens.cls_token")
        _remap_suffix("patch_embed.pos_embed", "patch_embed.special_tokens.pos_embed")
        _remap_suffix("patch_embed.register_tokens", "patch_embed.special_tokens.register_tokens")
        _remap_suffix("patch_embed.mask_token", "patch_embed.special_tokens.mask_token")
        return state_dict, remapped

    @staticmethod
    def _filter_optional_state_dict_diffs(missing: list[str], unexpected: list[str]) -> tuple[list[str], list[str]]:
        optional_tokens = ("indexer", "soft_view_bias")
        missing = [key for key in missing if not any(token in key for token in optional_tokens)]
        unexpected = [key for key in unexpected if not any(token in key for token in optional_tokens)]
        return missing, unexpected

    @staticmethod
    def _load_module_ckpt(module: nn.Module, ckpt_path: str, strict: bool, name: str) -> None:
        if module is None or not ckpt_path:
            return
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = state.get("model", state)
        if name == "aggregator":
            state_dict, _ = OfficialVGGTModel._remap_special_token_keys(state_dict)
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        if strict:
            missing, unexpected = OfficialVGGTModel._filter_optional_state_dict_diffs(missing, unexpected)
            if missing or unexpected:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {name}: missing={missing}, unexpected={unexpected}"
                )

    @staticmethod
    def _normalize_ckpt_key(key: str) -> str:
        while key.startswith("module."):
            key = key[len("module."):]
        return key

    @staticmethod
    def _parse_key_tokens(tokens) -> tuple[str, ...]:
        if tokens is None:
            return ("indexer",)
        if isinstance(tokens, str):
            tokens = tokens.replace(";", ",").split(",")
        parsed = tuple(str(token).strip() for token in tokens if str(token).strip())
        return parsed or ("indexer",)

    def _load_indexer_init_ckpt_once(self) -> None:
        if self._indexer_init_loaded:
            return
        ckpt_path = (
            self.indexer_cfg.get("init_ckpt", "")
            or self.indexer_cfg.get("init_from", "")
            or self.indexer_cfg.get("init_pretrained", "")
        )
        if not ckpt_path:
            return

        key_tokens = self._parse_key_tokens(self.indexer_cfg.get("init_key_tokens", ("indexer",)))
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = state.get("model", state)
        state_dict, _ = self._remap_special_token_keys(state_dict)

        current = self.state_dict()
        filtered = {}
        shape_mismatches = []
        for raw_key, value in state_dict.items():
            key = self._normalize_ckpt_key(raw_key)
            candidates = [key]
            if not key.startswith("vggt."):
                candidates.append(f"vggt.{key}")
            for candidate in candidates:
                if candidate not in current:
                    continue
                if not any(token in candidate for token in key_tokens):
                    continue
                if hasattr(value, "shape") and tuple(value.shape) != tuple(current[candidate].shape):
                    shape_mismatches.append(
                        f"{candidate}: ckpt={tuple(value.shape)} model={tuple(current[candidate].shape)}"
                    )
                    break
                filtered[candidate] = value
                break

        min_keys = int(self.indexer_cfg.get("init_min_keys", 1))
        if len(filtered) < min_keys:
            raise RuntimeError(
                f"Indexer init checkpoint {ckpt_path} matched {len(filtered)} keys, "
                f"below init_min_keys={min_keys}; key_tokens={key_tokens}"
            )

        self.load_state_dict(filtered, strict=False)
        self._indexer_init_loaded = True

        if shape_mismatches:
            preview = "; ".join(shape_mismatches[:5])
            log(yellow(f"Skipped {len(shape_mismatches)} indexer init keys with shape mismatch: {preview}"))
        log(
            f"Loaded {blue(len(filtered))} indexer init keys from {blue(ckpt_path)} "
            f"with key_tokens={key_tokens}"
        )

    def after_pretrained_model_loaded(self, runner=None) -> None:
        self._load_indexer_init_ckpt_once()

    def _get_hw(self, batch: dotdict) -> tuple:
        if hasattr(batch, "meta") and hasattr(batch.meta, "H") and hasattr(batch.meta, "W"):
            h = int(batch.meta.H.max().item())
            w = int(batch.meta.W.max().item())
            return h, w
        raise ValueError("Missing meta.H/meta.W for VGGT batch reshape")

    def _build_official_batch(self, batch: dotdict, images: torch.Tensor) -> dict:
        B, S = images.shape[:2]
        H, W = images.shape[-2:]

        if hasattr(batch, "msk"):
            point_masks = batch.msk.reshape(B, S, H, W)
        else:
            point_masks = torch.ones((B, S, H, W), device=images.device, dtype=images.dtype)

        point_masks = point_masks > 0

        if hasattr(batch, "dpt"):
            depths = batch.dpt.reshape(B, S, H, W)
        else:
            depths = torch.zeros((B, S, H, W), device=images.device, dtype=images.dtype)

        if hasattr(batch, "xyz"):
            world_points = batch.xyz.reshape(B, S, H, W, 3)
        else:
            world_points = torch.zeros((B, S, H, W, 3), device=images.device, dtype=images.dtype)

        if hasattr(batch, "w2cs"):
            extrinsics = batch.w2cs[..., :3, :4]
        elif hasattr(batch, "R") and hasattr(batch, "T"):
            extrinsics = torch.cat([batch.R, batch.T], dim=-1)
        else:
            raise ValueError("Missing w2cs or (R, T) for VGGT batch")

        if hasattr(batch, "ixts"):
            intrinsics = batch.ixts
        elif hasattr(batch, "K"):
            intrinsics = batch.K
        else:
            raise ValueError("Missing ixts or K for VGGT batch")

        return {
            "images": images,
            "point_masks": point_masks,
            "depths": depths,
            "world_points": world_points,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
        }

    @staticmethod
    def _get_global_step(batch: dotdict) -> int:
        if hasattr(batch, "meta") and hasattr(batch.meta, "iter"):
            try:
                return int(batch.meta.iter.item())
            except Exception:
                return int(batch.meta.iter)
        return 0

    def _set_indexer_trainable(self, only_indexer: bool) -> None:
        for name, param in self.vggt.named_parameters():
            if "indexer" in name:
                param.requires_grad = True
                continue
            param.requires_grad = False if only_indexer else self._base_requires_grad.get(name, True)

    def _indexer_state_from_step(self, step: int) -> dotdict:
        if not self.indexer_cfg.get("enabled", False):
            return dotdict(enabled=False)

        enable_sparse = self.indexer_cfg.get("enable_sparse", True)
        warmup_steps = int(self.indexer_cfg.get("warmup_steps", 0))
        sparse_start = int(self.indexer_cfg.get("sparse_start_step", warmup_steps))
        warmup = step < warmup_steps
        sparse = enable_sparse and step >= sparse_start

        if not self.training and self.indexer_cfg.get("inference_sparse", False):
            sparse = True

        loss_weight = self.indexer_cfg.get("loss_weight", 1.0)
        if warmup:
            loss_weight = self.indexer_cfg.get("warmup_loss_weight", loss_weight)
        elif sparse:
            loss_weight = self.indexer_cfg.get("sparse_loss_weight", loss_weight)

        compute_indexer_loss = self.indexer_cfg.get("compute_loss", True)
        if isinstance(compute_indexer_loss, str):
            compute_indexer_loss = compute_indexer_loss.strip().lower() not in ("0", "false", "no", "off")
        else:
            compute_indexer_loss = bool(compute_indexer_loss)

        return dotdict(
            enabled=True,
            warmup=warmup,
            sparse=sparse,
            compute_loss=self.training and compute_indexer_loss,
            detach_input=self.indexer_cfg.get("detach_input", True),
            score_dtype=self.indexer_cfg.get("score_dtype", None),
            sparse_use_mask=self.indexer_cfg.get("sparse_use_mask", False),
            topk=self.indexer_cfg.get("topk", 256),
            loss_weight=loss_weight,
            eps=self.indexer_cfg.get("eps", 1e-6),
            warmup_indexer_loss_mode=self.indexer_cfg.get("warmup_indexer_loss_mode", "kl"),
            warmup_topk_coverage_k=self.indexer_cfg.get(
                "warmup_topk_coverage_k",
                self.indexer_cfg.get("topk", 256),
            ),
            warmup_topk_coverage_chunk_size=self.indexer_cfg.get("warmup_topk_coverage_chunk_size", 128),
            warmup_topk_coverage_query_chunk_size=self.indexer_cfg.get("warmup_topk_coverage_query_chunk_size", 256),
            warmup_topk_coverage_query_sample_size=self.indexer_cfg.get(
                "warmup_topk_coverage_query_sample_size",
                0,
            ),
        )

    def _maybe_save_warmup_checkpoint(self, runner, step: int) -> None:
        if self._warmup_saved:
            return
        if not self.indexer_cfg.get("save_warmup_ckpt", True):
            return
        if runner is None or not hasattr(runner, "save_warmup_checkpoint"):
            return
        tag = self.indexer_cfg.get("warmup_ckpt_tag", "warmup_end")
        runner.save_warmup_checkpoint(step, tag=tag)
        self._warmup_saved = True

    def prepare_params(self, runner, batch: dotdict):
        if not self.indexer_cfg.get("enabled", False):
            return

        step = self._get_global_step(batch)
        state = self._indexer_state_from_step(step)
        if hasattr(self.vggt, "aggregator"):
            self.vggt.aggregator.set_indexer_state(state)

        stage = "warmup" if state.get("warmup", False) else "sparse" if state.get("sparse", False) else "dense"
        if stage != self._indexer_stage:
            if self._indexer_stage == "warmup" and stage != "warmup":
                # Defer checkpoint saving to post-optimizer hook so the last warmup update is included.
                self._pending_warmup_ckpt_step = step
            self._set_indexer_trainable(stage == "warmup")
            self._indexer_stage = stage
        self._bootstrap_initial_stage_lr(runner, step, state)

    def decorate_params(self, runner, batch: dotdict):
        if not self.indexer_cfg.get("enabled", False):
            return

        step = self._get_global_step(batch)
        state = self._indexer_state_from_step(step)
        warmup_lr = self.indexer_cfg.get("warmup_lr", None)
        sparse_lr = self.indexer_cfg.get("sparse_lr", None)
        if state.get("warmup", False) and warmup_lr is not None and runner is not None:
            for group in runner.optimizer.param_groups:
                scheduler_lr = float(group.get("lr", warmup_lr))
                group["lr"] = self._compute_warmup_lr(step, scheduler_lr)
        elif state.get("sparse", False) and runner is not None:
            for group in runner.optimizer.param_groups:
                scheduler_lr = float(group.get("lr", sparse_lr if sparse_lr is not None else 0.0))
                group["lr"] = self._compute_sparse_lr(scheduler_lr)
        else:
            pass

        self._maybe_save_warmup_checkpoint_after_step(runner, step, state)

    def _bootstrap_initial_stage_lr(self, runner, step: int, state: dotdict) -> None:
        if self._initial_stage_lr_bootstrapped:
            return
        if runner is None or not hasattr(runner, "optimizer") or runner.optimizer is None:
            return
        if step != 0:
            self._initial_stage_lr_bootstrapped = True
            return

        warmup_lr = self.indexer_cfg.get("warmup_lr", None)
        sparse_lr = self.indexer_cfg.get("sparse_lr", None)
        if state.get("warmup", False) and warmup_lr is not None:
            for group in runner.optimizer.param_groups:
                scheduler_lr = float(group.get("lr", warmup_lr))
                group["lr"] = self._compute_warmup_lr(step - 1, scheduler_lr)
        elif state.get("sparse", False):
            for group in runner.optimizer.param_groups:
                scheduler_lr = float(group.get("lr", sparse_lr if sparse_lr is not None else 0.0))
                group["lr"] = self._compute_sparse_lr(scheduler_lr)

        self._initial_stage_lr_bootstrapped = True

    def _compute_warmup_lr(self, step: int, fallback_lr: float) -> float:
        warmup_start_lr = float(self.indexer_cfg.get("warmup_lr"))
        warmup_steps = max(int(self.indexer_cfg.get("warmup_steps", 0)), 1)
        next_step = min(max(step + 1, 0), warmup_steps)
        progress = float(next_step) / float(warmup_steps)

        warmup_end_lr = self.indexer_cfg.get("warmup_end_lr", None)
        if warmup_end_lr is None:
            sparse_lr = self.indexer_cfg.get("sparse_lr", None)
            warmup_end_lr = fallback_lr if sparse_lr is None else sparse_lr
        warmup_end_lr = float(warmup_end_lr)

        decay_mode = str(self.indexer_cfg.get("warmup_lr_decay", "cosine")).lower()
        if decay_mode in ("constant", "none"):
            lr = warmup_start_lr
        elif decay_mode in ("linear", "lin"):
            lr = warmup_start_lr + (warmup_end_lr - warmup_start_lr) * progress
        elif decay_mode in ("exponential", "exp"):
            if warmup_start_lr > 0.0 and warmup_end_lr > 0.0:
                lr = warmup_start_lr * ((warmup_end_lr / warmup_start_lr) ** progress)
            else:
                lr = warmup_start_lr + (warmup_end_lr - warmup_start_lr) * progress
        elif decay_mode in ("cosine", "cos"):
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = warmup_end_lr + (warmup_start_lr - warmup_end_lr) * cosine
        else:
            raise ValueError(
                f"Unsupported warmup_lr_decay={decay_mode}, "
                "choose from: cosine, linear, exponential, constant"
            )

        warmup_min_lr = self.indexer_cfg.get("warmup_min_lr", None)
        if warmup_min_lr is not None:
            lr = max(float(warmup_min_lr), lr)
        warmup_max_lr = self.indexer_cfg.get("warmup_max_lr", None)
        if warmup_max_lr is not None:
            lr = min(float(warmup_max_lr), lr)
        return float(lr)

    def _compute_sparse_lr(self, fallback_lr: float) -> float:
        sparse_lr = self.indexer_cfg.get("sparse_lr", None)
        if sparse_lr is None:
            lr = float(fallback_lr)
        else:
            decay_mode = str(self.indexer_cfg.get("sparse_lr_decay", "scheduler")).lower()
            if decay_mode in ("scheduler", "sched", "follow_scheduler", "decay"):
                lr = float(fallback_lr)
            elif decay_mode in ("constant", "none"):
                lr = float(sparse_lr)
            elif decay_mode in ("cap", "clamp"):
                lr = min(float(sparse_lr), float(fallback_lr))
            else:
                raise ValueError(
                    f"Unsupported sparse_lr_decay={decay_mode}, "
                    "choose from: scheduler, constant, cap"
                )

        sparse_min_lr = self.indexer_cfg.get("sparse_min_lr", None)
        if sparse_min_lr is not None:
            lr = max(float(sparse_min_lr), lr)
        sparse_max_lr = self.indexer_cfg.get("sparse_max_lr", None)
        if sparse_max_lr is not None:
            lr = min(float(sparse_max_lr), lr)
        return float(lr)

    def _maybe_save_warmup_checkpoint_after_step(self, runner, step: int, state: dotdict) -> None:
        if self._warmup_saved:
            return
        if runner is None:
            return

        warmup_steps = int(self.indexer_cfg.get("warmup_steps", 0))
        if warmup_steps <= 0:
            return

        is_last_warmup_step = bool(state.get("warmup", False)) and (step + 1 >= warmup_steps)
        if is_last_warmup_step:
            self._maybe_save_warmup_checkpoint(runner, step + 1)
            self._pending_warmup_ckpt_step = None
            return

        if self._pending_warmup_ckpt_step is not None and step >= self._pending_warmup_ckpt_step:
            self._maybe_save_warmup_checkpoint(runner, step)
            self._pending_warmup_ckpt_step = None

    @staticmethod
    def _scale_module_grads(module: nn.Module, scale: float) -> None:
        if module is None:
            return
        for param in module.parameters():
            if param.grad is not None:
                param.grad.mul_(scale)

    def decorate_grads(self, runner, batch: dotdict):
        if not self.indexer_cfg.get("enabled", False):
            return

        step = self._get_global_step(batch)
        state = self._indexer_state_from_step(step)
        if state.get("warmup", False):
            return

        head_lr_scale = self.indexer_cfg.get("head_lr_scale", None)
        head_lr = self.indexer_cfg.get("head_lr", None)
        if head_lr is not None and runner is not None and runner.optimizer is not None:
            base_lr = runner.optimizer.param_groups[0].get("lr", None)
            if base_lr:
                head_lr_scale = float(head_lr) / float(base_lr)

        if head_lr_scale is None:
            return

        scale = float(head_lr_scale)
        if scale >= 1.0:
            return

        self._scale_module_grads(self.vggt.camera_head, scale)
        self._scale_module_grads(self.vggt.depth_head, scale)
        self._scale_module_grads(self.vggt.point_head, scale)
        self._scale_module_grads(self.vggt.track_head, scale)

    def forward(self, batch: dotdict, compute_loss: bool = False):
        if hasattr(batch, "rgb"):
            raw = batch.rgb
        elif hasattr(batch, "images"):
            raw = batch.images
        else:
            raise ValueError("Batch must contain rgb or images for VGGT training")

        if raw.dim() == 5 and raw.shape[2] == 3:
            images = raw
        elif raw.dim() == 4 and raw.shape[-1] == 3:
            H, W = self._get_hw(batch)
            B, S = raw.shape[:2]
            images = raw.reshape(B, S, H, W, 3).permute(0, 1, 4, 2, 3).contiguous()
        else:
            raise ValueError(f"Unexpected image tensor shape: {tuple(raw.shape)}")

        images = images.to(dtype=self.dtype)
        step = self._get_global_step(batch)
        official_batch = None
        if self.training or compute_loss:
            official_batch = self._build_official_batch(batch, images)

        if self.indexer_cfg.get("enabled", False) and hasattr(self.vggt, "aggregator"):
            state = self._indexer_state_from_step(step)
            state.loss_scaler = float(getattr(batch, "loss_scaler", 1.0))
            state.no_sync_context = getattr(batch, "no_sync_context", None)
            state = augment_indexer_state_with_camera(
                state,
                official_batch,
                image_height=int(images.shape[-2]),
                image_width=int(images.shape[-1]),
            )
            self.vggt.aggregator.set_indexer_state(state)

        predictions = self.vggt(images, batch=batch, official_batch=official_batch)
        output = dotdict(predictions)

        if "pose_enc_list" in predictions:
            output.cam_maps = predictions["pose_enc_list"]
            output.cam_map = predictions.get("pose_enc", predictions["pose_enc_list"][-1])

        if "depth" in predictions:
            depth = predictions["depth"]
            if depth.dim() == 4:
                depth = depth.unsqueeze(-1)
            output.dpt_map = depth.reshape(depth.shape[0], depth.shape[1], -1, depth.shape[-1])
            depth_conf = predictions.get("depth_conf", None)
            if depth_conf is not None:
                if depth_conf.dim() == 4:
                    depth_conf = depth_conf.unsqueeze(-1)
                output.dpt_cnf = depth_conf.reshape(depth_conf.shape[0], depth_conf.shape[1], -1, 1)

        if "world_points" in predictions:
            world_points = predictions["world_points"]
            output.xyz_map = world_points.reshape(world_points.shape[0], world_points.shape[1], -1, 3)
            world_points_conf = predictions.get("world_points_conf", None)
            if world_points_conf is not None:
                if world_points_conf.dim() == 4:
                    world_points_conf = world_points_conf.unsqueeze(-1)
                output.xyz_cnf = world_points_conf.reshape(world_points_conf.shape[0], world_points_conf.shape[1], -1, 1)

        # VGGT 论文 / README：Table2 稠密几何用 depth + 预测相机反投影，通常优于 point head 的 world_points。
        # GeometryVisualizer(dtu_official_ply_export) 优先写 xyz_bcd；此前未设置时会误用 xyz_map。
        #
        # pose_encoding_to_extri_intri 的 image_size_hw 必须与深度图/网络输入分辨率一致（如 518×518）。
        # 若误用 batch.meta 里的原始 DTU 分辨率，fx/fy/cx/cy 与 depth 网格不匹配，反投影会塌成错误 3D，
        # 评测侧出现 n_obsmask=0、acc 异常等。
        use_gt_cam = self.depth_xyz_use_gt_camera or (
            os.environ.get("DTU_DEPTH_USE_GT_CAMERA", "").strip() == "1"
        )
        if not self.training and "depth" in predictions:
            depth = predictions["depth"]
            if depth.dim() == 4:
                depth_hw = depth.unsqueeze(-1)
            else:
                depth_hw = depth
            B, S = depth_hw.shape[:2]
            H_d, W_d = int(depth_hw.shape[2]), int(depth_hw.shape[3])
            depth_z = depth_hw[..., 0]
            xyz_frames = []
            if (
                use_gt_cam
                and hasattr(batch, "w2cs")
                and hasattr(batch, "ixts")
                and batch.w2cs is not None
                and batch.ixts is not None
            ):
                # 论文 Table2 **上半（Known GT camera）**：预测深度 + 数据集 GT 内外参反投影（对齐 Gipuma/MVSNet 等设定）。
                w2c_all = batch.w2cs[..., :3, :4].to(device=depth_z.device, dtype=depth_z.dtype)
                K_all = batch.ixts.to(device=depth_z.device, dtype=depth_z.dtype)
                for b in range(B):
                    w2c = w2c_all[b]
                    K = K_all[b]
                    ray_o, ray_d = get_rays(
                        H_d,
                        W_d,
                        K,
                        w2c[..., :3, :3],
                        w2c[..., :3, 3:4],
                        z_depth=True,
                        correct_pix=True,
                    )
                    xyz_b = ray_o + ray_d * depth_z[b, ..., None]
                    xyz_frames.append(xyz_b.reshape(1, S, -1, 3))
                output.xyz_bcd = torch.cat(xyz_frames, dim=0)
            elif "cam_map" in output:
                # Table2 **下半（Ours / DUSt3R）**：预测深度 + 预测相机。
                extrinsic, intrinsic = pose_encoding_to_extri_intri(
                    output.cam_map, image_size_hw=(H_d, W_d)
                )
                for b in range(B):
                    w2c = extrinsic[b]
                    K = intrinsic[b]
                    ray_o, ray_d = get_rays(
                        H_d,
                        W_d,
                        K,
                        w2c[..., :3, :3],
                        w2c[..., :3, 3:4],
                        z_depth=True,
                        correct_pix=True,
                    )
                    xyz_b = ray_o + ray_d * depth_z[b, ..., None]
                    xyz_frames.append(xyz_b.reshape(1, S, -1, 3))
                output.xyz_bcd = torch.cat(xyz_frames, dim=0)

        if self.training or compute_loss:
            if self.loss_fn is None:
                raise ValueError("loss_cfg is required for training")
            depth_loss_dict = predictions.pop("depth_loss_dict", None)
            point_loss_dict = predictions.pop("point_loss_dict", None)

            loss_predictions = dict(predictions)
            if depth_loss_dict is not None:
                loss_predictions.pop("depth", None)
                loss_predictions.pop("depth_conf", None)
            if point_loss_dict is not None:
                loss_predictions.pop("world_points", None)
                loss_predictions.pop("world_points_conf", None)

            loss_dict = self.loss_fn(loss_predictions, official_batch)
            loss_for_backward = loss_dict["objective"]
            objective_log = loss_dict["objective"]

            scalar_stats = dotdict()
            for key, value in loss_dict.items():
                if key == "objective":
                    continue
                scalar_stats[key] = value

            if depth_loss_dict is not None:
                depth_weight = float(self.loss_cfg.get("depth", {}).get("weight", 1.0))
                depth_loss = (
                    depth_loss_dict["loss_conf_depth"]
                    + depth_loss_dict["loss_reg_depth"]
                    + depth_loss_dict["loss_grad_depth"]
                ) * depth_weight
                objective_log = objective_log + depth_loss.detach()
                scalar_stats.loss_depth = depth_loss.detach()
                for key, value in depth_loss_dict.items():
                    scalar_stats[key] = value

            if point_loss_dict is not None:
                point_weight = float(self.loss_cfg.get("point", {}).get("weight", 1.0))
                point_loss = (
                    point_loss_dict["loss_conf_point"]
                    + point_loss_dict["loss_reg_point"]
                    + point_loss_dict["loss_grad_point"]
                ) * point_weight
                objective_log = objective_log + point_loss.detach()
                scalar_stats.loss_point = point_loss.detach()
                for key, value in point_loss_dict.items():
                    scalar_stats[key] = value

            indexer_loss = getattr(self.vggt.aggregator, "indexer_loss", None)
            indexer_backwarded = bool(getattr(self.vggt.aggregator, "indexer_loss_backwarded", False))
            if indexer_loss is not None:
                state = self._indexer_state_from_step(self._get_global_step(batch))
                scalar_stats.indexer_loss = indexer_loss.detach()
                scalar_stats.indexer_sparse = float(state.get("sparse", False))
                scalar_stats.indexer_warmup = float(state.get("warmup", False))
                indexer_aux_stats = getattr(self.vggt.aggregator, "indexer_aux_stats", None)
                if indexer_aux_stats:
                    for key, value in indexer_aux_stats.items():
                        scalar_stats[key] = value.detach() if hasattr(value, "detach") else value
                if not indexer_backwarded:
                    if state.get("warmup", False) and self.indexer_cfg.get("warmup_only_indexer_loss", True):
                        loss_for_backward = indexer_loss
                    else:
                        loss_for_backward = loss_for_backward + indexer_loss
                elif state.get("warmup", False) and self.indexer_cfg.get("warmup_only_indexer_loss", True):
                    loss_for_backward = torch.zeros(
                        (), device=loss_for_backward.device, dtype=loss_for_backward.dtype, requires_grad=True
                    )
            layerwise_sync_loss = getattr(self.vggt.aggregator, "indexer_layerwise_sync_loss", None)
            if layerwise_sync_loss is not None:
                loss_for_backward = loss_for_backward + layerwise_sync_loss
                scalar_stats.indexer_layerwise_sync_loss = layerwise_sync_loss.detach()

            if "chunkwise_bp_loss" in batch:
                loss_for_backward = loss_for_backward + batch.chunkwise_bp_loss
                scalar_stats.chunkwise_bp_loss = batch.chunkwise_bp_loss.detach()

            output.loss = loss_for_backward
            scalar_stats.loss_objective = objective_log.detach()
            scalar_stats.loss = (loss_for_backward.detach())
            if depth_loss_dict is not None:
                scalar_stats.loss = scalar_stats.loss + depth_loss.detach()
            if point_loss_dict is not None:
                scalar_stats.loss = scalar_stats.loss + point_loss.detach()

            output.scalar_stats = scalar_stats
            output.image_stats = dotdict()

        return output
