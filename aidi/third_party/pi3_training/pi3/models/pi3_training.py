import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points, se3_inverse
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope, DSABlockRope
from .layers.attention import resolve_attention_backend
from .layers.transformer_head import TransformerDecoder, LinearPts3d, ContextTransformerDecoder
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
from easyvolcap.utils.base_utils import dotdict

def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False


def pose_prior_to_vggt_like_features(
    pose_prior,
    intrinsics=None,
    image_hw=None,
    pose_prior_mask=None,
    source_flag=0.0,
    pose_prior_format="vggt_w2c_quat10",
    return_valid_mask=False,
):
    """Encode c2w pose priors with VGGT business camera embedding format."""
    from easyvolcap.utils.cam_utils import mat_to_quat
    if pose_prior is None:
        return None
    pose_prior_format = str(pose_prior_format or "vggt_w2c_quat10").strip().lower()
    format_aliases = {
        "vggtstrict10": "vggt_w2c_quat10",
        "vggt_strict10": "vggt_w2c_quat10",
        "vggt_w2c10": "vggt_w2c_quat10",
        "w2c10": "vggt_w2c_quat10",
        "w2c_quat10": "vggt_w2c_quat10",
        "relative_w2c_quat10": "vggt_w2c_quat10",
        "c2w10": "c2w_quat10",
        "relative_c2w_quat10": "c2w_quat10",
    }
    pose_prior_format = format_aliases.get(pose_prior_format, pose_prior_format)
    if pose_prior_format not in {"vggt_w2c_quat10", "c2w_quat10"}:
        raise ValueError(
            f"Unsupported pose_prior_format={pose_prior_format!r}; "
            "expected vggt_w2c_quat10 or c2w_quat10."
        )
    if pose_prior.ndim != 4 or pose_prior.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"pose_prior must have shape BxNx3x4 or BxNx4x4, got {tuple(pose_prior.shape)}")

    pose_prior = pose_prior.float()
    if pose_prior.shape[-2:] == (3, 4):
        bottom = pose_prior.new_zeros(*pose_prior.shape[:-2], 1, 4)
        bottom[..., 0, 3] = 1.0
        pose_prior = torch.cat([pose_prior, bottom], dim=-2)

    finite_mask = torch.isfinite(pose_prior).flatten(-2).all(dim=-1)
    if pose_prior_mask is None:
        pose_prior_mask = finite_mask
    else:
        pose_prior_mask = pose_prior_mask.to(device=pose_prior.device, dtype=torch.bool) & finite_mask
    pose_prior_mask = pose_prior_mask & finite_mask[:, :1]

    pose_prior = torch.nan_to_num(pose_prior, nan=0.0, posinf=0.0, neginf=0.0)
    first_w2c = se3_inverse(pose_prior[:, :1])
    rel_c2w = first_w2c @ pose_prior

    translations = rel_c2w[..., :3, 3]
    if translations.shape[1] > 1:
        scale = translations[:, 1:].norm(dim=-1).mean(dim=1, keepdim=True)
    else:
        scale = translations.norm(dim=-1).mean(dim=1, keepdim=True)
    scale = scale.clamp_min(1e-6)
    rel_c2w = rel_c2w.clone()
    rel_c2w[..., :3, 3] = rel_c2w[..., :3, 3] / scale.unsqueeze(-1)

    if pose_prior_format == "vggt_w2c_quat10":
        encoded_pose = se3_inverse(rel_c2w)
    else:
        encoded_pose = rel_c2w
    trans = encoded_pose[..., :3, 3]
    quat = mat_to_quat(encoded_pose[..., :3, :3])

    B, N = pose_prior.shape[:2]
    if intrinsics is not None and image_hw is not None:
        intrinsics = intrinsics.to(device=pose_prior.device, dtype=pose_prior.dtype)
        H, W = image_hw
        H = torch.as_tensor(H, device=pose_prior.device, dtype=pose_prior.dtype)
        W = torch.as_tensor(W, device=pose_prior.device, dtype=pose_prior.dtype)
        fov_y = 2.0 * torch.atan(H / (2.0 * intrinsics[..., 1, 1].clamp_min(1e-6)))
        fov_x = 2.0 * torch.atan(W / (2.0 * intrinsics[..., 0, 0].clamp_min(1e-6)))
        fov = torch.stack([fov_y, fov_x], dim=-1)
    else:
        fov = pose_prior.new_zeros(B, N, 2)

    if isinstance(source_flag, torch.Tensor):
        source_flag = source_flag.to(device=pose_prior.device, dtype=pose_prior.dtype)
        if source_flag.ndim == 0:
            source_flag = source_flag.reshape(1, 1, 1)
        elif source_flag.ndim == 1:
            source_flag = source_flag.reshape(1, -1, 1)
        elif source_flag.ndim == 2:
            source_flag = source_flag.unsqueeze(-1)
        source_flag = source_flag.expand(B, N, 1)
    else:
        source_flag = pose_prior.new_full((B, N, 1), float(source_flag))

    valid = pose_prior_mask.to(dtype=pose_prior.dtype).unsqueeze(-1)
    features = torch.cat([trans, quat, fov, source_flag], dim=-1)
    features = features * valid
    if return_valid_mask:
        return features, pose_prior_mask
    return features

class Pi3(nn.Module):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            load_vggt=True,
            freeze_encoder=True,
            enable_point=True,
            enable_camera=True,
            use_global_points=False,
            train_conf=False,
            num_dec_blk_not_to_checkpoint=4,
            qk_norm_chunk_size=0,
            head_use_checkpoint=False,
            head_view_chunk_size=0,
            ckpt=None,
            indexer_init_ckpt=None,
            indexer_cfg=None,
            encoder_attn_backend="auto",
            decoder_attn_backend="auto",
            use_pose_prior=False,
            pose_prior_required=False,
            pose_prior_dropout=0.0,
            pose_prior_format="vggt_w2c_quat10",
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False, attn_backend=encoder_attn_backend)
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.indexer_cfg = dotdict(
            enabled=False,
            indexer_layers=None,
            n_heads=4,
            head_dim=64,
            topk=512,
            enable_sparse=True,
            inference_sparse=False,
            warmup_steps=0,
            sparse_start_step=0,
            compute_loss=True,
            freeze_indexer=False,
            warmup_only_indexer_loss=True,
            warmup_only_indexer_train=True,
            warmup_optimizer_filter=True,
            loss_weight=1.0,
            warmup_loss_weight=1.0,
            sparse_loss_weight=1.0,
            eps=1e-6,
            detach_input=True,
            score_dtype="",
            head_chunk_size=0,
            score_head_chunk_size=0,
            score_key_chunk_size=0,
            streaming_kl_loss=False,
            streaming_kl_autograd=False,
            warmup_indexer_loss_mode="kl",
            warmup_topk_coverage_k=1024,
            warmup_topk_coverage_chunk_size=128,
            warmup_topk_coverage_query_chunk_size=256,
            warmup_topk_coverage_query_sample_size=0,
            warmup_no_grad_attn=True,
            sparse_use_mask=False,
            use_sparse_flash_attn=True,
            use_topk_kernel=True,
            use_dense_flash_attn_warmup_kernel=False,
            topk_block=256,
            force_keep_special_tokens=False,
            force_keep_special_in_topk_budget=False,
            force_keep_register_tokens=False,
            force_keep_self_view_tokens=False,
            objective_value_gate_enabled=False,
            objective_value_gate_scale=0.0,
            objective_value_gate_tau=1.0,
            objective_value_gate_eps=1e-6,
            objective_value_gate_query_chunk_size=64,
            init_from_attn=True,
        )
        if indexer_cfg is not None:
            self.indexer_cfg.update(indexer_cfg)

        self.global_depth = dec_depth // 2
        self.indexer_layers = self._parse_indexer_layers(self.indexer_cfg.get("indexer_layers", None))
        if self.indexer_layers is not None:
            self.indexer_layers = {idx for idx in self.indexer_layers if 0 <= idx < self.global_depth}

        block_kwargs = dict(
            dim=dec_embed_dim,
            num_heads=dec_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            drop_path=0.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            ffn_layer=Mlp,
            init_values=0.01,
            qk_norm=True,
            qk_norm_chunk_size=int(qk_norm_chunk_size or 0),
            rope=self.rope,
        )
        decoder_attn_class, self.decoder_attn_backend = resolve_attention_backend(decoder_attn_backend, rope=True)

        decoder_blocks = []
        global_layer_idx = 0
        for layer_idx in range(dec_depth):
            is_global = (layer_idx % 2 == 1)
            use_indexer = False
            if is_global and self.indexer_cfg.get("enabled", False):
                use_indexer = self.indexer_layers is None or global_layer_idx in self.indexer_layers
            if use_indexer:
                decoder_blocks.append(DSABlockRope(indexer_cfg=self.indexer_cfg, **block_kwargs))
            else:
                decoder_blocks.append(BlockRope(attn_class=decoder_attn_class, **block_kwargs))
            if is_global:
                global_layer_idx += 1

        self.decoder = nn.ModuleList(decoder_blocks)
        self.dec_embed_dim = dec_embed_dim
        self.indexer_state = dotdict(enabled=False)
        self.indexer_loss = None
        self._indexer_stage = "none"
        self._base_requires_grad = {}
        self.enable_point = bool(enable_point)
        self.enable_camera = bool(enable_camera)
        self.head_use_checkpoint = bool(head_use_checkpoint)
        self.head_view_chunk_size = int(head_view_chunk_size)
        self.use_pose_prior = bool(use_pose_prior)
        self.pose_prior_required = bool(pose_prior_required)
        self.pose_prior_dropout = float(pose_prior_dropout or 0.0)
        self.pose_prior_format = str(pose_prior_format or "vggt_w2c_quat10")

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)
        if self.use_pose_prior:
            self.pose_prior_embed = nn.Sequential(
                nn.Linear(10, self.dec_embed_dim),
                nn.GELU(),
                nn.Linear(self.dec_embed_dim, self.dec_embed_dim),
            )
            nn.init.zeros_(self.pose_prior_embed[-1].weight)
            nn.init.zeros_(self.pose_prior_embed[-1].bias)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        if self.enable_point:
            self.point_decoder = TransformerDecoder(
                in_dim=2*self.dec_embed_dim, 
                dec_embed_dim=1024,
                dec_num_heads=16,
                out_dim=1024,
                rope=self.rope,
                use_checkpoint=self.head_use_checkpoint,
            )
            self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        if self.enable_camera:
            self.camera_decoder = TransformerDecoder(
                in_dim=2*self.dec_embed_dim, 
                dec_embed_dim=1024,
                dec_num_heads=16,                # 8
                out_dim=512,
                rope=self.rope,
                use_checkpoint=self.head_use_checkpoint,
            )
            self.camera_head = CameraHead(dim=512)
        

        # ----------------------
        #  Global Points Decoder
        # ----------------------
        self.use_global_points = use_global_points
        if use_global_points:
            self.global_points_decoder = ContextTransformerDecoder(
                in_dim=2*self.dec_embed_dim, 
                dec_embed_dim=1024,
                dec_num_heads=16,
                out_dim=1024,
                rope=self.rope,
            )
            self.global_point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        if load_vggt:
            vggt_weight = load_file('ckpts/VGGT-1B/model.safetensors')
            vggt_enc_weight = {k.replace('aggregator.patch_embed.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.patch_embed.')}
            print("Loading vggt encoder", self.encoder.load_state_dict(vggt_enc_weight, strict=False))

            vggt_dec_weight = {k.replace('aggregator.global_blocks.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.global_blocks.')}
            vggt_dec_weight1 = {}
            for k in list(vggt_dec_weight.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight1[f'{int(idx)*2 + 1}{other}'] = vggt_dec_weight[k]
            vggt_dec_weight = vggt_dec_weight1 

            vggt_dec_weight_frame = {k.replace('aggregator.frame_blocks.', ''):vggt_weight[k] for k in list(vggt_weight.keys()) if k.startswith('aggregator.frame_blocks.')}
            for k in list(vggt_dec_weight_frame.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight[f'{int(idx)*2}{other}'] = vggt_dec_weight_frame[k]

            print("Loading vggt decoder", self.decoder.load_state_dict(vggt_dec_weight, strict=False))
            self._init_indexer_from_attention()

        self.train_conf = train_conf
        if train_conf:
            if not self.enable_point:
                raise ValueError("train_conf=True requires enable_point=True")
            assert ckpt is not None

            # ----------------------
            #     Conf Decoder
            # ----------------------
            self.conf_decoder = deepcopy(self.point_decoder)
            self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

            modules_to_freeze = [self.encoder, self.decoder, self.register_token]
            if self.enable_point:
                modules_to_freeze.extend([self.point_decoder, self.point_head])
            if self.enable_camera:
                modules_to_freeze.extend([self.camera_decoder, self.camera_head])
            freeze_all_params(modules_to_freeze)
            if use_global_points:
                freeze_all_params([self.global_points_decoder, self.global_point_head])

        if freeze_encoder:
            print('Freezing the encoder.')
            freeze_all_params([self.encoder])

        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint

        if ckpt is not None:
            ckpt_path = str(ckpt)
            if ckpt_path.endswith('.safetensors'):
                checkpoint = load_file(ckpt_path)
            else:
                checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cpu')

            state_dict = checkpoint
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']

            res = self.load_state_dict(state_dict, strict=False)
            print(f'[Pi3] Load checkpoints from {ckpt}: {res}')
            missing_indexer_keys = [
                key for key in getattr(res, 'missing_keys', [])
                if '.attn.indexer.' in key
            ]
            loaded_indexer_keys = [
                key for key in state_dict.keys()
                if '.attn.indexer.' in key
            ]
            if missing_indexer_keys and not loaded_indexer_keys:
                self._init_indexer_from_attention()

            del checkpoint
            torch.cuda.empty_cache()

        if indexer_init_ckpt is not None:
            self._load_indexer_init_checkpoint(indexer_init_ckpt)

        self._base_requires_grad = {
            name: param.requires_grad for name, param in self.named_parameters()
        }
        if (
            self.indexer_cfg.get("enabled", False)
            and self.indexer_cfg.get("warmup_only_indexer_train", False)
            and int(self.indexer_cfg.get("warmup_steps", 0) or 0) > 0
            and self._indexer_cfg_bool("warmup_optimizer_filter", True)
        ):
            self._set_indexer_trainable(True)
            self._indexer_stage = "warmup"
        elif self.indexer_cfg.get("enabled", False) and self._indexer_cfg_bool("freeze_indexer", False):
            self._set_indexer_frozen(True)

    @staticmethod
    def _parse_indexer_layers(indexer_layers):
        if indexer_layers is None:
            return None
        if isinstance(indexer_layers, str):
            text = indexer_layers.strip()
            if not text:
                return set()
            lowered = text.lower()
            if lowered in ("all", "*"):
                return None
            if "-" in text:
                left, right = text.split("-", 1)
                left = left.strip()
                right = right.strip()
                if left and right:
                    return set(range(int(left), int(right) + 1))
            items = [item.strip() for item in text.split(",") if item.strip()]
            return set(int(item) for item in items)
        if isinstance(indexer_layers, (list, tuple, set)):
            return set(int(item) for item in indexer_layers)
        return None

    @staticmethod
    def _normalize_view_chunk_size(chunk_size, total_views):
        if chunk_size is None:
            return 0
        chunk_size = int(chunk_size)
        if chunk_size <= 0 or chunk_size >= int(total_views):
            return 0
        return chunk_size

    def _run_decoder_with_view_chunking(self, decoder, hidden, xpos=None, chunk_size=0):
        chunk_size = self._normalize_view_chunk_size(chunk_size, hidden.shape[0])
        if chunk_size == 0:
            return decoder(hidden, xpos=xpos)

        outputs = []
        for start in range(0, hidden.shape[0], chunk_size):
            end = min(start + chunk_size, hidden.shape[0])
            xpos_chunk = None if xpos is None else xpos[start:end]
            outputs.append(decoder(hidden[start:end], xpos=xpos_chunk))
        return torch.cat(outputs, dim=0)

    def _run_linear_head_with_view_chunking(self, head, hidden, img_shape, chunk_size=0):
        chunk_size = self._normalize_view_chunk_size(chunk_size, hidden.shape[0])
        if chunk_size == 0:
            return head([hidden], img_shape)

        outputs = []
        for start in range(0, hidden.shape[0], chunk_size):
            end = min(start + chunk_size, hidden.shape[0])
            outputs.append(head([hidden[start:end]], img_shape))
        return torch.cat(outputs, dim=0)

    def _run_camera_head_with_view_chunking(self, head, hidden, patch_h, patch_w, chunk_size=0):
        chunk_size = self._normalize_view_chunk_size(chunk_size, hidden.shape[0])
        if chunk_size == 0:
            return head(hidden, patch_h, patch_w)

        outputs = []
        for start in range(0, hidden.shape[0], chunk_size):
            end = min(start + chunk_size, hidden.shape[0])
            outputs.append(head(hidden[start:end], patch_h, patch_w))
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _extract_checkpoint_state_dict(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("model", "state_dict", "model_state_dict"):
                value = checkpoint.get(key, None)
                if hasattr(value, "items"):
                    return value
        return checkpoint

    @staticmethod
    def _normalize_indexer_checkpoint_key(key: str) -> str:
        key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True

        if "pi3.decoder." in key:
            return "decoder." + key.split("pi3.decoder.", 1)[1]
        if "decoder." in key:
            return key[key.find("decoder."):]
        return key

    def _load_indexer_init_checkpoint(self, ckpt):
        if not self.indexer_cfg.get("enabled", False):
            raise RuntimeError(
                "indexer_init_ckpt was provided, but indexer_cfg.enabled=False; "
                "enable the indexer before loading indexer weights."
            )

        ckpt_path = str(ckpt)
        if ckpt_path.endswith(".safetensors"):
            checkpoint = load_file(ckpt_path)
        else:
            checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")

        state_dict = self._extract_checkpoint_state_dict(checkpoint)
        if not hasattr(state_dict, "items"):
            raise RuntimeError(f"Unsupported indexer checkpoint format: {ckpt_path}")

        target_state = self.state_dict()
        indexer_state = {}
        source_indexer_keys = 0
        skipped_shape = []
        for src_key, value in state_dict.items():
            if "indexer" not in str(src_key):
                continue
            if not torch.is_tensor(value):
                continue
            source_indexer_keys += 1
            dst_key = self._normalize_indexer_checkpoint_key(src_key)
            if "indexer" not in dst_key or dst_key not in target_state:
                continue
            if tuple(value.shape) != tuple(target_state[dst_key].shape):
                skipped_shape.append((src_key, dst_key, tuple(value.shape), tuple(target_state[dst_key].shape)))
                continue
            indexer_state[dst_key] = value

        if not indexer_state:
            details = ""
            if skipped_shape:
                details = f" Shape mismatches: {skipped_shape[:5]}"
            raise RuntimeError(
                f"No indexer parameters loaded from {ckpt_path}. "
                f"source_indexer_keys={source_indexer_keys}.{details}"
            )

        self.load_state_dict(indexer_state, strict=False)
        print(
            f"[Pi3] Loaded {len(indexer_state)} indexer parameters "
            f"from {ckpt_path} (source_indexer_keys={source_indexer_keys})."
        )

        del checkpoint
        torch.cuda.empty_cache()

    def _set_indexer_trainable(self, only_indexer: bool) -> None:
        for name, param in self.named_parameters():
            if "indexer" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False if only_indexer else self._base_requires_grad.get(name, True)

    @staticmethod
    def _as_bool(value, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _indexer_cfg_bool(self, key: str, default: bool = False) -> bool:
        return self._as_bool(self.indexer_cfg.get(key, default), default=default)

    def _set_indexer_frozen(self, freeze_indexer: bool) -> None:
        for name, param in self.named_parameters():
            if "indexer" in name:
                param.requires_grad = False if freeze_indexer else self._base_requires_grad.get(name, True)
            else:
                param.requires_grad = self._base_requires_grad.get(name, True)

    def _indexer_state_from_step(self, step: int, training: bool = True):
        if not self.indexer_cfg.get("enabled", False):
            return dotdict(enabled=False)

        warmup_steps = int(self.indexer_cfg.get("warmup_steps", 0))
        sparse_start = int(self.indexer_cfg.get("sparse_start_step", warmup_steps))
        warmup = bool(training) and step < warmup_steps
        sparse = bool(self.indexer_cfg.get("enable_sparse", True)) and step >= sparse_start
        if not training and self.indexer_cfg.get("inference_sparse", False):
            sparse = True

        loss_weight = float(self.indexer_cfg.get("loss_weight", 1.0))
        if warmup:
            loss_weight = float(self.indexer_cfg.get("warmup_loss_weight", loss_weight))
        elif sparse:
            loss_weight = float(self.indexer_cfg.get("sparse_loss_weight", loss_weight))

        compute_loss = bool(training) and self._indexer_cfg_bool("compute_loss", True)

        return dotdict(
            enabled=True,
            warmup=warmup,
            sparse=sparse,
            compute_loss=compute_loss,
            topk=int(self.indexer_cfg.get("topk", 512)),
            detach_input=bool(self.indexer_cfg.get("detach_input", True)),
            score_dtype=self.indexer_cfg.get("score_dtype", ""),
            head_chunk_size=int(self.indexer_cfg.get("head_chunk_size", 0) or 0),
            score_head_chunk_size=int(self.indexer_cfg.get("score_head_chunk_size", 0) or 0),
            score_key_chunk_size=int(self.indexer_cfg.get("score_key_chunk_size", 0) or 0),
            streaming_kl_loss=bool(self.indexer_cfg.get("streaming_kl_loss", False)),
            streaming_kl_autograd=bool(self.indexer_cfg.get("streaming_kl_autograd", False)),
            warmup_indexer_loss_mode=self.indexer_cfg.get("warmup_indexer_loss_mode", "kl"),
            warmup_topk_coverage_k=int(
                self.indexer_cfg.get("warmup_topk_coverage_k", self.indexer_cfg.get("topk", 512))
            ),
            warmup_topk_coverage_chunk_size=int(
                self.indexer_cfg.get("warmup_topk_coverage_chunk_size", 128)
            ),
            warmup_topk_coverage_query_chunk_size=int(
                self.indexer_cfg.get("warmup_topk_coverage_query_chunk_size", 256)
            ),
            warmup_topk_coverage_query_sample_size=int(
                self.indexer_cfg.get("warmup_topk_coverage_query_sample_size", 0)
            ),
            warmup_only_indexer_loss=bool(self.indexer_cfg.get("warmup_only_indexer_loss", True)),
            warmup_no_grad_attn=bool(self.indexer_cfg.get("warmup_no_grad_attn", True)),
            sparse_use_mask=bool(self.indexer_cfg.get("sparse_use_mask", False)),
            force_keep_special_tokens=bool(self.indexer_cfg.get("force_keep_special_tokens", False)),
            force_keep_special_in_topk_budget=bool(self.indexer_cfg.get("force_keep_special_in_topk_budget", False)),
            force_keep_register_tokens=bool(self.indexer_cfg.get("force_keep_register_tokens", False)),
            force_keep_self_view_tokens=bool(self.indexer_cfg.get("force_keep_self_view_tokens", False)),
            use_sparse_flash_attn=bool(self.indexer_cfg.get("use_sparse_flash_attn", True)),
            use_dense_flash_attn_warmup_kernel=bool(
                self.indexer_cfg.get("use_dense_flash_attn_warmup_kernel", False)
            ),
            loss_weight=loss_weight,
            eps=float(self.indexer_cfg.get("eps", 1e-6)),
        )

    def set_indexer_state(self, state: dict):
        self.indexer_state = dotdict(state)

    def get_indexer_state(self):
        return dotdict(self.indexer_state)

    def set_indexer_state_by_step(self, step: int, training: bool = True):
        state = self._indexer_state_from_step(step, training=training)
        self.set_indexer_state(state)

        if not self.indexer_cfg.get("enabled", False):
            return state

        stage = "warmup" if state.get("warmup", False) else "sparse" if state.get("sparse", False) else "dense"
        if training and stage != self._indexer_stage:
            if self.indexer_cfg.get("warmup_only_indexer_train", True):
                if stage == "warmup":
                    self._set_indexer_trainable(True)
                elif self._indexer_cfg_bool("freeze_indexer", False):
                    self._set_indexer_frozen(True)
                else:
                    self._set_indexer_trainable(False)
            elif self._indexer_cfg_bool("freeze_indexer", False):
                self._set_indexer_frozen(stage == "sparse")
            self._indexer_stage = stage
        return state

    def _indexer_state_for_global_layer(self, layer_idx: int):
        state = dotdict(self.indexer_state)
        state.layer_idx = layer_idx
        if not state.get("enabled", False):
            return state
        if self.indexer_layers is None or layer_idx in self.indexer_layers:
            return state
        state.enabled = False
        state.sparse = False
        state.compute_loss = False
        return state

    def _init_indexer_from_attention(self):
        if not self.indexer_cfg.get("enabled", False):
            return
        if not self.indexer_cfg.get("init_from_attn", True):
            return

        initialized_layers = 0
        for blk in self.decoder:
            if not isinstance(blk, DSABlockRope):
                continue
            if not hasattr(blk.attn, "indexer") or blk.attn.indexer is None:
                continue
            qkv = blk.attn.qkv
            indexer = blk.attn.indexer
            dim = qkv.in_features
            indexer_dim = int(indexer.n_heads) * int(indexer.head_dim)
            if indexer_dim > dim:
                raise ValueError(
                    f"Indexer projection dim ({indexer_dim}) exceeds attention dim ({dim})."
                )

            with torch.no_grad():
                qkv_weight = qkv.weight.data
                q_weight = qkv_weight[:dim]
                k_weight = qkv_weight[dim:2 * dim]
                indexer.q_proj.weight.copy_(q_weight[:indexer_dim])
                indexer.k_proj.weight.copy_(k_weight[:indexer_dim])

                if qkv.bias is not None:
                    qkv_bias = qkv.bias.data
                    q_bias = qkv_bias[:dim]
                    k_bias = qkv_bias[dim:2 * dim]
                    if indexer.q_proj.bias is not None:
                        indexer.q_proj.bias.copy_(q_bias[:indexer_dim])
                    if indexer.k_proj.bias is not None:
                        indexer.k_proj.bias.copy_(k_bias[:indexer_dim])

                nn.init.zeros_(indexer.w_proj.weight)
                if indexer.w_proj.bias is not None:
                    nn.init.ones_(indexer.w_proj.bias)

            initialized_layers += 1

        if initialized_layers > 0:
            print(f"[Pi3] Initialized indexer projections from decoder attention for {initialized_layers} layers.")

    def _prepare_pose_prior_embedding(self, pose_prior, intrinsics, H, W):
        if not self.use_pose_prior:
            return None
        if pose_prior is None:
            if self.pose_prior_required:
                raise RuntimeError("model.use_pose_prior=True requires batch pose_prior, but none was provided.")
            return None

        features, valid = pose_prior_to_vggt_like_features(
            pose_prior=pose_prior,
            intrinsics=intrinsics,
            image_hw=(H, W),
            source_flag=0.0,
            pose_prior_format=self.pose_prior_format,
            return_valid_mask=True,
        )
        if self.pose_prior_required and not bool(valid.all()):
            missing = int((~valid).sum().item())
            raise RuntimeError(f"pose_prior_required=True but {missing} view priors are invalid.")

        if self.training and self.pose_prior_dropout > 0.0:
            keep = torch.rand(valid.shape, device=features.device) >= self.pose_prior_dropout
            keep = keep & valid
            features = features.clone()
            features = features * keep.unsqueeze(-1).to(features.dtype)

        return self.pose_prior_embed(features.to(dtype=self.pose_prior_embed[0].weight.dtype)).to(dtype=pose_prior.dtype)

    def decode(self, hidden, N, H, W, collect_head_input=True, pose_prior_embedding=None):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = [] if collect_head_input else None
        
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])
        if pose_prior_embedding is not None:
            pose_prior_embedding = pose_prior_embedding.reshape(B * N, 1, -1).to(
                device=register_token.device,
                dtype=register_token.dtype,
            )
            register_token = register_token.clone()
            register_token[:, :1, :] = register_token[:, :1, :] + pose_prior_embedding

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        use_layerwise_indexer_backward = bool(
            self.training
            and self.indexer_cfg.get("warmup_only_indexer_train", False)
            and self.indexer_cfg.get("layerwise_backward", False)
            and self.indexer_cfg.get("detach_input", False)
        )
        indexer_loss_total = [] if use_layerwise_indexer_backward else None
        global_layer_idx = 0
        if self.indexer_cfg.get("enabled", False):
            self.indexer_state.tokens_per_view = hw
            self.indexer_state.num_views = N
            self.indexer_state.patch_start_idx = self.patch_start_idx
            self.indexer_state.camera_tokens_per_view = 0

        for i in range(len(self.decoder)):
            blk = self.decoder[i]
            is_global = (i % 2 == 1)
            is_dsa = isinstance(blk, DSABlockRope)

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            if is_dsa:
                layer_indexer_state = self._indexer_state_for_global_layer(global_layer_idx)
                blk.indexer_state = layer_indexer_state
                blk.return_indexer_loss = bool(layer_indexer_state.get("compute_loss", False))

            if i >= self.num_dec_blk_not_to_checkpoint and self.training:
                out = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
            else:
                out = blk(hidden, xpos=pos)

            if is_dsa:
                if blk.return_indexer_loss:
                    hidden, indexer_loss = out
                else:
                    hidden, indexer_loss = out, None
                if indexer_loss is not None:
                    if use_layerwise_indexer_backward:
                        indexer_loss_total.append(indexer_loss)
                    else:
                        indexer_loss_total = indexer_loss if indexer_loss_total is None else indexer_loss_total + indexer_loss
            else:
                hidden = out

            if is_global:
                global_layer_idx += 1

            if collect_head_input and i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        self.indexer_loss = indexer_loss_total
        if not collect_head_input:
            return None, None
        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    
    def forward(self, imgs, warmup_only_indexer_loss=False, pose_prior=None, pose_prior_intrinsics=None):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        pose_prior_embedding = self._prepare_pose_prior_embedding(
            pose_prior=pose_prior,
            intrinsics=pose_prior_intrinsics,
            H=H,
            W=W,
        )
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        hidden, pos = self.decode(
            hidden,
            N,
            H,
            W,
            collect_head_input=not warmup_only_indexer_loss,
            pose_prior_embedding=pose_prior_embedding,
        )
        if warmup_only_indexer_loss:
            return dict(indexer_warmup_only=True)
        if not self.enable_point or not self.enable_camera:
            raise RuntimeError("Non-warmup Pi3 forward requires point/camera heads to be enabled.")

        point_hidden = self._run_decoder_with_view_chunking(
            self.point_decoder,
            hidden,
            xpos=pos,
            chunk_size=self.head_view_chunk_size,
        )
        if self.train_conf:
            conf_hidden = self._run_decoder_with_view_chunking(
                self.conf_decoder,
                hidden,
                xpos=pos,
                chunk_size=self.head_view_chunk_size,
            )
        camera_hidden = self._run_decoder_with_view_chunking(
            self.camera_decoder,
            hidden,
            xpos=pos,
            chunk_size=self.head_view_chunk_size,
        )
        if self.use_global_points:
            context = hidden.reshape(B, N, patch_h*patch_w+self.patch_start_idx, -1)[:, 0:1].repeat(1, N, 1, 1).reshape(B*N, patch_h*patch_w+self.patch_start_idx, -1)
            global_point_hidden = self.global_points_decoder(hidden, context, xpos=pos, ypos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self._run_linear_head_with_view_chunking(
                self.point_head,
                point_hidden[:, self.patch_start_idx:],
                img_shape=(H, W),
                chunk_size=self.head_view_chunk_size,
            ).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            if self.train_conf:
                conf_hidden = conf_hidden.float()
                conf = self._run_linear_head_with_view_chunking(
                    self.conf_head,
                    conf_hidden[:, self.patch_start_idx:],
                    img_shape=(H, W),
                    chunk_size=self.head_view_chunk_size,
                ).reshape(B, N, H, W, -1)
            else:
                conf = None
                
            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self._run_camera_head_with_view_chunking(
                self.camera_head,
                camera_hidden[:, self.patch_start_idx:],
                patch_h=patch_h,
                patch_w=patch_w,
                chunk_size=self.head_view_chunk_size,
            ).reshape(B, N, 4, 4)

            # Global points
            if self.use_global_points:
                global_point_hidden = global_point_hidden.float()
                global_points = self._run_linear_head_with_view_chunking(
                    self.global_point_head,
                    global_point_hidden[:, self.patch_start_idx:],
                    img_shape=(H, W),
                    chunk_size=self.head_view_chunk_size,
                ).reshape(B, N, H, W, -1)
            else:
                global_points = None
            
            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            global_points=global_points
        )
