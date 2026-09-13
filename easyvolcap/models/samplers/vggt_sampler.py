import torch
import random
import numpy as np
from torch import nn
from einops import rearrange
from typing import List, Literal

import accelerate
from accelerate.utils import AutocastKwargs

from easyvolcap.engine import cfg, args
from easyvolcap.engine import call_from_cfg
from easyvolcap.engine import SAMPLERS, REGRESSORS
from easyvolcap.models.networks.noop_network import NoopNetwork

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.timer_utils import timer
from easyvolcap.utils.ray_utils import get_rays
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import export_pts
from easyvolcap.utils.vggt.layers.mlp import CamMlp
from easyvolcap.utils.vggt.heads.dpt_head import DPTHead, DPTHeadWithChunkwiseBP
from easyvolcap.utils.cam_utils import decode_camera_params
from easyvolcap.utils.acc_utils import get_accelerator_dtype
from easyvolcap.utils.vggt.heads.track_head import TrackHead
from easyvolcap.utils.vggt.heads.camera_head import CameraHead
from easyvolcap.utils.vggt.models.aggregator import Aggregator
from easyvolcap.utils.data_utils import to_precision, remove_batch
from easyvolcap.utils.image_utils import interpolate_image, pad_image
from easyvolcap.utils.net_utils import freeze_module, VolumetricVideoModule


@SAMPLERS.register_module()
class VGGTSampler(VolumetricVideoModule):
    def __init__(self,
                 # Legacy network
                 network: NoopNetwork,

                 # Load pretrained DINOv2 backbone
                 dinov2_ckpt: str = 'data/trained_model/dinov2/dinov2_vitl14_reg4_pretrain.pth',
                 use_dinov2_init_frame_block: bool = True,  # use the init frame block of DINOv2
                 use_dinov2_init_global_block: bool = True,  # use the init global block of DINOv2
                 load_pretrained: bool = False,

                 # Using the VGGT batching or not
                 vggt_batching: bool = False,  # remove the default torch batch dimension

                 # Gradient checkpointing
                 use_checkpoint: bool = False,
                 use_chunkwise_checkpoint: bool = False,  # save more memory
                 use_reentrant: bool = False,
                 dpt_head_use_checkpoint: bool = False,

                 # Aggregator configurations, default use VGGT's encoder
                 agg_regator_cfg: dotdict = dotdict(
                    img_size=518,
                    patch_size=14,
                    embed_dim=1024,
                 ),
                 agg_regator_ckpt: str = 'data/trained_model/vggt/aggregator.pt',

                 # Condition
                 use_ray_map: bool = False,
                 use_cam_emb: bool = False,
                 cam_emb_dropout: float = 0.1,
                 use_3ddr: bool = False,
                 use_3ddr_ratio: float = 0.7,
                 test_with_gt_cam: bool = False,
                 cam_emb_layer: List[int] = [0, None, 1],  # select all layers
                 use_cam_token: bool = False,
                 cam_embed_cfg: dotdict = dotdict(
                     type=CamMlp.__name__,
                     in_features=9,
                     hidden_features=4096,  # embed_dim * mlp_ratio in the Aggregator
                     out_features=4096,  # embed_dim * 4
                 ),

                 # Camera decoder head configurations
                 cam_decoder_cfg: dotdict = dotdict(
                    dim_in=2048,
                 ),
                 cam_decoder_ckpt: str = 'data/trained_model/vggt/camera.pt',
                 # Point decoder head configurations
                 xyz_decoder_cfg: dotdict = dotdict(
                    dim_in=2048,
                    output_dim=4,
                    activation='inv_log',
                    conf_activation='expp1',
                 ),
                 xyz_decoder_ckpt: str = 'data/trained_model/vggt/point.pt',
                 # Deptth decoder head configurations
                 dpt_decoder_cfg: dotdict = dotdict(
                    dim_in=2048,
                    output_dim=2,
                    activation='exp',
                    conf_activation='expp1',
                 ),
                 dpt_decoder_ckpt: str = 'data/trained_model/vggt/depth.pt',
                 # Track decoder head configurations
                 tra_decoder_cfg: dotdict = dotdict(
                    dim_in=2048,
                    patch_size=14,
                 ),
                 tra_decoder_ckpt: str = 'data/trained_model/vggt/track.pt',

                 # Back-projection configuration
                 bp_use_gt_pose: bool = False,

                 # Freeze module list
                 freeze_list: List[str] = [],

                 patch_size: List[int] = [14, 14],
                 dtype = torch.float,
                 low_vram: bool = False,
                 use_xyz_head: bool = True,
                 use_dptpose_as_xyz: bool = False,
                 use_chunkwise_bp_dpt_decoder: bool = False,
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)

        # Condition
        self.use_ray_map = use_ray_map
        self.use_cam_emb = use_cam_emb
        self.cam_emb_dropout = cam_emb_dropout
        self.use_3ddr = use_3ddr
        self.use_3ddr_ratio = use_3ddr_ratio
        self.test_with_gt_cam = test_with_gt_cam
        self.use_cam_token = use_cam_token
        self.cam_embed_cfg = cam_embed_cfg
        self.use_xyz_head = use_xyz_head
        self.use_dptpose_as_xyz = use_dptpose_as_xyz
        self.use_chunkwise_bp_dpt_decoder = use_chunkwise_bp_dpt_decoder

        # Build the aggregator and all the decoders
        self.agg_regator: Aggregator = Aggregator(
            **agg_regator_cfg,
            use_checkpoint=use_checkpoint,
            use_reentrant=use_reentrant,
            use_cam_emb=use_cam_emb,
            cam_emb_layer=cam_emb_layer,
            use_cam_token=use_cam_token,
            cam_embed_cfg=cam_embed_cfg,
            use_chunkwise_checkpoint=use_chunkwise_checkpoint,
            low_vram=low_vram,
        )
        self.cam_decoder: CameraHead = CameraHead(**cam_decoder_cfg, use_checkpoint=use_checkpoint, use_reentrant=use_reentrant)
        if not self.use_chunkwise_bp_dpt_decoder:
            if self.use_xyz_head:
                self.xyz_decoder: DPTHead = DPTHead(**xyz_decoder_cfg, use_checkpoint=dpt_head_use_checkpoint)
            self.dpt_decoder: DPTHead = DPTHead(**dpt_decoder_cfg, use_checkpoint=dpt_head_use_checkpoint)
        else:
            if self.use_xyz_head:
                self.xyz_decoder: DPTHeadWithChunkwiseBP = DPTHeadWithChunkwiseBP(**xyz_decoder_cfg, use_checkpoint=dpt_head_use_checkpoint, supervisor_type="PointSupervisor")
            self.dpt_decoder: DPTHeadWithChunkwiseBP = DPTHeadWithChunkwiseBP(**dpt_decoder_cfg, use_checkpoint=dpt_head_use_checkpoint, supervisor_type="DepthSupervisor")
        # self.tra_decoder: TrackHead  = TrackHead( **tra_decoder_cfg)

        # Record the configurations
        self.agg_regator_cfg = agg_regator_cfg
        self.cam_decoder_cfg = cam_decoder_cfg
        self.xyz_decoder_cfg = xyz_decoder_cfg
        self.dpt_decoder_cfg = dpt_decoder_cfg
        self.tra_decoder_cfg = tra_decoder_cfg
        self.low_vram = low_vram

        # Always load the pretrained DINOv2 backbone
        if dinov2_ckpt is not None:
            dinov2_weight = torch.load(
                dinov2_ckpt,
                map_location='cpu',
                weights_only=False
            )
            self.agg_regator.patch_embed.load_state_dict(dinov2_weight)

            # Maybe initialize the frame and global blocks from DINOv2
            dinov2_weight_modify = dotdict()
            if use_dinov2_init_frame_block or use_dinov2_init_global_block:
                for k, v in dinov2_weight.items():
                    if 'blocks.' in k:
                        dinov2_weight_modify[k.replace('blocks.', '')] = v
                    else:
                        dinov2_weight_modify[k] = v

            # Maybe initialize the frame and global blocks from DINOv2
            if use_dinov2_init_frame_block:
                self.agg_regator.frame_blocks.load_state_dict(
                    dinov2_weight_modify,
                    strict=False
                )
            if use_dinov2_init_global_block:
                self.agg_regator.global_blocks.load_state_dict(
                    dinov2_weight_modify,
                    strict=False
                )
            del dinov2_weight, dinov2_weight_modify  # delete the loaded weights
        else:
            print("not use dinov2 pretrained weights")

        # Load the pretrained weights if set
        if load_pretrained:
            if self.use_cam_emb:
                # Load corresponding weights from the pretrained model
                orig_dict = torch.load(
                    agg_regator_ckpt,
                    map_location='cpu',
                    weights_only=False
                )
                full_dict = self.agg_regator.state_dict()
                full_dict.update({
                    k: v for k, v in orig_dict.items() if k in full_dict
                })
                self.agg_regator.load_state_dict(full_dict)
                # Delete the loaded weights
                del orig_dict, full_dict
            else:
                self.agg_regator.load_state_dict(
                    torch.load(
                        agg_regator_ckpt,
                        map_location='cpu',
                        weights_only=False
                ))
            self.cam_decoder.load_state_dict(
                torch.load(
                    cam_decoder_ckpt,
                    map_location='cpu',
                    weights_only=False
            ))
            if self.use_xyz_head:
                self.xyz_decoder.load_state_dict(
                    torch.load(
                        xyz_decoder_ckpt,
                        map_location='cpu',
                        weights_only=False
                ))
            self.dpt_decoder.load_state_dict(
                torch.load(
                    dpt_decoder_ckpt,
                    map_location='cpu',
                    weights_only=False
            ))
            # self.tra_decoder.load_state_dict(
            #     torch.load(
            #         tra_decoder_ckpt,
            #         map_location='cpu',
            #         weights_only=False
            # ))

        # Freeze the modules, NOTE: use `self.get_submodule()`
        for module in freeze_list:
            try:
                freeze_module(self.get_submodule(module))
            except AttributeError:
                for name, param in self.named_parameters():
                    if name == module:
                        param.requires_grad = False

        # Bookkeepings
        self.dtype = dtype
        self.patch_size = patch_size
        self.vggt_batching = vggt_batching
        self.bp_use_gt_pose = bp_use_gt_pose

    def forward(self, batch: dotdict):
        # Remove the useless batch dimension if using VGGTDataset
        # TODO: maybe use a more general way to remove the batch dimension?
        if self.training and self.vggt_batching:
            batch = remove_batch(batch)

        # Record the middle and output results
        output = dotdict()
        output.update(scalar_stats=dotdict(scale=batch.scale.mean()))

        # Deal with shape things
        B, N = batch.rgb.shape[:2]
        Ho, Wo = batch.meta.H[0].item(), batch.meta.W[0].item()
        # Network dimensions, for painless skip connection
        Hn = int(np.ceil(Ho / self.patch_size[0])) * self.patch_size[0]
        Wn = int(np.ceil(Wo / self.patch_size[1])) * self.patch_size[1]

        # Pad the input images for painless skip connection
        rgbs = rearrange(
            batch.rgb,
            'b n (h w) c -> b n c h w',
            h=Ho, w=Wo
        )  # (B, S, 3, Ho, Wo)
        rgbs = pad_image(rgbs, size=(Hn, Wn))  # (B, S, 3, Hn, Wn)

        # Encode the RGB images
        # NOTE: there is no need to explicitly cast the input dtype to `dtype` since
        #       `torch.cuda.amp.autocast()` will do the job during the pass
        if self.use_cam_emb:
            if self.use_3ddr:
                if self.training:
                    use_gt_cam = random.random() >= self.use_3ddr_ratio  # True → cam, False → cam_3ddr
                else:
                    if self.test_with_gt_cam:
                        use_gt_cam = True
                        print("Warning: test_with_gt_cam is True, use gt as cam input")
                    else:
                        use_gt_cam = False
                cam = batch.cam if use_gt_cam else batch.cam_3ddr
                flag_value = 1.0 if use_gt_cam else 0.0
                cam_flag = torch.full_like(cam[..., :1], flag_value)
                cam = torch.cat([cam, cam_flag], dim=-1)
            else:
                cam = batch.cam

            rgb_feats, idx_patch = self.agg_regator(
                rgbs,
                cam,
                self.training and random.random() < self.cam_emb_dropout
            )
        else:
            rgb_feats, idx_patch = self.agg_regator(rgbs)  # (B, S, L=1041, C=2048)

        # Determine the autocast with block, accelerate or torch
        get_autocast = lambda enabled: (
            cfg.runner.accelerator.autocast(AutocastKwargs(enabled=enabled))
            if getattr(cfg, "runner", None) is not None and getattr(cfg.runner, "accelerator", None) is not None
            else torch.cuda.amp.autocast(enabled=enabled)
        )

        # Turn off the AMP for the decoder heads for better stability
        with get_autocast(False):
            # Camera decoder head
            cam_maps = self.cam_decoder(
                rgb_feats
            )  # [(B, S, 9), ...]

            chunkwise_bp_kwargs = {"batch": batch, "loss_scaler": batch.loss_scaler} if self.use_chunkwise_bp_dpt_decoder and self.training else {}

            # Xyz point decoder head
            if self.use_xyz_head:
                if not self.low_vram:
                    xyz_map, xyz_cnf = self.xyz_decoder(
                        rgb_feats,
                        images=rgbs,
                        patch_start_idx=idx_patch,
                        **chunkwise_bp_kwargs,
                    )  # (B, S, Hn, Wn, 3), (B, S, Hn, Wn)
                else:
                    xyz_map, xyz_cnf = None, None
                    
            # Depth decoder head
            dpt_map, dpt_cnf = self.dpt_decoder(
                rgb_feats,
                images=rgbs,
                patch_start_idx=idx_patch,
                **chunkwise_bp_kwargs,
            )  # (B, S, Hn, Wn, 1), (B, S, Hn, Wn)

            # Restore to the original shape
            if self.use_xyz_head:
                xyz_map = xyz_map[..., :Ho, :Wo, :] if xyz_map is not None else None  # (B, S, Ho, Wo, 3)
                xyz_cnf = xyz_cnf[..., :Ho, :Wo] if xyz_cnf is not None else None  # (B, S, Ho, Wo)
            dpt_map = dpt_map[..., :Ho, :Wo, :]  # (B, S, Ho, Wo, 1)
            dpt_cnf = dpt_cnf[..., :Ho, :Wo]  # (B, S, Ho, Wo)

            if self.use_dptpose_as_xyz:
                w2c, ixt = decode_camera_params(
                    cam_maps[-1], Ho, Wo
                )  # (B, S, 4, 4), (B, S, 3, 3)
                ray_o, ray_d = get_rays(  # (B, S, H, W, 3), (B, S, H, W, 3)
                    Ho, Wo, ixt,
                    w2c[..., :3, :3],
                    w2c[..., :3, 3:],
                    z_depth=True,
                    correct_pix=True,
                )
                xyz_map = ray_o + ray_d * dpt_map  # (B, S, Ho, Wo, 3)
                xyz_cnf = torch.full_like(dpt_cnf, 2)

            # Record the results
            output.cam_maps = cam_maps  # [(B, S, 9), ...], every prediction will be used for supervision
            output.cam_map = cam_maps[-1]  # (B, S, 9), serve as the final inference output
            output.dpt_map = rearrange(dpt_map, 'b s h w c -> b s (h w) c')  # (B, S, P, 1)
            output.dpt_cnf = rearrange(dpt_cnf, 'b s h w -> b s (h w) 1')  # (B, S, P, 1)
            if (self.use_xyz_head and xyz_map is not None) or self.use_dptpose_as_xyz:
                output.xyz_map = rearrange(xyz_map, 'b s h w c -> b s (h w) c')  # (B, S, P, 3)
                output.xyz_cnf = rearrange(xyz_cnf, 'b s h w -> b s (h w) 1')  # (B, S, P, 1)

        # Back-projection if testing
        if not self.training:
            # Decode the predicted camera parameters
            if self.bp_use_gt_pose:
                w2c, ixt = batch.w2cs, batch.ixts
            else:
                w2c, ixt = decode_camera_params(
                    cam_maps[-1], Ho, Wo
                )  # (B, S, 4, 4), (B, S, 3, 3)
            # Generate the rays
            ray_o, ray_d = get_rays(  # (B, S, H, W, 3), (B, S, H, W, 3)
                Ho, Wo, ixt,
                w2c[..., :3, :3],
                w2c[..., :3, 3:],
                z_depth=True,
                correct_pix=True,
            )
            ray_o = ray_o.reshape(B, N, -1, 3)  # (B, S, P, 3)
            ray_d = ray_d.reshape(B, N, -1, 3)  # (B, S, P, 3)
            # Compute the back-projected xyz using the predicted depth and pose
            output.xyz_bcd = ray_o + ray_d * output.dpt_map  # (B, S, P, 3)

        # Update the batch
        batch.output.update(output)
        return batch


def merge_cfg_if_not_exist(cfg_dict, **kwargs):
    for k, v in kwargs.items():
        if k not in cfg_dict:
            cfg_dict[k] = v
    return cfg_dict


@SAMPLERS.register_module()
class VGGTAggSmall(VGGTSampler):
    def __init__(self, **kwargs):
        kwargs["agg_regator_cfg"] = merge_cfg_if_not_exist(kwargs.get("agg_regator_cfg", dotdict()),
            depth=12,
            intermediate_layer_idx=[2, 5, 8, 11],
        )
        kwargs["dpt_decoder_cfg"] = merge_cfg_if_not_exist(kwargs.get("dpt_decoder_cfg", dotdict()),
            dim_in=2048,
            output_dim=2,
            activation='exp',
            conf_activation='expp1',
            intermediate_layer_idx=[2, 5, 8, 11],
        )
        kwargs["xyz_decoder_cfg"] = merge_cfg_if_not_exist(kwargs.get("xyz_decoder_cfg", dotdict()),
            dim_in=2048,
            output_dim=4,
            activation='inv_log',
            conf_activation='expp1',
            intermediate_layer_idx=[2, 5, 8, 11],
        )
        super().__init__(**kwargs)


@SAMPLERS.register_module()
class VGGTAggTiny(VGGTSampler):
    def __init__(self, **kwargs):
        kwargs["agg_regator_cfg"] = merge_cfg_if_not_exist(kwargs.get("agg_regator_cfg", dotdict()),
            depth=6,
            intermediate_layer_idx=[1, 3, 4, 5],
        )
        kwargs["dpt_decoder_cfg"] = merge_cfg_if_not_exist(kwargs.get("dpt_decoder_cfg", dotdict()),
            dim_in=2048,
            output_dim=2,
            activation='exp',
            conf_activation='expp1',
            intermediate_layer_idx=[1, 3, 4, 5],
        )
        kwargs["xyz_decoder_cfg"] = merge_cfg_if_not_exist(kwargs.get("xyz_decoder_cfg", dotdict()),
            dim_in=2048,
            output_dim=4,
            activation='inv_log',
            conf_activation='expp1',
            intermediate_layer_idx=[1, 3, 4, 5],
        )
        super().__init__(**kwargs)
