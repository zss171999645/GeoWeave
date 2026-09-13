# Point loss module
import math
import torch
import random
from torch import nn
from torch.nn import functional as F

from easyvolcap.engine import SUPERVISORS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.depth_utils import normalize
from easyvolcap.utils.chunk_utils import multi_gather
from easyvolcap.utils.math_utils import affine_inverse
from easyvolcap.utils.cam_utils import compute_camera_similarity
from easyvolcap.utils.loss_utils import smoothl1, l1, l2, reg, wreg, conf_grad_loss, smooth, xyz_normal_diff, check_and_fix_inf_nan
from easyvolcap.utils.loss_utils import XyzLossType, ScaleAndShiftInvariantMSELoss, ScaleAndShiftInvariantMAELoss, OptimalAffineInvariantMAELoss, LocalOptimalAffineInvariantMAELoss

@SUPERVISORS.register_module()
class PointSupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module,
                 xyz_loss_weight: float = 0.0,  # xyz point supervison
                 xyz_loss_type: XyzLossType = XyzLossType.ROEMAE.name,  # depth loss type
                 scale_invariant_loss_cfg: dotdict = dotdict(),  # scale invariant depth loss

                 xyz_conf_grad_loss_weight: float = 0.0,  # xyz point supervison with confidence gradient
                 xyz_conf_grad_loss_normalize_prediction: bool = False,
                 xyz_conf_grad_loss_normalize_target: bool = False,
                 xyz_conf_grad_loss_separate: bool = True,
                 xyz_conf_grad_loss_range: float = -1.0,
                 xyz_conf_grad_loss_gamma: float = 1.0,
                 xyz_conf_grad_loss_alpha: float = 0.2,  # default value set by DUSt3R
                 xyz_conf_grad_loss_scales: int = 4,
                 xyz_conf_grad_loss_conf_disable: bool = False,
                 xyz_conf_grad_loss_grad_conf_disable: bool = True,

                 xyz_mask_loss_weight: float = 0.0,  # mask constraint
                 xyz_normal_loss_weight: float = 0.0,  # normal constraint

                 xyz_local_loss_weight: float = 0.0,  # multi-scale depth supervision
                 xyz_local_loss_type: XyzLossType = XyzLossType.LOCALROEMAE.name,  # multi-scale depth loss type
                 local_scale_invariant_loss_cfg: dotdict = dotdict(),  # local scale invariant depth loss
                 xyz_local_radius: List[int] = [4, 16, 64],  # alpha of multi-scale depth supervision
                 xyz_local_aligns: List[int] = [16, 8, 4],  # alignment of multi-scale depth supervision
                 xyz_local_anchor: List[int] = [16, 256, 4096],  # number of multi-scale anchor points

                 xyz_consistency_weight: float = 0.0,  # xyz point consistency supervision
                 xyz_consistency_n_srcs: int = 3,  # number of source views for depth consistency
                 xyz_consistency_extra_src_pool: int = 0,  # number of extra source views
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)
        self.xyz_loss_weight = xyz_loss_weight
        self.xyz_loss_type = XyzLossType[xyz_loss_type]
        self.scale_invariant_loss_cfg = scale_invariant_loss_cfg

        self.xyz_conf_grad_loss_weight = xyz_conf_grad_loss_weight
        self.xyz_conf_grad_loss_normalize_prediction = xyz_conf_grad_loss_normalize_prediction
        self.xyz_conf_grad_loss_normalize_target = xyz_conf_grad_loss_normalize_target
        self.xyz_conf_grad_loss_separate = xyz_conf_grad_loss_separate
        self.xyz_conf_grad_loss_range = xyz_conf_grad_loss_range
        self.xyz_conf_grad_loss_gamma = xyz_conf_grad_loss_gamma
        self.xyz_conf_grad_loss_alpha = xyz_conf_grad_loss_alpha
        self.xyz_conf_grad_loss_scales = xyz_conf_grad_loss_scales
        self.xyz_conf_grad_loss_conf_disable = xyz_conf_grad_loss_conf_disable
        self.xyz_conf_grad_loss_grad_conf_disable = xyz_conf_grad_loss_grad_conf_disable

        self.xyz_mask_loss_weight = xyz_mask_loss_weight
        self.xyz_normal_loss_weight = xyz_normal_loss_weight

        self.xyz_local_loss_weight = xyz_local_loss_weight
        self.xyz_local_loss_type = XyzLossType[xyz_local_loss_type]
        self.local_scale_invariant_loss_cfg = local_scale_invariant_loss_cfg
        self.xyz_local_radius = xyz_local_radius
        self.xyz_local_aligns = xyz_local_aligns
        self.xyz_local_anchor = xyz_local_anchor

        self.xyz_consistency_weight = xyz_consistency_weight
        self.xyz_consistency_n_srcs = xyz_consistency_n_srcs
        self.xyz_consistency_extra_src_pool = xyz_consistency_extra_src_pool

    @property
    def ssimse_loss(self):
        return self.ssimse_loss_reference[0]

    @property
    def ssimae_loss(self):
        return self.ssimae_loss_reference[0]

    @property
    def roemae_loss(self):
        return self.roemae_loss_reference[0]

    @property
    def local_roemae_loss(self):
        return self.local_roemae_loss_reference[0]

    def compute_xyz_loss(self, xyz_map: torch.Tensor, xyz_gt: torch.Tensor, mask: torch.Tensor,
                         H: int, W: int, type=XyzLossType.ROEMAE, **kwargs):
        if type == XyzLossType.SSIMSE:
            if not hasattr(self, 'ssimse_loss_reference'):
                self.ssimse_loss_reference = [ScaleAndShiftInvariantMSELoss(**self.scale_invariant_loss_cfg).cuda().to(self.dtype)]
            xyz_loss = self.ssimse_loss(xyz_map, xyz_gt, mask)
        elif type == XyzLossType.SSIMAE:
            if not hasattr(self, 'ssimae_loss_reference'):
                self.ssimae_loss_reference = [ScaleAndShiftInvariantMAELoss(**self.scale_invariant_loss_cfg).cuda().to(self.dtype)]
            xyz_loss = self.ssimae_loss(xyz_map, xyz_gt, mask)
        elif type == XyzLossType.ROEMAE:
            if not hasattr(self, 'roemae_loss_reference'):
                self.roemae_loss_reference = [OptimalAffineInvariantMAELoss(**self.scale_invariant_loss_cfg).cuda().to(self.dtype)]
            # Need to restore the image layout
            xyz_map = xyz_map.reshape(xyz_map.shape[:-2] + (H, W, 3))  # (B, ..., H, W, 3)
            xyz_gt = xyz_gt.reshape(xyz_gt.shape[:-2] + (H, W, 3))  # (B, ..., H, W, 3)
            mask = mask.reshape(mask.shape[:-2] + (H, W, 1))  # (B, ..., H, W, 1)
            xyz_loss = self.roemae_loss(xyz_map, xyz_gt, mask, **kwargs)
        elif type == XyzLossType.LOCALROEMAE:
            if not hasattr(self, 'local_roemae_loss_reference'):
                self.local_roemae_loss_reference = [LocalOptimalAffineInvariantMAELoss(**self.local_scale_invariant_loss_cfg).cuda().to(self.dtype)]
            xyz_loss = self.local_roemae_loss(xyz_map, xyz_gt, mask, H, W, **kwargs)
        elif type == XyzLossType.SMOOTHL1: xyz_loss = smoothl1(xyz_map[mask], xyz_gt[mask])
        elif type == XyzLossType.L1: xyz_loss = l1(xyz_map[mask], xyz_gt[mask])
        elif type == XyzLossType.L2: xyz_loss = l2(xyz_map[mask], xyz_gt[mask])

        return xyz_loss

    def compute_loss(self, output: dotdict, batch: dotdict, loss: torch.Tensor, scalar_stats: dotdict, image_stats: dotdict):
        # Compute the actual loss here
        def compute_xyz_loss(xyz_map: torch.Tensor, xyz_gt: torch.Tensor, mask: torch.Tensor,
                             H: int = batch.meta.H[0].item(), W: int = batch.meta.W[0].item(),
                             type=self.xyz_loss_type, **kwargs):
            return self.compute_xyz_loss(xyz_map, xyz_gt, mask, H, W, type, **kwargs)

        if 'xyz_map' in output and 'xyz' in batch and \
           self.xyz_loss_weight > 0:
            msk = batch.msk[..., 0] > 0  # (B, S, P)
            xyz_loss = compute_xyz_loss(
                output.xyz_map.float(), batch.xyz.float(), msk,
                batch=output  # for store extra information
            )
            scalar_stats.xyz_loss = xyz_loss
            loss += self.xyz_loss_weight * xyz_loss

        if 'xyz_map' in output and 'xyz' in batch and 'xyz_cnf' in output and \
           self.xyz_conf_grad_loss_weight > 0:
            # Deal with shape issues
            B, S = batch.xyz.shape[:2]
            H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
            msk = batch.msk.clone().reshape(B, S, H, W)  # (B, S, H, W)
            xyz = batch.xyz.clone().reshape(B, S, H, W, 3).float()  # (B, S, H, W, 3)
            xyz_map = output.xyz_map.reshape(B, S, H, W, 3).float()  # (B, S, H, W, 3)
            cnf_map = output.xyz_cnf.reshape(B, S, H, W, 1).float()  # (B, S, H, W, 1)

            # Check and fix the NaN and Inf value in the ground truth xyz
            xyz = check_and_fix_inf_nan(xyz, f'batch.xyz')

            # Normalize the predicted xyz if the option is set
            if self.xyz_conf_grad_loss_normalize_prediction:
                xyz_map, _ = normalize(xyz_map, msk)
            # Normalize the target xyz if the option is set
            if self.xyz_conf_grad_loss_normalize_target:
                xyz, _ = normalize(xyz, msk)

            if self.xyz_conf_grad_loss_separate:
                # Confidence-weighted depth and gradient loss for the first frame
                conf_loss_f, grad_loss_f = conf_grad_loss(
                    xyz_map[:, :1], xyz[:, :1], msk[:, :1], cnf_map[:, :1],
                    conf_disable=self.xyz_conf_grad_loss_conf_disable,
                    grad_conf_disable=self.xyz_conf_grad_loss_grad_conf_disable,
                    range=self.xyz_conf_grad_loss_range,
                    gamma=self.xyz_conf_grad_loss_gamma,
                    alpha=self.xyz_conf_grad_loss_alpha,
                    scales=self.xyz_conf_grad_loss_scales
                )
                # Confidence-weighted depth and gradient loss for the other frames
                if xyz_map.shape[1] > 1:
                    conf_loss_o, grad_loss_o = conf_grad_loss(
                        xyz_map[:, 1:], xyz[:, 1:], msk[:, 1:], cnf_map[:, 1:],
                        conf_disable=self.xyz_conf_grad_loss_conf_disable,
                        grad_conf_disable=self.xyz_conf_grad_loss_grad_conf_disable,
                        range=self.xyz_conf_grad_loss_range,
                        gamma=self.xyz_conf_grad_loss_gamma,
                        alpha=self.xyz_conf_grad_loss_alpha,
                        scales=self.xyz_conf_grad_loss_scales
                    )
                else:
                    conf_loss_o = torch.zeros_like(conf_loss_f)
                    grad_loss_o = torch.zeros_like(grad_loss_f)

                # Perform the confidence gradient loss reduction
                if self.xyz_conf_grad_loss_separate:
                    conf_loss = 0.0
                    conf_loss += conf_loss_f.mean() if conf_loss_f.numel() > 0 else 0.0
                    conf_loss += conf_loss_o.mean() if conf_loss_o.numel() > 0 else 0.0
                else:
                    conf_loss = torch.cat([conf_loss_f, conf_loss_o], dim=1)
                    conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0.0
                # Add the gradient loss since they are already averaged
                grad_loss = grad_loss_f + grad_loss_o
            else:
                conf_loss_all, grad_loss_all = conf_grad_loss(
                    xyz_map, xyz, msk, cnf_map,
                    conf_disable=self.xyz_conf_grad_loss_conf_disable,
                    grad_conf_disable=self.xyz_conf_grad_loss_grad_conf_disable,
                    range=self.xyz_conf_grad_loss_range,
                    gamma=self.xyz_conf_grad_loss_gamma,
                    alpha=self.xyz_conf_grad_loss_alpha,
                    scales=self.xyz_conf_grad_loss_scales
                )
                conf_loss = conf_loss_all.mean() if conf_loss_all.numel() > 0 else 0.0
                grad_loss = grad_loss_all.mean() if grad_loss_all.numel() > 0 else 0.0

            scalar_stats.xyz_conf_loss = conf_loss
            scalar_stats.xyz_grad_loss = grad_loss
            loss += (
                self.xyz_conf_grad_loss_weight * conf_loss +
                self.xyz_conf_grad_loss_weight * grad_loss
            )

        if 'xyz_map' in output and 'xyz' in batch and \
           self.xyz_local_loss_weight > 0:
            # Shape things
            H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
            focal = 1 / (
                1 / (batch.ixts[..., 0, 0] / W) ** 2 + 1 / (batch.ixts[..., 1, 1] / H) ** 2
            ) ** 0.5  # (B, S)

            # Compute the multi-scale supervision
            for r, a, n in zip(self.xyz_local_radius, self.xyz_local_aligns, self.xyz_local_anchor):
                xyz_local_loss = compute_xyz_loss(
                    output.xyz_map.float(), batch.xyz.float(), batch.msk > 0, H, W,
                    type=self.xyz_local_loss_type,
                    focal=focal.float(), level=r, align=a, n_patchs=n,
                    global_scale=output.get('global_scale', None),
                )

                scalar_stats[f'xyz_local_loss_{r:02d}'] = xyz_local_loss
                loss += self.xyz_local_loss_weight * xyz_local_loss

        # if 'xyz_map' in output and 'xyz' in batch and \
        #    self.xyz_local_loss_weight > 0:
        #     # Prepare the mask and inverse depth as loss scaling factor
        #     msk = batch.msk > 0  # (B, S, P, 1)
        #     wet = 1.0 / (batch.dpt / batch.scale)  # (B, S, P, 1)

        #     # Compute the multi-scale depth supervision
        #     xyz_local_loss = 0.0
        #     for idx, rad in zip(batch.anc_inds, batch.anc_rads):  # (B, S, M), (B, S, M)
        #         # TODO: find a better way to avoid the for loop
        #         for i in range(idx.shape[-1]):
        #             dist = torch.cdist(multi_gather(batch.xyz, idx[:, :, i:i+1]), batch.xyz).squeeze(-2)  # (B, S, P)
        #             msk = (dist < rad[:, :, i:i+1])[..., None] & (batch.msk > 0)  # (B, S, P, 1)
        #             # Each anchor point must has at least one valid pixel other than itself
        #             msk = msk * ((dist < rad[:, :, i:i+1])[..., None].sum(dim=-2, keepdim=True) > 1)  # (B, S, P, 1)
        #             msk = msk * (msk.sum(dim=-2, keepdim=True) > 1)  # NOTE: necessary for avoid zero division
        #             if msk.sum() > 0:
        #                 xyz_local_loss += compute_xyz_loss(output.xyz_map, batch.xyz, msk, wet)

        #     scalar_stats.xyz_local_loss = xyz_local_loss
        #     loss += self.xyz_local_loss_weight * xyz_local_loss

        if 'xyz_map' in output and 'xyz' in batch and \
           self.xyz_normal_loss_weight > 0:
            # Shape things
            H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
            msk = batch.msk.reshape(-1, H, W) > 0  # (B * S, H, W)
            xyz = batch.xyz.reshape(-1, H, W, 3)  # (B * S, H, W, 3)
            xyz_map = output.xyz_map.reshape(-1, H, W, 3)  # (B * S, H, W, 3)
            # Compute the normal loss
            xyz_normal_loss = xyz_normal_diff(xyz_map, xyz, msk)
            scalar_stats.xyz_normal_loss = xyz_normal_loss
            loss += self.xyz_normal_loss_weight * xyz_normal_loss

        if 'msk_map' in output and 'msk' in batch and \
           self.xyz_mask_loss_weight > 0:
            msk = batch.msk.float()  # (B, S, P, 1)
            msk_map = output.msk_map.float()  # (B, S, P, 1)
            xyz_mask_loss = msk * (1 - msk_map) + (1 - msk) * msk_map
            xyz_mask_loss = xyz_mask_loss.mean()
            scalar_stats.xyz_mask_loss = xyz_mask_loss

        # TODO: may need more unit tests to verify the correctness
        if 'xyz_map' in output and 'xyz' in batch and \
           self.xyz_consistency_weight > 0:
            # Choose the source views for depth consistency
            _, src_inds = compute_camera_similarity(
                affine_inverse(batch.w2cs).transpose(0, 1),  # (S, B, 4, 4)
                affine_inverse(batch.w2cs).transpose(0, 1)  # (S, B, 4, 4)
            )  # (S, S, B)
            src_inds = src_inds.permute(2, 0, 1)  # (B, S, S)
            # Always exclude the reference view itself
            inds = src_inds[..., 1:1 + self.xyz_consistency_n_srcs + self.xyz_consistency_extra_src_pool]  # (B, S, N')
            rand = torch.randperm(inds.shape[-1], device=inds.device)[:self.xyz_consistency_n_srcs]  # (N,)
            inds = inds[..., rand]  # (B, S, N)

            # Shape things
            B, S, N = inds.shape
            H, W = batch.meta.H[0].item(), batch.meta.W[0].item()

            # Prepare source views information
            src_ixts = multi_gather(
                batch.ixts.reshape(B, S, -1),  # (B, S, 3 * 3)
                inds.reshape(B, -1)  # (B, S * N)
            ).reshape((B, S, N) + batch.ixts.shape[2:])  # (B, S, N, 3, 3)
            src_w2cs = multi_gather(
                batch.w2cs.reshape(B, S, -1),  # (B, S, 4 * 4)
                inds.reshape(B, -1)  # (B, S * N)
            ).reshape((B, S, N) + batch.w2cs.shape[2:])  # (B, S, N, 4, 4)
            src_xyzs = multi_gather(
                output.xyz_map.reshape(B, S, -1),  # (B, S, P * 3)
                inds.reshape(B, -1)  # (B, S * N)
            ).reshape(B * S * N, H, W, 3)  # (B * S * N, H, W, 3)

            # Prepare target views xyz
            tar_xyzs = torch.cat([
                output.xyz_map,
                torch.ones_like(output.xyz_map[..., :1])
            ], dim=-1)[:, :, None]  # (B, S, 1, P, 4)

            # Project reference view xyz to source views
            rep_grid = (tar_xyzs @ src_w2cs[..., :3, :4].mT) @ src_ixts.mT  # (B, S, N, P, 3)
            rep_grid = rep_grid[..., :2] / rep_grid[..., 2:].clip(1e-6)  # (B, S, N, P, 2)
            rep_grid = torch.cat([
                rep_grid[..., :1] / (W - 1) * 2 - 1,
                rep_grid[..., 1:] / (H - 1) * 2 - 1
            ], dim=-1).reshape(B * S * N, H, W, 2)  # (B * S * N, H, W, 2)

            # Interpolate source views xyz
            rep_xyzs = F.grid_sample(
                src_xyzs.permute(0, 3, 1, 2),  # (B * S * N, 3, H, W)
                rep_grid,  # (B * S * N, H, W, 2)
                align_corners=True  # TODO: determine this
            ).permute(0, 2, 3, 1).reshape(B, S, N, -1, 3)  # (B * S * N, 3, H, W) -> (B, S, N, P, 3)

            # Exclude reprojection outside the image
            rep_mask = (rep_grid.abs() <= 1).all(dim=-1).reshape(B, S, N, -1)  # (B * S * N, H, W) -> (B, S, N, P)
            # We only supervise on those valid pixels
            msk = batch.msk[..., 0] > 0  # (B, S, P)
            msk = msk[:, :, None] & rep_mask  # (B, S, N, P)

            # Weight the loss with inverse of the normalized depth
            wet = (1.0 / (batch.dpt / batch.scale).clip(min=1e-6))[..., 0][:, :, None]  # (B, S, 1, P)
            wet = wet / wet[msk].max()  # normalize the weight

            # Compute depth consistency loss only if there are valid pixels
            if msk.sum() > 0:
                xyz_consistency_loss = wreg(
                    rep_xyzs[msk] - output.xyz_map[:, :, None].expand(-1, -1, N, -1, -1)[msk],
                    wet.expand(-1, -1, N, -1)[msk],
                )
            else:
                xyz_consistency_loss = torch.tensor(0.0, device=msk.device)

            scalar_stats.xyz_consistency_loss = xyz_consistency_loss
            loss += self.xyz_consistency_weight * xyz_consistency_loss

        return loss
