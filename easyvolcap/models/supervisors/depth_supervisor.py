# Depth loss module
import torch
from torch import nn
from torch.nn import functional as F

from easyvolcap.engine import SUPERVISORS
from easyvolcap.engine.registry import call_from_cfg

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.depth_utils import normalize
from easyvolcap.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor
from easyvolcap.utils.loss_utils import smoothl1, l1, l2, conf_grad_loss, DptLossType, ScaleAndShiftInvariantMSELoss, ScaleAndShiftInvariantMAELoss, ScaleInvariantLogLoss, check_and_fix_inf_nan


@SUPERVISORS.register_module()
class DepthSupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module,
                 dpt_loss_weight: float = 0.0,  # depth supervison
                 dpt_loss_type: DptLossType = DptLossType.SMOOTHL1.name,  # depth loss type
                 scale_invariant_loss_cfg: dotdict = dotdict(),  # scale invariant depth loss

                 dpt_conf_grad_loss_weight: float = 0.0,  # depth confidence gradient loss
                 dpt_conf_grad_loss_normalize_prediction: bool = False,
                 dpt_conf_grad_loss_normalize_target: bool = False,
                 dpt_conf_grad_loss_separate: bool = True,
                 dpt_conf_grad_loss_range: float = -1.0,
                 dpt_conf_grad_loss_gamma: float = 1.0,
                 dpt_conf_grad_loss_alpha: float = 0.2,  # default value set by DUSt3R
                 dpt_conf_grad_loss_scales: int = 4,
                 dpt_conf_grad_loss_conf_disable: bool = False,
                 dpt_conf_grad_loss_grad_conf_disable: bool = True,

                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)
        self.dpt_loss_weight = dpt_loss_weight
        self.dpt_loss_type = DptLossType[dpt_loss_type]
        self.scale_invariant_loss_cfg = scale_invariant_loss_cfg

        self.dpt_conf_grad_loss_weight = dpt_conf_grad_loss_weight
        self.dpt_conf_grad_loss_normalize_prediction = dpt_conf_grad_loss_normalize_prediction
        self.dpt_conf_grad_loss_normalize_target = dpt_conf_grad_loss_normalize_target
        self.dpt_conf_grad_loss_separate = dpt_conf_grad_loss_separate
        self.dpt_conf_grad_loss_range = dpt_conf_grad_loss_range
        self.dpt_conf_grad_loss_gamma = dpt_conf_grad_loss_gamma
        self.dpt_conf_grad_loss_alpha = dpt_conf_grad_loss_alpha
        self.dpt_conf_grad_loss_scales = dpt_conf_grad_loss_scales
        self.dpt_conf_grad_loss_conf_disable = dpt_conf_grad_loss_conf_disable
        self.dpt_conf_grad_loss_grad_conf_disable = dpt_conf_grad_loss_grad_conf_disable

    @property
    def ssimse_loss(self):
        return self.ssimse_loss_reference[0]

    @property
    def ssimae_loss(self):
        return self.ssimae_loss_reference[0]

    @property
    def silog_loss(self):
        return self.silog_loss_reference[0]

    def compute_depth_loss(self, dpt_map: torch.Tensor, dpt_gt: torch.Tensor, mask: torch.Tensor,
                           H: int, W: int, type=DptLossType.SMOOTHL1):
        if type == DptLossType.SSIMSE:
            if not hasattr(self, 'ssimse_loss_reference'):
                self.ssimse_loss_reference = [ScaleAndShiftInvariantMSELoss(**self.scale_invariant_loss_cfg).cuda().to(self.dtype)]
            dpt_loss = self.ssimse_loss(dpt_map, dpt_gt, mask)
        elif type == DptLossType.SSIMAE:
            if not hasattr(self, 'ssimae_loss_reference'):
                self.ssimae_loss_reference = [ScaleAndShiftInvariantMAELoss(**self.scale_invariant_loss_cfg).cuda().to(self.dtype)]
            dpt_loss = self.ssimae_loss(dpt_map, dpt_gt, mask)
        elif type == DptLossType.SILOG:
            if not hasattr(self, 'silog_loss_reference'):
                self.silog_loss_reference = [ScaleInvariantLogLoss(**self.scale_invariant_loss_cfg).cuda().to(self.dtype)]
            dpt_loss = self.silog_loss(dpt_map, dpt_gt, mask)
        elif type == DptLossType.SMOOTHL1: dpt_loss = smoothl1(dpt_map[mask], dpt_gt[mask])
        elif type == DptLossType.L1: dpt_loss = l1(dpt_map[mask], dpt_gt[mask])
        elif type == DptLossType.L2: dpt_loss = l2(dpt_map[mask], dpt_gt[mask])
        # TODO: Implement depth continuity loss (scale-invariant)
        # TODO: Implement depth ranking loss (scale-invariant)

        return dpt_loss

    def compute_loss(self, output: dotdict, batch: dotdict, loss: torch.Tensor, scalar_stats: dotdict, image_stats: dotdict):
        # Compute the actual loss here
        def compute_depth_loss(dpt_map: torch.Tensor, dpt_gt: torch.Tensor, mask: torch.Tensor,
                               H: int = batch.meta.H[0].item(), W: int = batch.meta.W[0].item(),
                               type=self.dpt_loss_type):
            return self.compute_depth_loss(dpt_map, dpt_gt, mask, H, W, type)

        if 'dpt_map' in output and 'dpt' in batch and \
           self.dpt_loss_weight > 0:
            msk = batch.msk[..., 0] > 0
            dpt_loss = compute_depth_loss(output.dpt_map, batch.dpt, msk)
            scalar_stats.dpt_loss = dpt_loss
            loss += self.dpt_loss_weight * dpt_loss

        if 'dpt_map' in output and 'dpt' in batch and 'dpt_cnf' in output and \
           self.dpt_conf_grad_loss_weight > 0:
            # Deal with shape issues
            B, S = batch.dpt.shape[:2]
            H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
            msk = batch.msk.clone().reshape(B, S, H, W) > 0  # (B, S, H, W)
            dpt = batch.dpt.clone().reshape(B, S, H, W, 1).float()  # (B, S, H, W, 1)
            dpt_map = output.dpt_map.reshape(B, S, H, W, 1).float()  # (B, S, H, W, 1)
            cnf_map = output.dpt_cnf.reshape(B, S, H, W, 1).float()  # (B, S, H, W, 1)

            # Check and fix the NaN and Inf values in the ground truth depth map
            dpt = check_and_fix_inf_nan(dpt, f'batch.dpt')

            # Normalize the predicted depth if the option is set
            if self.dpt_conf_grad_loss_normalize_prediction:
                dpt_map, _ = normalize(dpt_map, msk)
            # Normalize the ground truth depth if the option is set
            if self.dpt_conf_grad_loss_normalize_target:
                dpt, _ = normalize(dpt, msk)


            if self.dpt_conf_grad_loss_separate:
                # Confidence-weighted depth and gradient loss for the first frame
                conf_loss_f, grad_loss_f = conf_grad_loss(
                    dpt_map[:, :1], dpt[:, :1], msk[:, :1], cnf_map[:, :1],
                    conf_disable=self.dpt_conf_grad_loss_conf_disable,
                    grad_conf_disable=self.dpt_conf_grad_loss_grad_conf_disable,
                    range=self.dpt_conf_grad_loss_range,
                    gamma=self.dpt_conf_grad_loss_gamma,
                    alpha=self.dpt_conf_grad_loss_alpha,
                    scales=self.dpt_conf_grad_loss_scales
                )
                # Confidence-weighted depth and gradient loss for the other frames
                if dpt_map.shape[1] > 1:
                    conf_loss_o, grad_loss_o = conf_grad_loss(
                        dpt_map[:, 1:], dpt[:, 1:], msk[:, 1:], cnf_map[:, 1:],
                        conf_disable=self.dpt_conf_grad_loss_conf_disable,
                        grad_conf_disable=self.dpt_conf_grad_loss_grad_conf_disable,
                        range=self.dpt_conf_grad_loss_range,
                        gamma=self.dpt_conf_grad_loss_gamma,
                        alpha=self.dpt_conf_grad_loss_alpha,
                        scales=self.dpt_conf_grad_loss_scales
                    )
                else:
                    conf_loss_o = torch.zeros_like(conf_loss_f)
                    grad_loss_o = torch.zeros_like(grad_loss_f)

                # Perform the confidence-weighted loss reduction
                if self.dpt_conf_grad_loss_separate:
                    conf_loss = 0.0
                    conf_loss += conf_loss_f.mean() if conf_loss_f.numel() > 0 else 0.0
                    conf_loss += conf_loss_o.mean() if conf_loss_o.numel() > 0 else 0.0
                else:
                    conf_loss = torch.cat([conf_loss_f, conf_loss_o], dim=1)
                    conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0.0
                # Add the gradient loss since they are already averaged
                grad_loss = grad_loss_f + grad_loss_o
            else:
                # Confidence-weighted depth and gradient loss for the first frame
                conf_loss_all, grad_loss_all = conf_grad_loss(
                    dpt_map, dpt, msk, cnf_map,
                    conf_disable=self.dpt_conf_grad_loss_conf_disable,
                    grad_conf_disable=self.dpt_conf_grad_loss_grad_conf_disable,
                    range=self.dpt_conf_grad_loss_range,
                    gamma=self.dpt_conf_grad_loss_gamma,
                    alpha=self.dpt_conf_grad_loss_alpha,
                    scales=self.dpt_conf_grad_loss_scales
                )

                conf_loss = 0.0
                conf_loss += conf_loss_all.mean() if conf_loss_all.numel() > 0 else 0.0
                grad_loss = grad_loss_all.mean() if grad_loss_all.numel() > 0 else 0.0

            scalar_stats.dpt_conf_loss = conf_loss
            scalar_stats.dpt_grad_loss = grad_loss
            loss += (
                self.dpt_conf_grad_loss_weight * conf_loss +
                self.dpt_conf_grad_loss_weight * grad_loss
            )

        return loss
