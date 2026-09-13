import torch
from torch import nn
from torch.nn import functional as F

from easyvolcap.engine import SUPERVISORS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.metric_utils import camera_accuracy_auc
from easyvolcap.utils.loss_utils import PoseLossType, huber, l1, l2, check_and_fix_inf_nan, l21


def mean_scalar_stat(value, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.mean()

    dtype = like.dtype if like.is_floating_point() else torch.float32
    return torch.as_tensor(value, device=like.device, dtype=dtype).mean()


@SUPERVISORS.register_module()
class PoseSupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module, 
                 pose_loss_type: PoseLossType = PoseLossType.L1.name,
                 pose_loss_huber_delta: float = 0.1,

                 pose_loss_weight: float = 0.0,

                 seq_pose_loss_translation_weight: float = 1.0,
                 seq_pose_loss_rotation_weight: float = 1.0,
                 seq_pose_loss_focal_weight: float = 0.5,
                 seq_pose_loss_gamma: float = 0.6,
                 seq_pose_loss_translation_max: float = 100.,
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)

        # Determine the global pose loss type
        self.pose_loss_type = PoseLossType[pose_loss_type]
        self.pose_loss_huber_delta = pose_loss_huber_delta

        self.pose_loss_weight = pose_loss_weight

        self.seq_pose_loss_translation_weight = seq_pose_loss_translation_weight
        self.seq_pose_loss_rotation_weight = seq_pose_loss_rotation_weight
        self.seq_pose_loss_focal_weight = seq_pose_loss_focal_weight
        self.seq_pose_loss_gamma = seq_pose_loss_gamma
        self.seq_pose_loss_translation_max = seq_pose_loss_translation_max

    def compute_pose_loss(self, cam_map: torch.Tensor, cam: torch.Tensor,
                          type=PoseLossType.HUBER, **kwargs):
        if type == PoseLossType.HUBER:
            return huber(cam_map, cam, **kwargs)
        elif type == PoseLossType.L1:
            return l1(cam_map, cam)
        elif type == PoseLossType.L2:
            return l2(cam_map, cam)
        elif type == PoseLossType.L21:
            return l21(cam_map, cam)

    def compute_loss(self, output: dotdict, batch: dotdict, loss: torch.Tensor, scalar_stats: dotdict, image_stats: dotdict):
        if 'cam_map' in output and 'cam' in batch and self.pose_loss_weight > 0:
            pose_loss = self.compute_pose_loss(
                output.cam_map, batch.cam,
                type=self.pose_loss_type,
                delta=self.pose_loss_huber_delta
            )
            scalar_stats.pose_loss = pose_loss
            loss += self.pose_loss_weight * pose_loss

        if 'cam_maps' in output and 'cam' in batch and (self.seq_pose_loss_translation_weight > 0 or self.seq_pose_loss_rotation_weight > 0 or self.seq_pose_loss_focal_weight > 0):
            # Get the valid frame mask
            msk = batch.msk[:, 0].clone().sum(dim=(-1, -2)) > 100

            # Compute the sequence pose loss
            seq_pose_loss = [0, 0, 0]
            iters = len(output.cam_maps)
            for i in range(iters):
                w = self.seq_pose_loss_gamma ** (iters - i - 1)
                cam_map = output.cam_maps[i].clone()  # current prediction
                cam = batch.cam.clone()

                # Compute the sequence pose loss
                if msk.sum() > 0:
                    # Translation loss
                    tloss = self.compute_pose_loss(
                        cam_map[msk][..., :3],
                        cam[msk][..., :3],
                        type=self.pose_loss_type, delta=self.pose_loss_huber_delta
                    )
                    # Rotation loss
                    rloss = self.compute_pose_loss(
                        cam_map[msk][..., 3:7],
                        cam[msk][..., 3:7],
                        type=self.pose_loss_type, delta=self.pose_loss_huber_delta
                    )
                    # Focal loss
                    floss = self.compute_pose_loss(
                        cam_map[msk][..., 7:],
                        cam[msk][..., 7:],
                        type=self.pose_loss_type, delta=self.pose_loss_huber_delta
                    )
                else:
                    # Fill with zero if the frame is invalid
                    tloss = (cam_map * 0).mean()
                    rloss = (cam_map * 0).mean()
                    floss = (cam_map * 0).mean()

                # Check and fix NaN and Inf values in the losses
                tloss = check_and_fix_inf_nan(tloss, f'translation_loss_{i}')
                rloss = check_and_fix_inf_nan(rloss, f'rotation_loss_{i}')
                floss = check_and_fix_inf_nan(floss, f'focal_loss_{i}')

                # Add the losses to the sequence pose loss
                seq_pose_loss[0] += tloss * w
                seq_pose_loss[1] += rloss * w
                seq_pose_loss[2] += floss * w

            # Record the sequence pose loss separately
            scalar_stats.translation_loss = seq_pose_loss[0]
            scalar_stats.rotation_loss = seq_pose_loss[1]
            scalar_stats.focal_loss = seq_pose_loss[2]

            # Add and average the sequence pose loss
            seq_pose_total_loss = 0
            seq_pose_total_loss += self.seq_pose_loss_translation_weight * seq_pose_loss[0]
            seq_pose_total_loss += self.seq_pose_loss_rotation_weight * seq_pose_loss[1]
            seq_pose_total_loss += self.seq_pose_loss_focal_weight * seq_pose_loss[2]
            seq_pose_total_loss = seq_pose_total_loss / iters
            loss += seq_pose_total_loss

            # Add training metrics
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                # Compute the accuracy
                m = camera_accuracy_auc(
                    batch.cam.clone(), output.cam_map.clone(), batch,
                    acc_thresh=[15],
                    auc_thresh=[3, 5, 10, 30],
                )
                for k, v in m.items():
                    scalar_stats[k] = mean_scalar_stat(v, seq_pose_total_loss)

        return loss
