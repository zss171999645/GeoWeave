# Default pipeline for volumetric videos
# This corresponds to the tranditional implementation's renderer
from __future__ import annotations
from typing import TYPE_CHECKING

from easyvolcap.models.noop_model import NoopModel
if TYPE_CHECKING:
    from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner

import time
import torch
from torch import nn
from typing import Union
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.timer_utils import timer  # global timer
from easyvolcap.utils.base_utils import dotdict

from easyvolcap.engine import cfg, args
from easyvolcap.engine import MODELS, CAMERAS, SAMPLERS, NETWORKS, RENDERERS, SUPERVISORS, REGRESSORS, EMBEDDERS
from easyvolcap.models.samplers.vggt_sampler import VGGTSampler
from easyvolcap.models.supervisors.sequential_supervisor import SequentialSupervisor


@MODELS.register_module()
class DepthModel(nn.Module):
    def __init__(self,
                 sampler_cfg: dotdict = dotdict(type=VGGTSampler.__name__),
                 supervisor_cfg: dotdict = dotdict(type=SequentialSupervisor.__name__),
                 dtype: Union[str, torch.dtype] = torch.float,
                 ):
        super().__init__()

        noop_network = NoopModel()  # for compatibility
        self.sampler: VGGTSampler = SAMPLERS.build(sampler_cfg, network=noop_network)
        self.supervisor: SequentialSupervisor = SUPERVISORS.build(supervisor_cfg, network=noop_network)

        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    def prepare_params(self, runner: 'VolumetricVideoRunner', batch: dotdict):
        for name, module in self.named_children():
            if hasattr(module, 'prepare_params'): module.prepare_params(runner, batch)

    def prepare_data(self, batch: dotdict):
        batch.K = batch.K.type(self.dtype)
        batch.R = batch.R.type(self.dtype)
        batch.T = batch.T.type(self.dtype)

        # Always forward IBR required camera parameters
        if 'src_exts' in batch:
            batch.src_exts = batch.src_exts.type(self.dtype)
            batch.src_ixts = batch.src_ixts.type(self.dtype)

        # Maybe forward input ray directions
        if 'ray_o' in batch:
            batch.ray_o = batch.ray_o.type(self.dtype)
            batch.ray_d = batch.ray_d.type(self.dtype)

        if 't' in batch:
            batch.t = batch.t.type(self.dtype)

        if 'n' in batch:
            batch.n = batch.n.type(self.dtype)
            batch.f = batch.f.type(self.dtype)

        if 'near' in batch:
            batch.near = batch.near.type(self.dtype)
            batch.far = batch.far.type(self.dtype)

        if 'bounds' in batch:
            batch.bounds = batch.bounds.type(self.dtype)

        if 'xyz' in batch:
            batch.xyz = batch.xyz.type(self.dtype)
            if 'dir' in batch: batch.dir = batch.dir.type(self.dtype)
            if 'dist' in batch: batch.dist = batch.dist.type(self.dtype)

        return batch

    def forward(self, batch: dotdict, compute_loss: bool = False):
        # prepare data
        self.prepare_data(batch)

        # forward
        timer.record()
        self.sampler(batch=batch)
        output = batch.output
        output.time = timer.record('model')

        # Loss computing part of the network
        if self.training or compute_loss:
            loss, scalar_stats, image_stats = self.supervisor.supervise(output, batch)
            if "chunkwise_bp_loss" in batch:
                loss += batch.chunkwise_bp_loss
                output.scalar_stats.chunkwise_bp_loss = batch.chunkwise_bp_loss.detach()
            output.loss = loss
            output.scalar_stats = scalar_stats
            output.image_stats = image_stats

        return output
