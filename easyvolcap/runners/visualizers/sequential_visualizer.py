# The user might specify multiple visualizers
# Call the visualize function, aggregate stats
# Default visualization module (called visualizer)
import copy
import torch
import numpy as np
from torch import nn
from typing import Union, List

from easyvolcap.engine import VISUALIZERS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.runners.visualizers.geometry_visualizer import GeometryVisualizer
from easyvolcap.runners.visualizers.volumetric_video_visualizer import VolumetricVideoVisualizer


@VISUALIZERS.register_module()
class SequentialVisualizer(VolumetricVideoVisualizer):
    def __init__(self,
                 visualizer_cfgs: List[dotdict] = [
                     dotdict(type=GeometryVisualizer.__name__),
                     dotdict(type=VolumetricVideoVisualizer.__name__),
                 ],
                 **kwargs,
                 ):
        kwargs = dotdict(kwargs)  # for recursive update
        call_from_cfg(super().__init__, kwargs)
        visualizer_cfgs = [copy.deepcopy(kwargs).update(visualizer_cfg) for visualizer_cfg in visualizer_cfgs]  # for recursive update
        self.visualizers: List[VolumetricVideoVisualizer] = [
            VISUALIZERS.build(visualizer_cfg) for visualizer_cfg in visualizer_cfgs
        ]

    def visualize(self, output: dotdict, batch: dotdict):
        stats = dotdict()
        for visualizer in self.visualizers:
            stats.update(visualizer.visualize(output, batch))
        return stats
    
    def summarize(self):
        stats = dotdict()
        for visualizer in self.visualizers:
            stats.update(visualizer.summarize())
        return stats

    def synchronize(self):
        for visualizer in self.visualizers:
            if hasattr(visualizer, "synchronize"):
                visualizer.synchronize()
