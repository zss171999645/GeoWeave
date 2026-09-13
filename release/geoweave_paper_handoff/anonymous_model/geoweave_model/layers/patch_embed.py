import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: torch.Tensor):
        patches = self.proj(images)
        grid_hw = patches.shape[-2:]
        tokens = patches.flatten(2).transpose(1, 2)
        return tokens, grid_hw
