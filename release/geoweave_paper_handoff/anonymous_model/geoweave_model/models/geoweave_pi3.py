import torch
from torch import nn

from ..configs import GeoWeaveConfig
from ..heads import GeometryHeads
from ..layers import GeoWeaveBlock, PatchEmbed


class GeoWeavePi3(nn.Module):
    """Pi3-style GeoWeave model with selected global decoder dependencies."""

    def __init__(self, config: GeoWeaveConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config.patch_size, config.embed_dim)
        self.patch_start_idx = config.num_register_tokens
        self.register_tokens = nn.Parameter(torch.randn(1, 1, self.patch_start_idx, config.embed_dim) * 1e-6)

        self.decoder = nn.ModuleList()
        global_layer = 0
        for layer_idx in range(config.depth):
            is_global = layer_idx % 2 == 1
            use_selection = is_global and global_layer in set(config.global_layers)
            self.decoder.append(
                GeoWeaveBlock(
                    dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dependency_heads=config.dependency_heads,
                    dependency_head_dim=config.dependency_head_dim,
                    topk=config.topk,
                    use_dependency_selection=use_selection,
                )
            )
            if is_global:
                global_layer += 1

        self.heads = GeometryHeads(config.embed_dim, self.patch_start_idx)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor):
        if images.dim() != 5:
            raise ValueError("images must have shape [batch, views, 3, height, width]")
        images = (images - self.image_mean) / self.image_std
        bsz, views, _, height, width = images.shape
        patch_tokens, grid_hw = self.patch_embed(images.reshape(bsz * views, 3, height, width))
        register_tokens = self.register_tokens.expand(bsz, views, -1, -1).reshape(bsz * views, self.patch_start_idx, -1)
        hidden = torch.cat([register_tokens, patch_tokens], dim=1)

        for layer_idx, block in enumerate(self.decoder):
            if layer_idx % 2 == 0:
                hidden = block(hidden)
            else:
                token_count = hidden.shape[1]
                hidden = hidden.reshape(bsz, views * token_count, -1)
                hidden = block(hidden)
                hidden = hidden.reshape(bsz * views, token_count, -1)

        tokens = hidden.reshape(bsz, views, hidden.shape[1], hidden.shape[2])
        return self.heads(tokens, grid_hw=grid_hw, image_hw=(height, width))


def build_geoweave_pi3(tiny: bool = False, **overrides) -> GeoWeavePi3:
    config = GeoWeaveConfig.tiny() if tiny else GeoWeaveConfig()
    for key, value in overrides.items():
        setattr(config, key, value)
    return GeoWeavePi3(config)
