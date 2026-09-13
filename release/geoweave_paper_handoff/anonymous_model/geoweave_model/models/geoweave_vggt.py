import torch
from torch import nn

from ..configs import GeoWeaveConfig
from ..heads import GeometryHeads
from ..layers import GeoWeaveBlock, PatchEmbed


class GeoWeaveVGGT(nn.Module):
    """VGGT-style GeoWeave model with frame blocks followed by global fusion."""

    def __init__(self, config: GeoWeaveConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config.patch_size, config.embed_dim)
        self.camera_tokens = nn.Parameter(torch.randn(1, 2, 1, config.embed_dim) * 1e-6)
        self.register_tokens = nn.Parameter(torch.randn(1, 2, config.num_register_tokens, config.embed_dim) * 1e-6)
        self.patch_start_idx = 1 + config.num_register_tokens

        self.frame_blocks = nn.ModuleList()
        self.global_blocks = nn.ModuleList()
        selected_layers = set(config.global_layers)
        for layer_idx in range(config.depth):
            block_kwargs = dict(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dependency_heads=config.dependency_heads,
                dependency_head_dim=config.dependency_head_dim,
                topk=config.topk,
            )
            self.frame_blocks.append(GeoWeaveBlock(use_dependency_selection=False, **block_kwargs))
            self.global_blocks.append(
                GeoWeaveBlock(use_dependency_selection=layer_idx in selected_layers, **block_kwargs)
            )

        self.heads = GeometryHeads(config.embed_dim, self.patch_start_idx)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1), persistent=False)

    def _special_tokens(self, batch_size: int, views: int):
        first = torch.zeros(views, dtype=torch.long, device=self.camera_tokens.device)
        first[1:] = 1
        camera = self.camera_tokens[:, first].expand(batch_size, views, -1, -1)
        register = self.register_tokens[:, first].expand(batch_size, views, -1, -1)
        return torch.cat([camera, register], dim=2)

    def forward(self, images: torch.Tensor):
        if images.dim() != 5:
            raise ValueError("images must have shape [batch, views, 3, height, width]")
        images = (images - self.image_mean) / self.image_std
        bsz, views, _, height, width = images.shape
        patch_tokens, grid_hw = self.patch_embed(images.reshape(bsz * views, 3, height, width))
        patch_tokens = patch_tokens.reshape(bsz, views, patch_tokens.shape[1], patch_tokens.shape[2])
        tokens = torch.cat([self._special_tokens(bsz, views), patch_tokens], dim=2)

        for frame_block, global_block in zip(self.frame_blocks, self.global_blocks):
            token_count = tokens.shape[2]
            tokens = frame_block(tokens.reshape(bsz * views, token_count, -1)).reshape(bsz, views, token_count, -1)
            tokens = global_block(tokens.reshape(bsz, views * token_count, -1)).reshape(bsz, views, token_count, -1)

        return self.heads(tokens, grid_hw=grid_hw, image_hw=(height, width))


def build_geoweave_vggt(tiny: bool = False, **overrides) -> GeoWeaveVGGT:
    config = GeoWeaveConfig.tiny() if tiny else GeoWeaveConfig()
    for key, value in overrides.items():
        setattr(config, key, value)
    return GeoWeaveVGGT(config)
