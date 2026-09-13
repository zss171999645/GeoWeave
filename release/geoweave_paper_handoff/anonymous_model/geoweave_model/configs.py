from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GeoWeaveConfig:
    image_size: int = 224
    patch_size: int = 14
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4
    dependency_heads: int = 4
    dependency_head_dim: int = 64
    topk: int = 1536
    global_layers: Tuple[int, ...] = field(default_factory=lambda: tuple(range(9, 20)))
    output_layers: Tuple[int, ...] = field(default_factory=lambda: (4, 11, 17, 23))

    @classmethod
    def tiny(cls):
        return cls(
            image_size=32,
            patch_size=8,
            embed_dim=64,
            depth=4,
            num_heads=4,
            mlp_ratio=2.0,
            num_register_tokens=2,
            dependency_heads=2,
            dependency_head_dim=16,
            topk=8,
            global_layers=(1, 2),
            output_layers=(1, 3),
        )
