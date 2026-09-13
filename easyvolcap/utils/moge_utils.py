# Adapted from:
# https://github.com/microsoft/MoGe/blob/dd158c05461f2353287a182afb2adf0fda46436f/moge/model/moge_model.py
# https://github.com/microsoft/MoGe/blob/dd158c05461f2353287a182afb2adf0fda46436f/moge/utils/geometry_torch.py#L40

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Literal, Tuple, Union, Optional, Callable

from easyvolcap.engine import REGRESSORS
from easyvolcap.utils.base_utils import dotdict


def get_normalized_uv(
        H: int,
        W: int,
        aspect: float = None,
        dtype: torch.dtype = None,
        device: torch.device = None
    ):
    """ Get the normalized UV coordinates for the image plane, with
        the left-top corner as (-W / diagonal, -H / diagonal) and,
        the right-bottom corner as (W / diagonal, H / diagonal)
    """
    # Compute the aspect ratio if not provided
    if aspect is None: aspect = W / H

    # Compute the span of the view plane
    span_x = aspect / (1 + aspect ** 2) ** 0.5
    span_y = 1 / (1 + aspect ** 2) ** 0.5

    # Compute the UV coordinates
    u = torch.linspace(
        -span_x * (W - 1) / W,
        span_x * (W - 1) / W,
        W,
        dtype=dtype,
        device=device
    )
    v = torch.linspace(
        -span_y * (H - 1) / H,
        span_y * (H - 1) / H,
        H,
        dtype=dtype,
        device=device
    )

    # Create the meshgrid
    uv = torch.stack(
        torch.meshgrid(u, v, indexing='xy'),
        dim=-1
    )
    return uv


class ResidualConvBlock(nn.Module):  
    def __init__(
            self,
            in_channels: int,
            out_channels: int = None,
            hidden_channels: int = None,
            padding_mode: str = 'replicate',
            activation: Literal['relu', 'leaky_relu', 'silu', 'elu'] = 'relu',
            norm: Literal['group_norm', 'layer_norm'] = 'group_norm',
        ):
        # Residual Convolutional Block
        super(ResidualConvBlock, self).__init__()  

        # Define the number of output and middle channels
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        # Define the activation function
        if activation =='relu': actvn = lambda: nn.ReLU(inplace=True)
        elif activation == 'leaky_relu': actvn = lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation =='silu': actvn = lambda: nn.SiLU(inplace=True)
        elif activation == 'elu': actvn = lambda: nn.ELU(inplace=True)
        else: raise ValueError(f'Unsupported activation function: {activation}')

        # Define the backbone of the residual block
        self.layers = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            actvn(),
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode
            ),
            nn.GroupNorm(
                hidden_channels // 32 if norm == 'group_norm' else 1,
                hidden_channels
            ),
            actvn(),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode
            )
        )

        # Define the skip connection
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0
            )
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x):  
        skip = self.skip_connection(x)  
        x = self.layers(x)
        x = x + skip
        return x  


@REGRESSORS.register_module()
class MoGeDecoder(nn.Module):
    def __init__(
        self, 
        num_features: int,
        dim_in: int, 
        dim_out: List[int], 
        dim_proj: int = 512,
        dim_upsample: List[int] = [256, 128, 128],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_norm: Literal['group_norm', 'layer_norm'] = 'group_norm',
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1
    ):
        # MoGe Decoder
        super().__init__()

        # Define the projection layers
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=dim_in,
                out_channels=dim_proj,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for _ in range(num_features)
        ])

        # Define the upsampling blocks
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                self._make_upsampler(in_ch + 2, out_ch),
                *(ResidualConvBlock(
                    out_ch,
                    out_ch,
                    dim_times_res_block_hidden * out_ch,
                    activation="relu",
                    norm=res_block_norm
                ) for _ in range(num_res_blocks))
            ) for in_ch, out_ch in zip([dim_proj] + dim_upsample[:-1], dim_upsample)
        ])

        # Define the output blocks
        self.output_block = nn.ModuleList([
            self._make_output_block(
                dim_upsample[-1] + 2,
                dim_out_,
                dim_times_res_block_hidden,
                last_res_blocks,
                last_conv_channels,
                last_conv_size,
                res_block_norm,
            ) for dim_out_ in dim_out
        ])

    def _make_upsampler(
            self,
            in_channels: int,
            out_channels: int
        ):
        # Upsampling block
        upsampler = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode='replicate'
            )
        )
        upsampler[0].weight.data[:] = upsampler[0].weight.data[:, :, :1, :1]
        return upsampler

    def _make_output_block(
            self,
            dim_in: int,
            dim_out: int,
            dim_times_res_block_hidden: int,
            last_res_blocks: int,
            last_conv_channels: int,
            last_conv_size: int,
            res_block_norm: Literal['group_norm', 'layer_norm']
        ):
        return nn.Sequential(
            nn.Conv2d(
                dim_in,
                last_conv_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode='replicate'
            ),
            *(ResidualConvBlock(
                last_conv_channels,
                last_conv_channels,
                dim_times_res_block_hidden * last_conv_channels,
                activation='relu',
                norm=res_block_norm
            ) for _ in range(last_res_blocks)),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                last_conv_channels,
                dim_out,
                kernel_size=last_conv_size,
                stride=1,
                padding=last_conv_size // 2,
                padding_mode='replicate'
            ),
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            batch: dotdict,
        ):
        # Shape things
        Hn, Wn = batch.Hn, batch.Wn
        Hp, Wp = batch.Hp, batch.Wp

        # Process the hidden states
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (Hp, Wp)).contiguous())
                for proj, feat in zip(self.projects, hidden_states)
        ], dim=1).sum(dim=1)  # (B, C, Hp, Wp)

        # Upsample: (Hp, Wp) -> (Hp * 2, Wp * 2) -> (Hp * 4, Wp * 4) -> (Hp * 8, Wp * 8)
        for _, block in enumerate(self.upsample_blocks):
            # UV coordinates is for awareness of image aspect ratio
            uv = get_normalized_uv(
                W=x.shape[-1],
                H=x.shape[-2],
                aspect=Wn / Hn,
                dtype=x.dtype,
                device=x.device
            ).permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, uv], dim=1)  # (B, C + 2, Hp, Wp)

            # Checkpointing
            for layer in block:
                x = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    use_reentrant=False
                )  # (B, C, Hp, Wp) -> (B, C, Hp * 2, Wp * 2)

        # Interpolate to the original image size
        x = F.interpolate(
            x,
            (Hn, Wn),
            mode="bilinear",
            align_corners=False
        )  # (B, C, Hn, Wn)

        # UV coordinates is for awareness of image aspect ratio
        uv = get_normalized_uv(
            W=x.shape[-1],
            H=x.shape[-2],
            aspect=Wn / Hn,
            dtype=x.dtype,
            device=x.device
        ).permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, uv], dim=1)

        # Output blocks
        if isinstance(self.output_block, nn.ModuleList):
            output = [
                torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    use_reentrant=False
                ) for block in self.output_block
            ]
        else:
            output = torch.utils.checkpoint.checkpoint(
                self.output_block,
                x,
                use_reentrant=False
            )

        return output


# Utility functions
# https://github.com/microsoft/MoGe/blob/dd158c05461f2353287a182afb2adf0fda46436f/utils3d/torch/utils.py
def sliding_window_1d(
    x: torch.Tensor,
    window_size: int,
    stride: int = 1,
    dim: int = -1
) -> torch.Tensor:
    """ Sliding window view of the input tensor. The dimension of the
        sliding window is appended to the end of the input tensor's shape.
        NOTE: Since Pytorch has `unfold` function, 1D sliding window view is just a wrapper of it.
    """
    return x.unfold(dim, window_size, stride)


def sliding_window_nd(
    x: torch.Tensor,
    window_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    dim: Tuple[int, ...]
) -> torch.Tensor:
    dim = [dim[i] % x.ndim for i in range(len(dim))]
    assert len(window_size) == len(stride) == len(dim)
    for i in range(len(window_size)):
        x = sliding_window_1d(x, window_size[i], stride[i], dim[i])
    return x


def sliding_window_2d(
    x: torch.Tensor,
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    dim: Union[int, Tuple[int, int]] = (-2, -1)
) -> torch.Tensor:
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    return sliding_window_nd(x, window_size, stride, dim)


def image_uv(
    height: int,
    width: int,
    left: int = None,
    top: int = None,
    right: int = None,
    bottom: int = None,
    device: torch.device = None,
    dtype: torch.dtype = None
) -> torch.Tensor:
    """
    Get image space UV grid, ranging in [0, 1]. 

    >>> image_uv(10, 10):
    [[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
     [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
      ...             ...                  ...
     [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]

    Args:
        width (int): image width
        height (int): image height

    Returns:
        np.ndarray: shape (height, width, 2)
    """
    if left is None: left = 0
    if top is None: top = 0
    if right is None: right = width
    if bottom is None: bottom = height
    u = torch.linspace(
        (left + 0.5) / width,
        (right - 0.5) / width,
        right - left,
        device=device,
        dtype=dtype
    )
    v = torch.linspace(
        (top + 0.5) / height,
        (bottom - 0.5) / height,
        bottom - top,
        device=device,
        dtype=dtype
    )
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def image_pixel_center(
    height: int,
    width: int,
    left: int = None,
    top: int = None,
    right: int = None,
    bottom: int = None,
    dtype: torch.dtype = None,
    device: torch.device = None
) -> torch.Tensor:
    """
    Get image pixel center coordinates, ranging in [0, width] and [0, height].
    `image[i, j]` has pixel center coordinates `(j + 0.5, i + 0.5)`.

    >>> image_pixel_center(10, 10):
    [[[0.5, 0.5], [1.5, 0.5], ..., [9.5, 0.5]],
     [[0.5, 1.5], [1.5, 1.5], ..., [9.5, 1.5]],
      ...             ...                  ...
    [[0.5, 9.5], [1.5, 9.5], ..., [9.5, 9.5]]]

    Args:
        width (int): image width
        height (int): image height

    Returns:
        np.ndarray: shape (height, width, 2)
    """
    if left is None: left = 0
    if top is None: top = 0
    if right is None: right = width
    if bottom is None: bottom = height
    u = torch.linspace(
        left + 0.5,
        right - 0.5,
        right - left,
        dtype=dtype,
        device=device
    )
    v = torch.linspace(
        top + 0.5,
        bottom - 0.5,
        bottom - top,
        dtype=dtype,
        device=device
    )
    u, v = torch.meshgrid(u, v, indexing='xy')
    return torch.stack([u, v], dim=2)


def mask_aware_nearest_resize(
    mask: torch.BoolTensor,
    W: int,
    H: int,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tuple[torch.LongTensor, ...], torch.BoolTensor]:
    """ This function is used to adjust the input 2D mask to the target size by
    nearest neighbor interpolation, taking into account the effective region of the mask.
    The core idea is to find the closest, valid (mask is True) pixel in the original image
    for each target pixel, and record its index (in the original image size).

    Args:
        mask (torch.BoolTensor), (..., Ho, Wo): the input 2D mask
        W (int): target width of the resized map
        H (int): target height of the resized map
        dtype (torch.dtype): data type of the output tensors

    Returns:
        idx (torch.Tensor), (..., H, W): nearest neighbor index of the resized map for each dimension
        msk (torch.Tensor), (..., H, W): mask of the resized map
    """
    # Shape and device things
    device = mask.device
    Ho, Wo = mask.shape[-2:]  # original height and width
    hf, wf = max(1, Ho / H), max(1, Wo / W)  # height and width factors
    hi, wi = math.ceil(hf), math.ceil(wf)  # height and width interpolation factors
    hp, wp = hi // 2 + 1, wi // 2 + 1  # height and width padding size

    # Create the 2d uv map, padded
    uv2d = torch.full(
        (Ho + 2 * hp, Wo + 2 * wp, 2),
        0,
        dtype=dtype,
        device=device
    )  # (Ho + 2 * hp, Wo + 2 * wp, 2)
    uv2d[hp:hp + Ho, wp:wp + Wo] = image_pixel_center(
        width=Wo, height=Ho, dtype=dtype, device=device
    )  # (Hp, Wp, 2)

    # Create the padded mask
    msks = torch.full(
        (*mask.shape[:-2], Ho + 2 * hp, Wo + 2 * wp),
        False,
        dtype=torch.bool,
        device=device
    )  # (..., Ho + 2 * hp, Wo + 2 * wp)
    msks[..., hp:hp + Ho, wp:wp + Wo] = mask  # (..., Hp, Wp)

    # Create the indices, padded
    inds = torch.full(
        (Ho + 2 * hp, Wo + 2 * wp),
        0,
        dtype=torch.long,
        device=device
    )  # (Ho + 2 * hp, Wo + 2 * wp)
    inds[hp:hp + Ho, wp:wp + Wo] = torch.arange(
        Ho * Wo, dtype=torch.long, device=device
    ).reshape(Ho, Wo)  # (Hp, Wp)

    # Window the original uv, mask, and indices
    win_uv2d = sliding_window_2d(uv2d, (hi, wi), 1, dim=(0, 1))  # (Hp, Wp, hi, wi, 2)
    win_msks = sliding_window_2d(msks, (hi, wi), 1, dim=(-2, -1))  # (..., Hp, Wp, hi, wi)
    win_inds = sliding_window_2d(inds, (hi, wi), 1, dim=(0, 1))  # (Hp, Wp, hi, wi)

    # Prepare the target uv and window
    tar_uv2d = image_uv(
        width=W, height=H, dtype=dtype, device=device
    ) * torch.tensor([Wo, Ho], dtype=dtype, device=device)  # (H, W, 2)
    tar_win = torch.round(
        tar_uv2d - torch.tensor(
            (wf / 2, hf / 2), dtype=dtype, device=device
        )  # (H, W, 2)
    ).long() + torch.tensor((wp, hp), dtype=torch.long, device=device)  # (H, W, 2)

    # Gather the target pixels local window
    tar_win_uv2d = win_uv2d[tar_win[..., 1], tar_win[..., 0], :, :, :].reshape(
        H, W, 2, hi * wi
    )  # (H, W, 2, filter_size)
    tar_win_msks = win_msks[..., tar_win[..., 1], tar_win[..., 0], :, :].reshape(
        *mask.shape[:-2], H, W, hi * wi
    )  # (..., H, W, filter_size)
    tar_win_inds = (
        win_inds[tar_win[..., 1], tar_win[..., 0], :, :]
        .reshape(H, W, hi * wi)
        .expand_as(tar_win_msks)
    )  # (..., H, W, filter_size)

    # Compute nearest neighbor in the local window for each pixel
    dist = torch.where(
        tar_win_msks,
        torch.norm(tar_win_uv2d - tar_uv2d[..., None], dim=-2),
        torch.inf
    )  # (..., H, W, filter_size)
    nearest = torch.argmin(dist, dim=-1, keepdim=True)  # (..., H, W, 1)

    # Gather the nearest neighbor index and mask
    idx = torch.gather(tar_win_inds, index=nearest, dim=-1).squeeze(-1)  # (..., H, W)
    msk = torch.any(tar_win_msks, dim=-1)  # (..., H, W)

    # Prepare the output indices
    widx, hidx = idx // Wo, idx % Wo
    bidx = [
        torch.arange(n, device=device).reshape(
            [1] * i + [n] + [1] * (mask.dim() - i - 1)
        )
        for i, n in enumerate(mask.shape[:-2])
    ]

    return (*bidx, widx, hidx), msk


def compute_anchor_sampling_weight(
    xyz: torch.Tensor,
    msk: torch.Tensor,
    r2d: torch.Tensor,
    r3d: torch.Tensor,
    n_anchors: int = 64,
) -> torch.Tensor:
    """ 
    """

    # Shape thigs
    H, W = xyz.shape[-3:-1]
    device = xyz.device

    # Original pixel coordinates
    pu, pv = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )  # (H, W), (H, W)

    # Delta sampling offset
    du = torch.randint(
        -r2d, r2d + 1, (H, W, n_anchors),
        device=device,
    )  # (n_anchors,)
    dv = torch.randint(
        -r2d, r2d + 1, (H, W, n_anchors),
        device=device,
    )  # (n_anchors,)

    # Anchor mask
    au = pu[..., None] + du  # (H, W, n_anchors)
    av = pv[..., None] + dv  # (H, W, n_anchors)
    amsk = (au >= 0) & (au < H) & (av >= 0) & (av < W)  # (H, W, n_anchors)
    au = au.clamp(0, H - 1)  # (H, W, n_anchors)
    av = av.clamp(0, W - 1)  # (H, W, n_anchors)
    amsk = amsk & msk[..., au, av]  # (..., H, W, n_anchors)

    # Compute the distance
    axyz = xyz[..., au, av, :]  # (..., H, W, n_anchors, 3)
    adist = (axyz - xyz[..., None, :]).norm(dim=-1)  # (..., H, W, n_anchors)

    weight = 1 / ((adist <= r3d[..., None]) & amsk).float().sum(
        dim=-1
    ).clamp_min(1)
    weight = torch.where(msk, weight, 0)
    weight = weight / weight.sum(dim=(-2, -1), keepdim=True).add(
        1e-7
    )  # (..., H, W)
    return weight
