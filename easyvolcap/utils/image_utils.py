import torch
from torch.nn import functional as F
from typing import List


def get_xywh_from_mask(msk):
    import torchvision
    X, Y, H, W = 0, 0, *msk.shape[-3:-1]  # EMOTIONAL DAMAGE
    bbox = torchvision.ops.boxes.masks_to_boxes(msk.view(-1, H, W))
    x = bbox[..., 0].min().round().int().item()  # round, smallest of all images
    y = bbox[..., 1].min().round().int().item()  # round, smallest of all images
    w = (bbox[..., 2] - bbox[..., 0]).max().round().int().item()  # round, biggest of all
    h = (bbox[..., 3] - bbox[..., 1]).max().round().int().item()  # round, biggest of all
    return x, y, w, h


def crop_using_mask(msk: torch.Tensor,
                    K: torch.Tensor,
                    *list_of_imgs: List[torch.Tensor]):
    # Deal with empty batch dimension
    bs = msk.shape[:-3]
    msk = msk.view(-1, *msk.shape[-3:])
    K = K.view(-1, *K.shape[-2:])
    list_of_imgs = [im.view(-1, *im.shape[-3:]) for im in list_of_imgs]

    # Assumes channel last format
    # Assumes batch dimension for msk
    # Will crop all images using msk
    # !: EVIL LIST COMPREHENSION
    xs, ys, ws, hs = zip(*[get_xywh_from_mask(m) for m in msk])  # all sizes
    K, *list_of_imgs = zip(*[crop_using_xywh(x, y, w, h, k, *im)
                             for x, y, w, h, k, *im
                             in zip(xs, ys, ws, hs, K, *list_of_imgs)
                             ])  # HACK: This is doable... # outermost: source -> outermost: batch
    K = torch.stack(K)  # stack source dim
    # Resize instead of filling things up?
    # Filling things up might be easier for masked output (bkgd should be black)
    H_max = max(hs)
    W_max = max(ws)
    list_of_imgs = [torch.stack([fill_nhwc_image(im, size=(H_max, W_max)) for im in img]) for img in list_of_imgs]  # HACK: evil list comprehension

    # Restore original dimensionality
    msk = msk.view(*bs, *msk.shape[-3:])
    K = K.view(*bs, *K.shape[-2:])
    list_of_imgs = [im.view(*bs, *im.shape[-3:]) for im in list_of_imgs]
    return K, *list_of_imgs


def crop_using_xywh(x, y, w, h, K, *list_of_imgs):
    K = K.clone()
    K[..., :2, -1] -= torch.as_tensor([x, y], device=K.device)  # crop K
    list_of_imgs = [img[..., y:y + h, x:x + w, :]
                    if isinstance(img, torch.Tensor) else
                    [im[..., y:y + h, x:x + w, :] for im in img]  # HACK: evil list comprehension
                    for img in list_of_imgs]
    return K, *list_of_imgs


def pad_image_to_divisor(img: torch.Tensor, div: List[int], mode='constant', value=0.0):
    H, W = img.shape[-2:]
    H = (H + div[0] - 1) // div[0] * div[0]
    W = (W + div[0] - 1) // div[0] * div[0]
    return pad_image(img, (H, W), mode, value)


def pad_image(img: torch.Tensor, size: List[int], mode='constant', value=0.0):
    bs = img.shape[:-3]  # batch size
    img = img.reshape(-1, *img.shape[-3:])
    H, W = img.shape[-2:]  # H, W
    Ht, Wt = size
    pad = (0, Wt - W, 0, Ht - H)

    if Wt >= W and Ht >= H:
        img = F.pad(img, pad, mode, value)
    else:
        if Wt < W and Ht >= H:
            img = F.pad(img, (0, 0, 0, Ht - H), mode, value)
        if Wt >= W and Ht < H:
            img = F.pad(img, (0, Wt - W, 0, 0), mode, value)
        img = img[..., :Ht, :Wt]

    img = img.reshape(*bs, *img.shape[-3:])
    return img


def fill_nchw_image(img: torch.Tensor, size: List[int], value: float = 0.0, center: bool = False):
    bs = img.shape[:-3]  # -3, -2, -1
    cs = img.shape[-3:-2]
    zeros = img.new_full((*bs, *cs, *size), value)
    target_h, target_w = size
    source_h, source_w = img.shape[-2], img.shape[-1]
    h = min(target_h, source_h)
    w = min(target_w, source_w)
    start_h = (target_h - h) // 2 if center else 0
    start_w = (target_w - w) // 2 if center else 0
    zeros[..., start_h:start_h + h, start_w:start_w + w] = img[..., :h, :w]
    return zeros


def fill_nhwc_image(img: torch.Tensor, size: List[int], value: float = 0.0, center: bool = False):
    bs = img.shape[:-3]  # -3, -2, -1
    cs = img.shape[-1:]
    zeros = img.new_full((*bs, *size, *cs), value)
    target_h, target_w = size
    source_h, source_w = img.shape[-3], img.shape[-2]
    h = min(target_h, source_h)
    w = min(target_w, source_w)
    start_h = (target_h - h) // 2 if center else 0
    start_w = (target_w - w) // 2 if center else 0
    zeros[..., start_h:start_h + h, start_w:start_w + w, :] = img[..., :h, :w, :]
    return zeros


def crop_nhwc_image(img: torch.Tensor, size: List[int], center: bool = True, strict_center: bool = False, K: torch.Tensor = None, return_offset: bool = False, is_seg: bool = False):
    assert not (center and strict_center), "center and strict_center cannot be True at the same time"
    if strict_center:
        assert K is not None, "K must be provided when strict_center is True"

    bs = img.shape[:-3]  # -3, -2, -1
    img = img.view(-1, *img.shape[-3:])  # -1, H, W, C
    H, W = img.shape[-3:-1]
    h, w = size
    if strict_center:
        start_h = int(round(K[1, 2].item())) - h // 2
        start_w = int(round(K[0, 2].item())) - w // 2
    elif center:
        start_h = (H - h) // 2
        start_w = (W - w) // 2
    else:
        start_h = 0
        start_w = 0
    end_h = start_h + h
    end_w = start_w + w
    offset_h = start_h
    offset_w = start_w

    if strict_center and (start_h < 0 or start_w < 0 or end_h > H or end_w > W):
        pad_top = max(0, -start_h)
        pad_bottom = max(0, end_h - H)
        pad_left = max(0, -start_w)
        pad_right = max(0, end_w - W)
        # img: (B, H, W, C) -> (B, H + pad_top + pad_bottom, W + pad_left + pad_right, C)
        if is_seg:
            pad_value = 0.38
        else:
            pad_value = 0.0
        img = F.pad(img, (0, 0, pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=pad_value)
        start_h += pad_top
        start_w += pad_left
        end_h = start_h + h
        end_w = start_w + w

    img = img[..., start_h:start_h + h, start_w:start_w + w, :]
    img = img.view(*bs, *img.shape[-3:])
    if return_offset: return img, offset_h, offset_w
    else: return img    


def interpolate_image(img: torch.Tensor, mode='bilinear', align_corners=False, *args, **kwargs):
    # Performs F.interpolate as images (always augment to B, C, H, W)
    sh = img.shape
    img = img.view(-1, *sh[-3:])
    img = F.interpolate(img, *args, mode=mode, align_corners=align_corners if mode != 'nearest' else None, **kwargs)
    img = img.view(sh[:-3] + img.shape[-3:])
    return img


def resize_image(img: torch.Tensor, mode='bilinear', align_corners=False, *args, **kwargs):
    sh = img.shape
    if len(sh) == 4:  # assumption
        img = img.permute(0, 3, 1, 2)
    elif len(sh) == 3:  # assumption
        img = img.permute(2, 0, 1)[None]
    img = interpolate_image(img, mode=mode, align_corners=align_corners, *args, **kwargs)  # uH, uW, 3
    if len(sh) == 4:
        img = img.permute(0, 2, 3, 1)
    elif len(sh) == 3:  # assumption
        img = img[0].permute(1, 2, 0)
    return img


def rotate_90_degree(
    image: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    clockwise: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Rotate the input image and the camera parameters 90 degrees
    in the specified direction, namely clockwise or counterclockwise.

    Args:
        image (torch.Tensor), (H, W, C): image to rotate
        K (torch.Tensor), (3, 3): intrinsic matrix
        w2c (torch.Tensor), (3, 4) or (4, 4): camera to world matrix
        clockwise (bool): if True, rotate clockwise, otherwise counterclockwise

    Returns:
        new_image (torch.Tensor), (W, H, C): rotated image
        new_K (torch.Tensor), (3, 3): rotated intrinsic matrix
        new_w2c (torch.Tensor), (3, 4): rotated camera to world matrix
    """
    H, W = image.shape[:2]
    new_image = rotate_image_rot90(image, clockwise)
    new_K = adjust_intrinsic_rot90(K, W, H, clockwise)
    new_w2c = adjust_extrinsic_rot90(w2c, clockwise)
    return new_image, new_K, new_w2c


def rotate_image_rot90(
    image: torch.Tensor,
    clockwise: bool
):
    """ Rotate the given image 90 degrees in the specified direction.
    Namely clockwise or counterclockwise using a transpose and flip.

    Args:
        image (torch.Tensor), (H, W, C): image to rotate
        clockwise (bool): if True, rotate clockwise, otherwise counterclockwise

    Returns:
        (W, H, C) tensor: rotated image
    """
    # Transpose H, W and then flip
    if clockwise:
        # rotate 90° CW = transpose + flip left‐right
        new_image = image.permute(1, 0, 2).flip(1)
    else:
        # rotate 90° CCW = transpose + flip up‐down
        new_image = image.permute(1, 0, 2).flip(0)
    return new_image.clone().contiguous()


def adjust_intrinsic_rot90(
    K: torch.Tensor,
    W: int,
    H: int,
    clockwise: bool
) -> torch.Tensor:
    """
    K: (3,3) tensor
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    M = torch.eye(3, dtype=K.dtype, device=K.device)
    if clockwise:
        M[0, 0], M[1, 1] = fy, fx
        M[0, 2], M[1, 2] = H - cy, cx
    else:
        M[0, 0], M[1, 1] = fy, fx
        M[0, 2], M[1, 2] = cy, W - cx
    return M


def adjust_extrinsic_rot90(
    w2c: torch.Tensor,
    clockwise: bool
) -> torch.Tensor:
    """
    w2c: (3, 4) tensor
    """
    # w2c is the camera to world matrix in OpenCV convention
    R, t = w2c[:3, :3], w2c[:3, 3]

    if clockwise:
        R_rot = torch.tensor([
            [0, -1,  0],
            [1,  0,  0],
            [0,  0,  1]
        ], dtype=R.dtype, device=R.device)
    else:
        R_rot = torch.tensor([
            [0,  1,  0],
            [-1, 0,  0],
            [0,  0,  1]
        ], dtype=R.dtype, device=R.device)

    new_R = R_rot @ R
    new_t = R_rot @ t

    if w2c.shape[0] == 4:
        new_w2c = torch.cat([
            torch.cat([new_R, new_t.unsqueeze(1)], dim=1),
            w2c[3:4, :]
        ], dim=0)
    else:
        new_w2c = torch.cat([new_R, new_t.unsqueeze(1)], dim=1)
    return new_w2c
