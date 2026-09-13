# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np


def load_and_preprocess_images_square(image_path_list, target_size=1024):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 518.

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 5) containing [x1, y1, x2, y2, width, height] for each image
        )

    Raises:
        ValueError: If the input list is empty
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []  # Renamed from position_info to be more descriptive
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        # Convert to RGB
        img = img.convert("RGB")

        # Get original dimensions
        width, height = img.size

        # Make the image square by padding the shorter dimension
        max_dim = max(width, height)

        # Calculate padding
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Calculate scale factor for resizing
        scale = target_size / max_dim

        # Calculate final coordinates of original image in target space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        # Store original image coordinates and scale
        original_coords.append(np.array([x1, y1, x2, y2, width, height]))

        # Create a new black square image and paste original
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Resize to target size
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        # Convert to tensor
        img_tensor = to_tensor(square_img)
        images.append(img_tensor)

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)

    return images, original_coords


def load_and_preprocess_images(image_path_list, mode="crop", target_size=518):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to target_size and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension target_size
                               and padding the smaller dimension to reach a square shape.
        target_size (int, optional): Target size used by crop/pad preprocessing. Defaults to 518.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=target_size while maintaining aspect ratio
          and height is center-cropped if larger than target_size
        - When mode="pad": The function ensures the largest dimension is target_size while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (target_size x target_size)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = int(target_size)
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension target_size while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to target_size
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than target_size (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


def load_and_preprocess_images_evc_compatible(image_path_list, mode="crop", target_size=518):
    """
    Load and preprocess images using the same resize/crop/pad path as the old EVC
    RE10K/CO3Dv2 VGGT official-eval dataset branch.

    Notes:
        - Uses the legacy image loader + OpenCV cubic resize on NHWC tensors.
        - Returns NCHW tensors because the lightweight evaluator feeds VGGT
          directly instead of going through the EVC dataset wrapper.
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    import cv2

    from easyvolcap.utils.data_utils import as_torch_func, load_image_file
    from easyvolcap.utils.image_utils import fill_nhwc_image

    target_size = int(target_size)
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")

    div = 14
    images = [torch.as_tensor(load_image_file(path)) for path in image_path_list]
    heights = [int(im.shape[0]) for im in images]
    widths = [int(im.shape[1]) for im in images]

    def _resize_nhwc(im: torch.Tensor, new_h: int, new_w: int):
        return as_torch_func(
            lambda arr: cv2.resize(arr, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)
        )(im)

    for idx, image in enumerate(images):
        h0, w0 = heights[idx], widths[idx]
        if h0 <= 0 or w0 <= 0:
            raise ValueError(f"Invalid image size: H={h0}, W={w0}, path={image_path_list[idx]}")

        if mode == "pad":
            if w0 >= h0:
                new_w = target_size
                new_h = round((h0 * (new_w / w0)) / div) * div
            else:
                new_h = target_size
                new_w = round((w0 * (new_h / h0)) / div) * div
        else:
            new_w = target_size
            new_h = round((h0 * (new_w / w0)) / div) * div

        new_h = max(div, int(new_h))
        new_w = max(div, int(new_w))

        if new_h != h0 or new_w != w0:
            image = _resize_nhwc(image, new_h, new_w)

        if mode == "crop" and new_h > target_size:
            start_y = (new_h - target_size) // 2
            image = image[start_y:start_y + target_size, :, :]
            new_h = target_size

        if mode == "pad" and (new_h != target_size or new_w != target_size):
            image = fill_nhwc_image(image, size=(target_size, target_size), value=1.0, center=True)
            new_h, new_w = target_size, target_size

        images[idx] = image
        heights[idx], widths[idx] = int(new_h), int(new_w)

    max_h, max_w = max(heights), max(widths)
    if len(set(heights)) > 1 or len(set(widths)) > 1:
        for idx, image in enumerate(images):
            if heights[idx] == max_h and widths[idx] == max_w:
                continue
            images[idx] = fill_nhwc_image(image, size=(max_h, max_w), value=1.0, center=True)

    return torch.stack([image.permute(2, 0, 1).contiguous() for image in images], dim=0)
