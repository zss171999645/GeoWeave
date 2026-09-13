# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import torch
import numpy as np
from PIL import Image
from copy import deepcopy
from torchvision import transforms as TF
from multiprocessing import Pool

from easyvolcap.utils.image_utils import crop_nhwc_image

from concurrent.futures import ThreadPoolExecutor

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s", force=True)


def read_image(args):
    path, is_seg = args
    if not is_seg:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(path, -1)
    return img


def compute_aspect_ratio_from_first_image(image_path, max_size=518, align_size=14, is_seg=False, proc_align_size=1):
    img = read_image([image_path, is_seg])
    oH, oW = img.shape[:2]
    ratio = max_size / max(oH, oW)
    H, W = int(oH * ratio), int(oW * ratio)
    H, W = round(H / align_size) * align_size, round(W / align_size) * align_size
    return H / W


def load_and_preprocess_images_K_parallel(image_path_list, K_list, num_workers=8, **kwargs):
    # ------------------------
    # 计算第一张图的aspect_ratio
    # ------------------------
    fixed_aspect_ratio = compute_aspect_ratio_from_first_image(image_path_list[0], **kwargs)

    # ------------------------
    # 线程池处理
    # ------------------------
    def thread_pool_wrapper(func):
        def wrapped(list_data, Ks, **kwargs):
            list_data_prepped = [[[x], [Ks[i]]] for i, x in enumerate(list_data)]
            arg_func = lambda x: func(*x, fixed_aspect_ratio=fixed_aspect_ratio, **kwargs)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(arg_func, list_data_prepped))

            imgs = []
            Ks_out = []
            for result in results:
                imgs.append(result[0])
                Ks_out.append(result[1])
            imgs = torch.cat(imgs, dim=0)
            Ks_out = torch.cat(Ks_out, dim=0)
            return imgs, Ks_out
        return wrapped
    return thread_pool_wrapper(load_and_preprocess_images_K)(image_path_list, K_list, **kwargs)



def load_and_preprocess_images_K(image_path_list, K_list, is_seg=False, fixed_aspect_ratio=None,
                                 safe_bound=4, max_size=518, align_size=14,
                                 proc_max_size=-1, proc_align_size=1):
    to_tensor = TF.ToTensor()
    images = []
    Ks = []
    shapes = set()

    for i, path in enumerate(image_path_list):
        img = read_image([path, is_seg])
        K = deepcopy(K_list[i])
        oW, oH = img.shape[1], img.shape[0]

        # resize 1
        ratio = max_size / max(oH, oW)
        H, W = int(oH * ratio), int(oW * ratio)
        H, W = round(H / align_size) * align_size, round(W / align_size) * align_size
        rH, rW = H / oH, W / oW
        K = K.copy()
        K[0:1] = K[0:1] * rW  # K[0, 0] *= rW
        K[1:2] = K[1:2] * rH  # K[1, 1] *= rH
        if not is_seg:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)

        # resize 2
        oW, oH = img.shape[1], img.shape[0]
        aspect_ratio = fixed_aspect_ratio if fixed_aspect_ratio is not None else oH / oW
        if proc_max_size > 0: h, w = int(aspect_ratio * proc_max_size), proc_max_size
        else: h, w = int(aspect_ratio * max(oH,oW)), max(oH, oW) # 294 518
        
        ht, wt = h // proc_align_size * proc_align_size, w // proc_align_size * proc_align_size
        ratio2 = max((ht + safe_bound) / oH, (wt + safe_bound) / oW)
        new_height, new_width = int(oH * ratio2), int(oW * ratio2)

        # Resize with new dimensions (width, height)
        if not is_seg:
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            img = np.array(img / 255.0, dtype=np.float32)
        else:
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            img = np.array(img / 100.0, dtype=np.float32)
            
        img = to_tensor(img)  # Convert to tensor (0, 1)
        img = img.permute(1, 2, 0)  # Convert to (H, W, C)
        resize_K = K.copy()
        resize_K[0:1] = resize_K[0:1] * (new_width / oW)
        resize_K[1:2] = resize_K[1:2] * (new_height / oH)
        # Center crop height if it's larger than 518
        img, crop_h, crop_w = crop_nhwc_image(img, size=(ht, wt), center=False, strict_center=True, K=resize_K, return_offset=True, is_seg=is_seg)
        resize_K[0, 2] -= crop_w
        resize_K[1, 2] -= crop_h
        img = img.permute(2, 0, 1)  # Convert to (C, H, W)
        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)
        Ks.append(resize_K)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        padded_Ks = []
        for i in range(len(images)):
            img = images[i]
            pad_K = Ks[i].copy()
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left 
                if is_seg:
                    pad_value = 0.38
                else:
                    pad_value = 0.0       
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=pad_value
                )
                pad_K[0, 2] += pad_left
                pad_K[1, 2] += pad_top
            padded_images.append(img)
            padded_Ks.append(pad_K)
        images = padded_images
        Ks = padded_Ks

    images = torch.stack(images)  # concatenate images
    Ks = torch.from_numpy(np.stack(Ks))
    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)
    if is_seg:
        images *= 100
    return images, Ks
