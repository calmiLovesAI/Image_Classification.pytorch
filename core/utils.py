from typing import List

import cv2
import numpy as np
import torch


def read_image(image_paths:List[str], add_dim=False, convert_to_tensor=False):
    n = len(image_paths)
    if n == 1:
        # 只有一张图片
        image_array = cv2.imread(image_paths[0], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        if add_dim:
            image_array = np.expand_dims(image_array, axis=0)
    else:
        # 有多张图片
        image_list = list()
        for i in range(n):
            image = cv2.imread(image_paths[i], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_list.append(image)
        image_array = np.stack(image_list, axis=0)
    if convert_to_tensor:
        image_array /= 255.0
        image_tensor = torch.from_numpy(image_array)   # (N, H, W, C)
        image_tensor = torch.permute(image_tensor, dims=(0, 2, 3, 1))  # (N, C, H, W)
        return image_tensor
    return image_array