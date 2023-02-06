import math
import os.path
import time
from typing import List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# def auto_padding(k, s, d=1):
#     """
#
#     Args:
#         k: kernel size
#         s: stride
#         d: dilation
#
#     Returns: padding value
#
#     """
#     return math.ceil((d*(k-1)+1-s)/2)


def opencv_read_image(image_path):
    image_array = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return image_array  # dtype = np.uint8


def read_image(image_paths: List[str], add_dim=False, convert_to_tensor=False, resize=False, size=None):
    n = len(image_paths)
    if n == 1:
        # 只有一张图片
        image_array = opencv_read_image(image_paths[0])
        if add_dim:
            image_array = np.expand_dims(image_array, axis=0)
    else:
        # 多张图片
        image_list = list()
        for i in range(n):
            image = opencv_read_image(image_paths[i])
            image_list.append(image)
        image_array = np.stack(image_list, axis=0)
    if convert_to_tensor:
        image_array = image_array.astype(np.float32)
        image_array /= 255.0
        image_tensor = torch.from_numpy(image_array)  # (N, H, W, C)
        image_tensor = torch.permute(image_tensor, dims=(0, 3, 1, 2))  # (N, C, H, W)
        if resize and size is not None:
            image_tensor = F.resize(image_tensor, size)
        return image_tensor
    return image_array


def download_file(url, model_dir):
    auto_make_dirs(model_dir)
    if os.path.exists(model_dir):
        print(f"File '{model_dir}' already exists.")
    else:
        print(f"Start downloading from: {url}")
        torch.hub.download_url_to_file(url=url, dst=model_dir)
        # r = requests.get(url, stream=True)
        # total = int(r.headers.get("content-length", 0))
        # with open(model_dir, mode="wb") as f, \
        #         tqdm(desc=model_path, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
        #     for data in r.iter_content(chunk_size=1024):
        #         size = f.write(data)
        #         bar.update(size)
        print("Download completed!")


def load_state_dict_from_url(url, model_dir, map_location=None):
    download_file(url, model_dir)
    return torch.load(model_dir, map_location=map_location)


def get_current_format_time():
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def get_format_filename(model_name: str, dataset_name: str, addition: str = None) -> str:
    return model_name + "_" + dataset_name + "_" + addition


def auto_make_dirs(file_path):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)