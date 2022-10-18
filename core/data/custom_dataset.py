from pathlib import Path

import torch
from torch.utils.data import Dataset

from core.utils import read_image


class ImageDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(ImageDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform

        self.images_root = root
        self.images = list()
        # 获取所有的图片路径和它们所属的类别
        for img_dir in Path(self.images_root).rglob("*.*"):
            image_class = img_dir.parts[-2]
            self.images.append([str(img_dir), image_class])

        # 'self.images_root'路径下的文件夹名就是类别名
        self.classes = [f.name for f in Path(self.images_root).iterdir()]
        self.class2id = dict((c, i) for (i, c) in enumerate(self.classes))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_path, label = self.images[item]
        label = torch.tensor(self.class2id[label], dtype=torch.int64)
        # 读取图片
        image = read_image(image_paths=[image_path])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def get_classes(self):
        return self.classes, len(self.classes)

    def __repr__(self):
        return "自定义数据集"
