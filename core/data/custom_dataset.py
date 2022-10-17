from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(ImageDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform

        self.images_root = root
        self.images = list()
        # 获取所有的图片路径和它们所属的类别
        for img_dir in Path(self.images_root).rglob("*.*"):
            image_dir = str(img_dir)
            image_class = img_dir.parts[-2]
            self.images.append([image_dir, image_class])

        # 'self.images_root'路径下的文件夹名就是类别名
        self.classes = [f.name for f in Path(self.images_root).iterdir()]
        self.class2id = dict((c, i) for (i, c) in enumerate(self.classes))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_path, label = self.images[item]
        label = torch.tensor(self.class2id[label], dtype=torch.float32)
        # 读取图片
        image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def get_classes(self):
        return self.classes, len(self.classes)
