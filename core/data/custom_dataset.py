from pathlib import Path

import torch
from torch.utils.data import Dataset

from core.utils import read_image


class ImageDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(ImageDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform

        self.images_root = root
        self.images = list()

        # 'self.images_root'路径下的文件夹名就是类别名
        self.classes = []
        if train:
            self.images, _ = self._split_train_test()
        else:
            _, self.images = self._split_train_test()
        self.class2id = dict((c, i) for (i, c) in enumerate(self.classes))

    def _split_train_test(self):
        """
        划分训练集和测试集
        :return:  Type: [List, List]
        """
        train_ratio = 0.8
        train_set = []
        test_set = []
        for folder in Path(self.images_root).iterdir():
            # 单独某一类别的数据
            class_set = []
            self.classes.append(folder.name)
            class_dir = Path(self.images_root).joinpath(folder)
            for img_dir in class_dir.glob("*.*"):
                class_set.append([str(img_dir), folder.name])
            # 属于个类别的图片数量
            num_images = len(class_set)
            train_set.append(*class_set[:int(train_ratio * num_images)])
            test_set.append(*class_set[int(train_ratio * num_images):])
        return train_set, test_set

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
