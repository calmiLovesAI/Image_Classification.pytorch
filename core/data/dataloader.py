from torch.utils.data import DataLoader

import core.data.transforms as T
from core.data.custom_dataset import ImageDataset


class BaseLoader:
    """
    适用于自定义数据集
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_size = cfg["Train"]["batch_size"]
        self.input_size = cfg["Train"]["input_size"][1:]
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(size=self.input_size),
        ])

    def __call__(self, *args, **kwargs):
        dataset = ImageDataset(self.cfg["Custom"]["root"], transform=self.transforms, target_transform=None)
        classes, num_classes = dataset.get_classes()
        return classes, num_classes, DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


class Cifar10Loader(BaseLoader):
    def __call__(self, *args, **kwargs):
        pass


class Cifar100Loader(BaseLoader):
    def __call__(self, *args, **kwargs):
        pass