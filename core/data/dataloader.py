from torch.utils.data import DataLoader

import core.data.transforms as T
from core.data.custom_dataset import ImageDataset


class TrainLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.input_size = cfg.input_size
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(size=self.input_size),
        ])

    def __call__(self, *args, **kwargs):
        dataset = ImageDataset(self.cfg, transform=self.transforms, target_transform=None)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


class Cifar10Loader(TrainLoader):
    def __call__(self, *args, **kwargs):
        pass


class Cifar100Loader(TrainLoader):
    def __call__(self, *args, **kwargs):
        pass