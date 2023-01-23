from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

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
        self.name = "Custom Dataset"

    def __call__(self, *args, **kwargs):
        train_data = ImageDataset(self.cfg["Custom"]["root"], train=True, transform=self.transforms,
                                  target_transform=None)
        test_data = ImageDataset(self.cfg["Custom"]["root"], train=False, transform=self.transforms,
                                 target_transform=None)
        classes, num_classes = train_data.get_classes()
        print("正在使用{}, 其中有{}个图像类别，分别为：{}".format(self.cfg["Custom"]["root"], num_classes, classes))
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        return classes, num_classes, train_dataloader, test_dataloader


class Cifar10Loader(BaseLoader):
    def __init__(self, cfg):
        super(Cifar10Loader, self).__init__(cfg)
        self.name = "cifar10"

    def __call__(self, *args, **kwargs):
        cifar10_train = CIFAR10(root=self.cfg["Cifar10"]["root"],
                                train=True,
                                transform=self.transforms,
                                download=True)
        cifar10_test = CIFAR10(root=self.cfg["Cifar10"]["root"],
                               train=False,
                               transform=self.transforms,
                               download=True)
        classes = self.cfg["Cifar10"]["categories"]
        num_classes = len(classes)
        print("正在使用{}\n {}".format(cifar10_train, cifar10_test))
        print("其中有{}个类别，分别是：{}".format(num_classes, classes))
        train_dataloader = DataLoader(cifar10_train, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(cifar10_test, batch_size=self.batch_size, shuffle=False)
        return classes, num_classes, train_dataloader, test_dataloader


class Cifar100Loader(BaseLoader):
    def __init__(self, cfg):
        super(Cifar100Loader, self).__init__(cfg)
        self.name = "cifar100"

    def __call__(self, *args, **kwargs):
        cifar100_train = CIFAR100(root=self.cfg["Cifar100"]["root"],
                                  train=True,
                                  transform=self.transforms,
                                  download=True)
        cifar100_test = CIFAR100(root=self.cfg["Cifar100"]["root"],
                                 train=False,
                                 transform=self.transforms,
                                 download=True)
        classes = self.cfg["Cifar100"]["categories"]
        num_classes = 100
        print("正在使用{}\n {}".format(cifar100_train, cifar100_test))
        print("其中有{}个类别，分别是：{}等".format(num_classes, classes))
        train_dataloader = DataLoader(cifar100_train, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(cifar100_test, batch_size=self.batch_size, shuffle=False)
        return classes, num_classes, train_dataloader, test_dataloader


class SVHNLoader(BaseLoader):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.name = "svhn"

    def __call__(self, *args, **kwargs):
        svhn_train = SVHN(root=self.cfg["SVHN"]["root"],
                          split="train",
                          transform=self.transforms,
                          download=True)
        svhn_test = SVHN(root=self.cfg["SVHN"]["root"],
                         split="test",
                         transform=self.transforms,
                         download=True)
        svhn_extra = SVHN(root=self.cfg["SVHN"]["root"],
                          split="extra",
                          transform=self.transforms,
                          download=True)
        classes = self.cfg["SVHN"]["categories"]
        num_classes = 10
        print(f"正在使用{svhn_train}\n {svhn_test}\n {svhn_extra}")
        print(f"其中有{num_classes}个类别，分别是：{classes}等")
        train_dataloader = DataLoader(svhn_train, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(svhn_test, batch_size=self.batch_size, shuffle=False)
        return classes, num_classes, train_dataloader, test_dataloader
