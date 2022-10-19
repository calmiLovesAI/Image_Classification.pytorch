import warnings

import torch
import torch.nn as nn


class BaseVGG(nn.Module):
    def __init__(self, cfg, structure, num_classes, use_bn=True):
        """
        URL: https://arxiv.org/abs/1409.1556
        @article{simonyan2014very,
          title={Very deep convolutional networks for large-scale image recognition},
          author={Simonyan, Karen and Zisserman, Andrew},
          journal={arXiv preprint arXiv:1409.1556},
          year={2014}
        }
        """
        super(BaseVGG, self).__init__()
        c_in = cfg["Train"]["input_size"][0]
        self.structure = structure

        self.features = self._make_layers(c_in, batch_norm=use_bn)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def _make_layers(self, c_in, batch_norm=False):
        layers = []
        in_channels = c_in
        for v in self.structure:
            if v == 'M':  # 池化层
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:  # 卷积层
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.classifier(x)
        return x


class VGG11(BaseVGG):
    def __init__(self, cfg, num_classes):
        super(VGG11, self).__init__(cfg,
                                    [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
                                    num_classes,
                                    True)
        self.model_name = "VGG11"
        default_shape = (224, 224)
        input_shape = tuple(cfg["Train"]["input_size"][1:])
        if input_shape != default_shape:
            warnings.warn(
                "你正在使用的输入图片大小：{}与{}默认的输入图片大小：{}不符！".format(input_shape, self.model_name,
                                                                                   default_shape))


class VGG13(BaseVGG):
    def __init__(self, cfg, num_classes):
        super(VGG13, self).__init__(cfg,
                                    [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
                                    num_classes,
                                    True)
        self.model_name = "VGG13"
        default_shape = (224, 224)
        input_shape = tuple(cfg["Train"]["input_size"][1:])
        if input_shape != default_shape:
            warnings.warn(
                "你正在使用的输入图片大小：{}与{}默认的输入图片大小：{}不符！".format(input_shape, self.model_name,
                                                                                   default_shape))


class VGG16(BaseVGG):
    def __init__(self, cfg, num_classes):
        super(VGG16, self).__init__(cfg,
                                    [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512,
                                     "M"],
                                    num_classes,
                                    True)
        self.model_name = "VGG16"
        default_shape = (224, 224)
        input_shape = tuple(cfg["Train"]["input_size"][1:])
        if input_shape != default_shape:
            warnings.warn(
                "你正在使用的输入图片大小：{}与{}默认的输入图片大小：{}不符！".format(input_shape, self.model_name,
                                                                                   default_shape))


class VGG19(BaseVGG):
    def __init__(self, cfg, num_classes):
        super(VGG19, self).__init__(cfg,
                                    [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512,
                                     512, 512, 512, "M"],
                                    num_classes,
                                    True)
        self.model_name = "VGG19"
        default_shape = (224, 224)
        input_shape = tuple(cfg["Train"]["input_size"][1:])
        if input_shape != default_shape:
            warnings.warn(
                "你正在使用的输入图片大小：{}与{}默认的输入图片大小：{}不符！".format(input_shape, self.model_name,
                                                                                   default_shape))
