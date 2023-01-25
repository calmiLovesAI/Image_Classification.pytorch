import warnings
from functools import partial
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from core.models.weights import ShuffleNetV2_weights
from core.utils import load_state_dict_from_url


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int) -> None:
        super().__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        if (self.stride == 1) and (inp != branch_features << 1):
            raise ValueError(
                f"Invalid combination of stride {stride}, inp {inp} and oup {oup} values. If stride == 1 then inp should be equal to oup // 2 << 1."
            )

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
            i: int, o: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(
            self,
            stages_repeats: List[int],
            stages_out_channels: List[int],
            num_classes: int = 1000,
            inverted_residual: Callable[..., nn.Module] = InvertedResidual,
    ) -> None:
        super().__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = [f"stage{i}" for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc_in = output_channels
        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ShuffleNetV2_x0_5(ShuffleNetV2):
    model_name = "ShuffleNetV2_x0_5"

    def __init__(self, cfg):
        super().__init__(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 48, 96, 192, 1024])
        pretrained = cfg["Train"]["pretrained"]
        device = cfg["device"]
        num_classes = cfg["num_classes"]
        default_shape = (224, 224)
        input_shape = tuple(cfg["Train"]["input_size"][1:])
        if input_shape != default_shape:
            warnings.warn(
                "你正在使用的输入图片大小：{}与{}默认的输入图片大小：{}不符！".format(input_shape, self.model_name,
                                                                                   default_shape))

        if pretrained:
            # 加载预训练模型
            state_dict = load_state_dict_from_url(url=ShuffleNetV2_weights.shufflenet_v2_x0_5_weights_url,
                                                  model_dir="web/shufflenet_v2_x0_5_ImageNet1K.pth",
                                                  map_location=device)
            self.load_state_dict(state_dict)
            print("Successfully loaded the state dict!")
            # 修改最后一层的结构
            self.fc = nn.Linear(self.fc_in, num_classes)


class ShuffleNetV2_x1_0(ShuffleNetV2):
    model_name = "ShuffleNetV2_x1_0"

    def __init__(self, cfg):
        super().__init__(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 116, 232, 464, 1024])
        pretrained = cfg["Train"]["pretrained"]
        device = cfg["device"]
        num_classes = cfg["num_classes"]
        default_shape = (224, 224)
        input_shape = tuple(cfg["Train"]["input_size"][1:])
        if input_shape != default_shape:
            warnings.warn(
                "你正在使用的输入图片大小：{}与{}默认的输入图片大小：{}不符！".format(input_shape, self.model_name,
                                                                                   default_shape))

        if pretrained:
            # 加载预训练模型
            state_dict = load_state_dict_from_url(url=ShuffleNetV2_weights.shufflenet_v2_x1_0_weights_url,
                                                  model_dir="web/shufflenet_v2_x1_0_ImageNet1K.pth",
                                                  map_location=device)
            self.load_state_dict(state_dict)
            print("Successfully loaded the state dict!")
            # 修改最后一层的结构
            self.fc = nn.Linear(self.fc_in, num_classes)


class ShuffleNetV2_x1_5(ShuffleNetV2):
    model_name = "ShuffleNetV2_x1_5"

    def __init__(self, cfg):
        super().__init__(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 176, 352, 704, 1024])
        pretrained = cfg["Train"]["pretrained"]
        device = cfg["device"]
        num_classes = cfg["num_classes"]
        default_shape = (224, 224)
        input_shape = tuple(cfg["Train"]["input_size"][1:])
        if input_shape != default_shape:
            warnings.warn(
                "你正在使用的输入图片大小：{}与{}默认的输入图片大小：{}不符！".format(input_shape, self.model_name,
                                                                                   default_shape))

        if pretrained:
            # 加载预训练模型
            state_dict = load_state_dict_from_url(url=ShuffleNetV2_weights.shufflenet_v2_x1_5_weights_url,
                                                  model_dir="web/shufflenet_v2_x1_5_ImageNet1K.pth",
                                                  map_location=device)
            self.load_state_dict(state_dict)
            print("Successfully loaded the state dict!")
            # 修改最后一层的结构
            self.fc = nn.Linear(self.fc_in, num_classes)


class ShuffleNetV2_x2_0(ShuffleNetV2):
    model_name = "ShuffleNetV2_x2_0"

    def __init__(self, cfg):
        super().__init__(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 244, 488, 976, 2048])
        pretrained = cfg["Train"]["pretrained"]
        device = cfg["device"]
        num_classes = cfg["num_classes"]
        default_shape = (224, 224)
        input_shape = tuple(cfg["Train"]["input_size"][1:])
        if input_shape != default_shape:
            warnings.warn(
                "你正在使用的输入图片大小：{}与{}默认的输入图片大小：{}不符！".format(input_shape, self.model_name,
                                                                                   default_shape))

        if pretrained:
            # 加载预训练模型
            state_dict = load_state_dict_from_url(url=ShuffleNetV2_weights.shufflenet_v2_x2_0_weights_url,
                                                  model_dir="web/shufflenet_v2_x2_0_ImageNet1K.pth",
                                                  map_location=device)
            self.load_state_dict(state_dict)
            print("Successfully loaded the state dict!")
            # 修改最后一层的结构
            self.fc = nn.Linear(self.fc_in, num_classes)
