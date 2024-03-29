from .alexnet import AlexNet
from .vgg import VGG16, VGG19
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, \
    ResNeXt50_32x4d, ResNeXt101_32x8d, ResNeXt101_64x4d, Wide_ResNet_50_2, Wide_ResNet_101_2
from .vit import ViT_B_16, ViT_B_32, ViT_L_16, ViT_L_32, ViT_H_14
from .regnet import RegNet_Y_400MF, RegNet_Y_800MF, RegNet_Y_8GF, RegNet_Y_128GF
from .mobilenet import MobileNetV1
from .mobilenetv2 import MobileNetV2Pretrained
from .mobilenetv3 import MobileNetV3Large, MobileNetV3Small
from .shufflenet import ShuffleNetV2_x0_5, ShuffleNetV2_x1_0, ShuffleNetV2_x1_5, ShuffleNetV2_x2_0
from .efficientnet import EfficientNet_B0, EfficientNet_V2_S
from .convnext import ConvNeXt_Tiny, ConvNeXt_Small, ConvNeXt_Base, ConvNeXt_Large

MODELS = [AlexNet, VGG16, VGG19,
          ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
          ResNeXt50_32x4d, ResNeXt101_32x8d, ResNeXt101_64x4d,
          Wide_ResNet_50_2, Wide_ResNet_101_2,
          ViT_B_16, ViT_B_32, ViT_L_16, ViT_L_32, ViT_H_14,
          RegNet_Y_400MF, RegNet_Y_800MF, RegNet_Y_8GF, RegNet_Y_128GF,
          MobileNetV1, MobileNetV2Pretrained, MobileNetV3Large, MobileNetV3Small,
          ShuffleNetV2_x0_5, ShuffleNetV2_x1_0, ShuffleNetV2_x1_5, ShuffleNetV2_x2_0,
          EfficientNet_B0, EfficientNet_V2_S,
          ConvNeXt_Tiny, ConvNeXt_Small, ConvNeXt_Base, ConvNeXt_Large]

MODELS_DICT = dict((k, v) for k, v in enumerate(MODELS))


def select_model():
    # 选择网络模型
    print("请从下面的网络模型中选择一个：")
    for k, v in MODELS_DICT.items():
        print("序号：{}，网络模型名：{}".format(k, v.model_name))
    idx = int(input("它的序号为："))
    if idx < 0 or idx >= len(MODELS_DICT):
        raise ValueError("输入序号<{}>非法".format(idx))
    print("已选择模型：{}".format(MODELS_DICT[idx].model_name))
    return MODELS_DICT[idx]
