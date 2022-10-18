from .alexnet import AlexNet
from .vgg import VGG16, VGG19

MODELS = {
    0: [AlexNet, "AlexNet"],
    1: [VGG16, "VGG16"],
    2: [VGG19, "VGG19"]
}


def select_model():
    # 选择网络模型
    print("请从下面的网络模型中选择一个：")
    for k, v in MODELS.items():
        print("序号：{}，网络模型名：{}".format(k, v[1]))
    idx = int(input("它的序号为："))
    if idx < 0 or idx >= len(MODELS):
        raise ValueError("输入序号<{}>非法".format(idx))
    print("已选择模型：{}".format(MODELS[idx][1]))
    return MODELS[idx][0]
