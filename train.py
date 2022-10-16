import torch

from config import BaseCfg
from core.models import select_model


def train(cfg, device):
    print("Pytorch version: {}, Train on {}".format(torch.__version__, device))


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建网络模型
    model = select_model()

    # 加载数据集
