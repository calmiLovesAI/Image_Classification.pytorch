import torch

from core.data import load_dataset
from core.models import select_model
from core.parse_yaml import Yaml


def train(cfg, device):
    print("Pytorch version: {}, Train on {}".format(torch.__version__, device))


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读取配置文件
    cfg = Yaml(yaml_filepath=["./experiments/config.yaml", "./experiments/data.yaml"]).parse()
    print(cfg)

    # 加载数据集
    classes, num_classes, train_dataloader = load_dataset(cfg)
    print("nc = {}".format(num_classes))

    # 创建网络模型
    model = select_model()(cfg, num_classes)

