from pathlib import Path

import torch
from tqdm import tqdm

from core.data import load_dataset
from core.loss import cross_entropy_loss
from core.metrics import MeanMetric
from core.models import select_model
from core.parse_yaml import Yaml


def train_loop(cfg, model, dataloader, device):
    print("Pytorch version: {}, Train on {}".format(torch.__version__, device))
    # 训练轮数
    epochs = cfg["Train"]["epochs"]
    save_frequency = cfg["Train"]["save_frequency"]
    save_path = cfg["Train"]["save_path"]
    # 优化器
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg["Train"]["learning_rate"])
    # 损失函数
    loss_fn = cross_entropy_loss()
    loss_mean = MeanMetric()

    for epoch in range(epochs):
        with tqdm(dataloader, desc="Epoch-{}/{}".format(epoch, epochs)) as pbar:
            for i, (images, targets) in enumerate(pbar):

                images = images.to(device)
                targets = targets.to(device)

                preds = model(images)
                loss = loss_fn(preds, targets)
                loss_mean.update(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({"loss": "{}".format(loss_mean.result())})
        loss_mean.reset()

        if epoch % save_frequency == 0:
            torch.save(model.state_dict(), Path(save_path).joinpath("{}_epoch-{}.pth".format(model.model_name, epoch)))

    torch.save(model, Path(save_path).joinpath("{}_entire_model.pth".format(model.model_name)))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取配置文件
    cfg = Yaml(yaml_filepath=["./experiments/config.yaml", "./experiments/data.yaml"]).parse()
    print(cfg)

    # 加载数据集
    classes, num_classes, train_dataloader = load_dataset(cfg)
    print("nc = {}".format(num_classes))

    # 创建网络模型
    model = select_model()(cfg, num_classes)
    model.to(device=device)

    train_loop(cfg, model, train_dataloader, device)