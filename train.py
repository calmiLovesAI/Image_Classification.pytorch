import os.path
import traceback
import logging
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from core.checkpoint import CheckPoint
from core.data import load_dataset
from core.loss import cross_entropy_loss
from core.metrics import MeanMetric, multi_label_f1_score, accuracy
from core.models import select_model
from core.optimizer import get_optimizer, get_lr_scheduler
from core.parse_yaml import Yaml
from core.utils import get_format_filename, get_current_format_time, auto_make_dirs
from evaluate import evaluate_loop


def train_loop(cfg, model, train_loader, test_loader, device):
    train_logger = logging.getLogger("TRAIN")
    train_logger_file = os.path.join(cfg["Train"]["log"]["root"], get_format_filename(model_name=model.model_name,
                                                                                      dataset_name=cfg["dataset"],
                                                                                      addition=get_current_format_time() + ".log"))
    auto_make_dirs(train_logger_file)
    handler = logging.FileHandler(filename=train_logger_file, encoding="utf-8")
    train_logger.setLevel(logging.INFO)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    train_logger.addHandler(handler)

    print("Pytorch version: {}, Train on {}".format(torch.__version__, device))
    print("训练参数如下：")
    for k, v in cfg["Train"].items():
        print(f"{k} : {v}")
    # 训练轮数
    epochs = cfg["Train"]["epochs"]
    save_frequency = cfg["Train"]["save_frequency"]
    save_path = cfg["Train"]["save_path"]
    load_weights = cfg["Train"]["load_weights"]
    tensorboard_on = cfg["Train"]["tensorboard_on"]
    input_size = cfg["Train"]["input_size"]
    batch_size = cfg["Train"]["batch_size"]
    mixed_precision = cfg["Train"]["mixed_precision"]
    # 优化器
    optimizer = get_optimizer(model, optimizer_cfg=cfg["Optimizer"])
    scheduler = get_lr_scheduler(optimizer, scheduler_cfg=cfg["Scheduler"])
    # 损失函数
    loss_fn = cross_entropy_loss()
    loss_mean = MeanMetric()
    f1_mean = MeanMetric()
    acc_mean = MeanMetric()

    start_epoch = 0
    if load_weights != "":
        model, optimizer, scheduler, start_epoch = CheckPoint.load(path=load_weights, device=device, model=model,
                                                                   optimizer=optimizer, scheduler=scheduler)
        start_epoch += 1
        print(f"加载权重文件{load_weights}成功！将从epoch-{start_epoch}处恢复训练")

    if tensorboard_on:
        # 在控制台使用命令 tensorboard --logdir=runs 进入tensorboard面板
        writer = SummaryWriter()
        try:
            writer.add_graph(model, torch.randn(batch_size, *input_size, dtype=torch.float32, device=device))
        except Exception:
            traceback.print_exc()

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, epochs):
        model.train()  # 切换为训练模式
        loss_mean.reset()
        f1_mean.reset()
        acc_mean.reset()
        with tqdm(train_loader, desc="Epoch-{}/{}".format(epoch, epochs)) as pbar:
            for i, (images, targets) in enumerate(pbar):
                batch_size = images.size(0)
                images = images.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        preds = model(images)
                        loss = loss_fn(preds, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    preds = model(images)
                    loss = loss_fn(preds, targets)
                    loss.backward()
                    optimizer.step()

                loss_mean.update(loss.item())
                f1_mean.update(multi_label_f1_score(preds, targets, cfg["num_classes"], device).item())
                acc_mean.update(accuracy(preds, targets, num_batches=batch_size).item())

                if tensorboard_on:
                    writer.add_scalar(tag="Loss", scalar_value=loss_mean.result(),
                                      global_step=epoch * len(train_loader) + i)
                    writer.add_scalar(tag="Accuracy", scalar_value=acc_mean.result(),
                                      global_step=epoch * len(train_loader) + i)
                    writer.add_scalar(tag="F1 score", scalar_value=f1_mean.result(),
                                      global_step=epoch * len(train_loader) + i)
                    writer.add_scalar(tag="lr", scalar_value=optimizer.state_dict()["param_groups"][0]["lr"],
                                      global_step=epoch * len(train_loader) + i)

                pbar.set_postfix({"loss": "{}".format(loss_mean.result()),
                                  "accuracy": "{:.4f}%".format(100 * acc_mean.result()),
                                  "f1_score": "{:.6f}".format(f1_mean.result())})

                if i % cfg["Train"]["log"]["print_freq"] == 0:
                    train_logger.info(
                        msg="Epoch: {}/{}, step: {}/{}, Loss: {}, Accuracy: {:.4f}%, F1 score: {:.6f}".format(epoch,
                                                                                                              epochs,
                                                                                                              i,
                                                                                                              len(train_loader),
                                                                                                              loss_mean.result(),
                                                                                                              100 * acc_mean.result(),
                                                                                                              f1_mean.result()))

        if scheduler is not None:
            scheduler.step()

        # 验证
        evaluate_result = evaluate_loop(model, test_loader, cfg["num_classes"], device)
        train_logger.info(msg=f"===========Evaluate after epoch-{epoch}============\n {evaluate_result}")

        if epoch % save_frequency == 0:
            CheckPoint.save(model, optimizer, scheduler, epoch,
                            path=Path(save_path).joinpath(
                                "{}_{}_epoch-{}.pth".format(model.model_name, cfg["dataset"], epoch)))

    if tensorboard_on:
        writer.close()

    CheckPoint.save(model, optimizer, scheduler, epochs,
                    path=Path(save_path).joinpath("{}_{}_weights.pth".format(model.model_name, cfg["dataset"])))
    torch.save(model, Path(save_path).joinpath("{}_{}_entire_model.pth".format(model.model_name, cfg["dataset"])))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取配置文件
    cfg = Yaml(yaml_filepath=["./experiments/config.yaml",
                              "./experiments/data.yaml",
                              "./experiments/optimizer.yaml"]).parse()
    print(cfg)

    # 加载数据集
    dataset_name, classes, num_classes, train_dataloader, test_dataloader = load_dataset(cfg)
    cfg.update({"dataset": dataset_name})
    cfg.update({"num_classes": num_classes})
    cfg.update({"device": device})

    # 创建网络模型
    model = select_model()(cfg)
    model.to(device=device)

    train_loop(cfg, model, train_dataloader, test_dataloader, device)
