from typing import List

import torch
import sys


def get_optimizer(model, optimizer_cfg: dict):
    optimizer_name = optimizer_cfg["chosen_name"]
    if sys.version_info.major == 3 and sys.version_info.minor >= 10:
        # 使用macth-case语句
        match optimizer_name:
            case "SGD":
                return torch.optim.SGD(params=model.parameters(),
                                       lr=optimizer_cfg["SGD"]["lr"],
                                       momentum=optimizer_cfg["SGD"]["momentum"],
                                       weight_decay=optimizer_cfg["SGD"]["weight_decay"])
            case "Adam":
                return torch.optim.Adam(params=model.parameters(),
                                        lr=optimizer_cfg["Adam"]["lr"],
                                        betas=optimizer_cfg["Adam"]["betas"],
                                        eps=optimizer_cfg["Adam"]["eps"],
                                        weight_decay=optimizer_cfg["Adam"]["weight_decay"])
    else:
        if optimizer_name == "SGD":
            return torch.optim.SGD(params=model.parameters(),
                                   lr=optimizer_cfg["SGD"]["lr"],
                                   momentum=optimizer_cfg["SGD"]["momentum"],
                                   weight_decay=optimizer_cfg["SGD"]["weight_decay"])
        elif optimizer_name == "Adam":
            return torch.optim.Adam(params=model.parameters(),
                                    lr=optimizer_cfg["Adam"]["lr"],
                                    betas=optimizer_cfg["Adam"]["betas"],
                                    eps=optimizer_cfg["Adam"]["eps"],
                                    weight_decay=optimizer_cfg["Adam"]["weight_decay"])
    raise NotImplementedError(f"Optimizer {optimizer_name} is not implemented")


def get_lr_scheduler(optimizer, scheduler_cfg: dict):
    scheduler_name = scheduler_cfg["chosen_name"]
    if sys.version_info.major == 3 and sys.version_info.minor >= 10:
        match scheduler_name:
            case "MultiStepLR":
                return torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=scheduler_cfg["MultiStepLR"]["milestones"],
                                                            gamma=scheduler_cfg["MultiStepLR"]["gamma"],
                                                            last_epoch=-1)
            case "":
                return None
    else:
        if scheduler_name == "MultiStepLR":
            return torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=scheduler_cfg["MultiStepLR"]["milestones"],
                                                        gamma=scheduler_cfg["MultiStepLR"]["gamma"],
                                                        last_epoch=-1)
        elif scheduler_name == "":
            return None
    return NotImplementedError(f"Lr scheduler {scheduler_name} is not implemented")
