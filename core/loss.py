import torch


def cross_entropy_loss():
    return torch.nn.CrossEntropyLoss(reduction='mean')