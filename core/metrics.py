import torch
import torchmetrics


class MeanMetric:
    def __init__(self):
        self.accumulated = 0
        self.count = 0

    def update(self, value):
        self.accumulated += value
        self.count += 1

    def result(self):
        return self.accumulated / self.count

    def reset(self):
        self.__init__()


def multi_label_f1_score(preds, targets, num_labels, device):
    targets_one_hot = torch.nn.functional.one_hot(targets, num_labels)
    f1 = torchmetrics.classification.MultilabelF1Score(num_labels=num_labels,
                                                       average='macro').to(device)
    return f1(preds, targets_one_hot).to()


def accuracy(preds, targets, num_batches):
    return (preds.argmax(1) == targets).type(torch.float).sum() / num_batches
