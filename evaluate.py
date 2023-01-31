import torch

from core.loss import cross_entropy_loss
from core.metrics import accuracy, multi_label_f1_score


def evaluate_loop(model, dataloader, num_classes, device):
    model.eval()     # 切换为验证模式
    num_batches = len(dataloader)
    loss_fn = cross_entropy_loss()
    test_loss, acc, f1 = 0, 0, 0

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            print(f"Progress: {(100 * (i + 1) / len(dataloader)):.0f}%", end="\r")
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            pred = model(images)
            test_loss += loss_fn(pred, targets).item()
            acc += accuracy(pred, targets, batch_size).item()
            f1 += multi_label_f1_score(pred, targets, num_classes, device).item()

    test_loss /= num_batches
    acc /= num_batches
    f1 /= num_batches
    print_info = f"Accuracy: {(100*acc):>0.1f}%, F1 score: {(f1):>0.2f}, Avg loss: {test_loss:>8f}"
    print(f"Test: {print_info}")
    return print_info
