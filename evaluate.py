import torch

from core.loss import cross_entropy_loss


def evaluate_loop(model, dataloader, device):
    model.eval()     # 切换为验证模式
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_fn = cross_entropy_loss()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            pred = model(images)
            test_loss += loss_fn(pred, targets).item()
            correct += (pred.argmax(1) == targets).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print_info = f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}"
    print(f"Test Error: \n {print_info}")
    return print_info
