import torch


def train(device):
    print("Train on {}".format(device))


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(device)
