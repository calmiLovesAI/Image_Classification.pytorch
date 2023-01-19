import torch


class CheckPoint:
    @staticmethod
    def save(model, optimizer, epoch, path):
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }, path)

    @staticmethod
    def load(path, device, model, optimizer=None):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        epoch = ckpt["epoch"]
        del ckpt
        return model, optimizer, epoch
