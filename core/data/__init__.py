from core.data.dataloader import BaseLoader, Cifar10Loader, Cifar100Loader

DATASETS = {
    0: BaseLoader,
    1: Cifar10Loader,
    2: Cifar100Loader
}


def load_dataset(cfg):
    print("请从下面的数据集中选择一个：")
    for k, v in DATASETS.items():
        print("序号：{}，数据集：{}".format(k, v))
    idx = int(input("它的序号为："))
    if idx < 0 or idx >= len(DATASETS):
        raise ValueError("输入序号<{}>非法".format(idx))
    return DATASETS[idx](cfg).__call__()