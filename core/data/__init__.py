from core.data.dataloader import BaseLoader, Cifar10Loader, Cifar100Loader, SVHNLoader

DATASETS = {
    0: BaseLoader,
    1: Cifar10Loader,
    2: Cifar100Loader,
    3: SVHNLoader
}


def load_dataset(cfg):
    print("请从下面的数据集中选择一个：")
    for k, v in DATASETS.items():
        print("序号：{}，数据集：{}".format(k, v))
    idx = int(input("它的序号为："))
    if idx < 0 or idx >= len(DATASETS):
        raise ValueError("输入序号<{}>非法".format(idx))
    dataset = DATASETS[idx](cfg)
    print("你选择了{}数据集".format(dataset.name))
    return dataset.name, *dataset.__call__()