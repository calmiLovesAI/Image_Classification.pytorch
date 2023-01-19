import torch

from core.checkpoint import CheckPoint
from core.data import load_dataset
from core.models import select_model
from core.parse_yaml import Yaml
from core.inference import Classify
from core.utils import read_image

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取配置文件
    cfg = Yaml(yaml_filepath=["./experiments/config.yaml", "./experiments/data.yaml"]).parse()
    print(cfg)

    # 加载数据集
    _, classes, num_classes, _, _ = load_dataset(cfg)
    cfg.update({"num_classes": num_classes})
    cfg.update({"device": device})

    # 创建网络模型
    model = select_model()(cfg)
    model, _, _, _ = CheckPoint.load(path=cfg["Test"]["load_pth"], device=device, model=model)

    test_pictures = cfg["Test"]["test_pictures"]

    Classify(model,
             images=read_image(image_paths=test_pictures, add_dim=True, convert_to_tensor=True,
                               resize=True, size=cfg["Train"]["input_size"][1:]),
             class_name=classes,
             print_on=True).process_image()
