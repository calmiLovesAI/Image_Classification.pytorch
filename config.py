# 配置文件
from typing import List, Tuple


class BaseCfg:
    def __init__(self, model_cfg, input_size=None):
        self.epoch = 50
        self.batch_size = 8

        if model_cfg is not None and isinstance(model_cfg, dict):
            self.input_size = model_cfg["input_size"]
        if input_size is None:
            self.input_size = (224, 224)
        else:
            if isinstance(input_size, (List, Tuple)):
                self.input_size = tuple(input_size)
            if isinstance(input_size, (int, float)):
                self.input_size = (input_size, input_size)