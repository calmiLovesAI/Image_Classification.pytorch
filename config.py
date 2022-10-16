# 配置文件
from typing import List, Tuple


class BaseCfg:
    def __init__(self, num_classes, input_size=None):
        self.epoch = 50
        self.batch_size = 8
        self.num_classes = num_classes

        if input_size is None:
            self.input_size = (224, 224)
        else:
            if isinstance(input_size, (List, Tuple)):
                self.input_size = tuple(input_size)
            if isinstance(input_size, (int, float)):
                self.input_size = (input_size, input_size)

        self.dataset_root = ""

        