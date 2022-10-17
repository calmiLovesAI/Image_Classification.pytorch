from typing import Union, List, Tuple

import yaml


class Yaml:
    def __init__(self, yaml_filepath: Union[List, Tuple]):
        self.filepath = yaml_filepath

    def parse(self):
        cfg = dict()
        for file in self.filepath:
            print("解析 {}...".format(self.filepath))
            with open(file, encoding="utf-8") as f:
                # cfg |= yaml.load(f.read(), Loader=yaml.FullLoader)
                cfg.update(yaml.load(f.read(), Loader=yaml.FullLoader))
        print("合并解析结果")
        return cfg
