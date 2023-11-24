import torch


class ConfigBuilder:
    def __init__(self) -> None:
        pass

    def g_thick_map(self, input):
        out = (input * 10) + 15  # 5-25
        return out

    def a_thick_map(self, input):
        out = (input * 50) + 60  # 10-110
        return out

    def connect(self, input):
        config_list = []
        for i, bs in enumerate(input):
            config = [
                (bs[0], self.g_thick_map(bs[1]), 1.5168, 40),
                (bs[2], self.a_thick_map(bs[3]), 1, 40),
                (torch.inf, torch.inf, 1, 40),
            ]
            config_list.append(config)
        return config_list
