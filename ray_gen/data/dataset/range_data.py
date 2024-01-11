from torch.utils.data.dataset import Dataset
import torch
from ...config.config import TrainCfg


class RangeData(Dataset):
    def __init__(self):
        self.num = 20000
        # self.g_t_min = 5
        # self.g_t_max = 20
        # self.a_t_min = 10
        # self.a_t_max = 100
        self.epd = 1  # random
        self.hfov = 1  # random

    def __getitem__(self, idx):
        random_tensor = torch.rand([TrainCfg().sys_num]) * 2 - 1  # -1-1
        return random_tensor

    def __len__(self):
        return self.num


class ValidRangeData(Dataset):
    def __init__(self):
        self.num = 80000
        self.epd = 1
        self.hfov = 1

    def __getitem__(self, idx):
        random_tensor = torch.rand([3])
        random_tensor[0] = random_tensor[0] * 2 - 1
        random_tensor[1] = random_tensor[1] * 1 / 15 * 2 - 1
        random_tensor[2] = random_tensor[2] * 2 - 1
        return random_tensor

    def __len__(self):
        return self.num
