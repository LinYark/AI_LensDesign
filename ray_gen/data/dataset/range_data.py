from torch.utils.data.dataset import Dataset
import torch
class RangeData(Dataset):
    def __init__(self, ): 
        self.num = 99     
        self.g_t_min = 5
        self.g_t_max = 30
        self.a_t_min = 10
        self.a_t_max = 200
        self.epd = 1  #random
        self.hfov = 1 #random

    def __getitem__(self, idx):
        a = torch.tensor([1.,1.])
        return a


    def __len__(self):
        return self.num




