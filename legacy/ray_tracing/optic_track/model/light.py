import torch
import torch.nn as nn


class LightModule(nn.Module):
    def __init__(self, q=10, u=0, p=0, c="r"):
        super(LightModule, self).__init__()
        if torch.is_tensor(q):
            self.q = q
        else:
            self.q = torch.tensor(q, dtype=float)
        if torch.is_tensor(u):
            self.u = u
        else:
            self.u = torch.tensor(u, dtype=float)
        if torch.is_tensor(p):
            self.p = p
        else:
            self.p = torch.tensor(p, dtype=float)
        # self.q = torch.tensor(q, dtype=float)
        # self.p = torch.tensor(p, dtype=float)
        # self.u = torch.tensor(u, dtype=float)
        self.c = c

    def forward(self, light):
        pass
