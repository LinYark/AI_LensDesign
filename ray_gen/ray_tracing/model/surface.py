import torch
import torch.nn as nn


class SurfaceModule(nn.Module):
    def __init__(self, c, t, n, h=torch.inf):
        super(SurfaceModule, self).__init__()
        self.c = c
        if torch.is_tensor(t):
            self.t = 1.0 / t
        else:
            self.t = torch.tensor(1.0 / t)
        self.n = torch.tensor(n, dtype=float)
        self.h = torch.tensor(h, dtype=float)
        self.z = torch.tensor(0, dtype=float)

    def forward_track(self):
        pass

    def reverse_track(self):
        pass
