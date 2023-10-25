import torch
import torch.nn as nn

class SurfaceModule(nn.Module):
    def __init__(self, r=torch.inf, t=torch.inf, v=[], n=1,h=torch.inf):
        super(SurfaceModule, self).__init__()
        self.r = nn.Parameter(torch.tensor(r,dtype=float))
        self.t = nn.Parameter(torch.tensor(t,dtype=float))
        self.r.requires_grad = v[0]
        self.t.requires_grad = v[1]

        self.h = torch.tensor(h,dtype=float)
        self.n = torch.tensor(n,dtype=float)
        self.z = torch.tensor(0,dtype=float,requires_grad=True)

    def forward_track(self,):
        pass

    def reverse_track(self,):
        pass