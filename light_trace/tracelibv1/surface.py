import torch
import torch.nn as nn
import numpy as np

class Surface(nn.Module):
    def __init__(self, r=np.inf, t=np.inf, v=[], n=1, h=np.inf):
        super(Surface, self).__init__()
        self.r = nn.Parameter(torch.tensor(r,dtype=float))
        self.t = nn.Parameter(torch.tensor(t,dtype=float))
        self.h = nn.Parameter(torch.tensor(h,dtype=float))
        self.n = nn.Parameter(torch.tensor(n,dtype=float))
        self.r.requires_grad = v[0]
        self.t.requires_grad = v[1]
        self.h.requires_grad = False
        self.n.requires_grad = False

    def forward(self,):
        pass

class OpticalSystem(nn.Module):
    def __init__(self,):
        super(OpticalSystem, self).__init__()
        self.surfaces = nn.ModuleList()

    def add(self, r=np.inf, t=np.inf, v=[], n=1, h=np.inf):
        self.surfaces.append(Surface(r,t,v,n,h))
    
    def forward(self,):
        pass

if __name__=="__main__":
    os = OpticalSystem()
    os.add(100,100,[True,True],1,50)
    a = 1