
# import matplotlib.pyplot as plt
# import numpy as np
# a = np.append(np.arange(2,3,0.1),3)
# thetas = np.flip(a) 

# center = (1, 1)
# radius = 5
# theta = np.radians(np.arange(-90, 90, 0.1))
# x = center[0] + radius * np.cos(theta)
# y = center[1] + radius * np.sin(theta)
# plt.plot(x, y)
# plt.show()

import torch
import torch.nn as nn
device = torch.device("cuda:0")

class ct(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor([1.]))

class net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.ct1 = ct()
        # self.ct2 = ct()
        # self.list = [self.ct1,self.ct2]
        # self.list = nn.ModuleList([self.ct1,self.ct2])
        self.list = nn.ModuleList([nn.Linear(2, 1)])#nn.ModuleList([nn.Linear(2, 1)])

        self.x	=	 torch.tensor([2.,3.]).to(device)

    def forward(self,):
        y = self.list[0](self.x)
        # y = self.list[0].p*self.x[0] + self.list[1].p*self.x[1] 
        # print(y)
        return y
    

net1 = net()
net1.cuda()
optim	=	torch.optim.SGD(net1.parameters(),	lr=0.01)

def lossf(z):
    loss = (z )**2
    return loss

lossf2 = nn.MSELoss()

for i in range(10):
    z = net1()
    loss = lossf(z)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(z, net1.list,"\n")#, net1.ct2.p, net1.x,

a  = torch.tensor([1.]).cuda()
b  = torch.tensor([2.])
c = a * b
# c = list(range(4,-1,-1))
# c1 = [None]*5
# c1[5]=0
# c.reverse()

a = 1