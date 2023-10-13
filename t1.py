
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

# class ct(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.p1 = nn.Parameter(torch.tensor([1.]))

# class net(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.ct1 = ct()
#         self.x	=	torch.tensor([2.])

#     def forward(self,):
#         y	= self.ct1.p1 ** self.x
#         return y
    
# net1 = net()
# optim	=	torch.optim.SGD(net1.parameters(),	lr=0.01)

# def lossf(z):
#     loss = (z - torch.tensor([4.]))**2
#     return loss

# lossf2 = nn.MSELoss()

# for i in range(100):
#     z = net1()
#     loss = lossf(z)
#     optim.zero_grad()
#     loss.backward()
#     # net1.ct1.p1.data = net1.ct1.p1.data- net1.ct1.p1.grad.data*0.3
#     # net1.ct1.p1.grad.zero_()
#     optim.step()
#     print(loss,net1.ct1.p1,net1.x,"\n")


c = list(range(4,-1,-1))
c1 = [None]*5
c1[5]=0
# c.reverse()

a = 1