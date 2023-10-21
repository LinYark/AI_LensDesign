import torch.nn as nn
import torch 

# a = torch.tensor([1.],requires_grad=True)
# b = torch.tensor([2.],requires_grad=True)
# # l = torch.stack([a,b])
# l = [a,b]
# c = l[0]**2-l[1]**2

# c.backward()
# print(a.grad,b.grad)
# print(36.0334*torch.sin(torch.tensor(5.0/180*torch.pi))
# for i in range(3):
#     print(i)
theta = 0.1
thetas = torch.linspace(torch.pi+theta, torch.pi-theta, 10)
thetas = torch.flip(thetas) 

a = 1

# import torch
# x = torch.tensor([1.0,2.0],requires_grad=True)
# y = (x + 2)**2
# z = torch.mean(y)
# z.backward()
# print(x.grad)