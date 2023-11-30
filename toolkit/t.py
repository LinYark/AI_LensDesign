import torch

for i in range(10):
    print((torch.rand([2])))

a = torch.rand([2])
a1 = a.cuda()
b = 1
