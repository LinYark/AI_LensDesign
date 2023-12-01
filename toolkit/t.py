import torch

h = torch.tensor([-1, 2, 0.5])
a1 = (h > 1) + (h < 0)
p = torch.where(h > 1)
p1 = torch.where(h < 0)
a = h[p]
a = 1
