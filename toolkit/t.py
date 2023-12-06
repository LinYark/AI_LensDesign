import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import numpy as np

a = torch.tensor([1.0])
b = torch.tensor([1.0])
c = torch.tensor([1.0])
d = torch.tensor([1.0])
e = [[a, b], [c, d]]
f = np.array(e)


t = torch.stack(e, 1)
