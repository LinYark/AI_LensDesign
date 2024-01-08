import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import numpy as np

a = torch.tensor([1.0, 2.0])
b = torch.tensor([1.0, 2.0])
c = [a, b]
t = torch.stack(c, 1)
m = torch.mean(t)
d = 1
