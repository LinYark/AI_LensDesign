import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
from .fc.fc import FCSmall

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        self.model = FCSmall()

    def forward(self, x):
        output = self.model(x)
        return output







