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
        self.map = nn.Sigmoid()

    def forward(self, x):
        output = self.model(x)
        # output_maped = self.map(output)
        return output
