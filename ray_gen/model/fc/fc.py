import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np


class FCSmall(nn.Module):
    def __init__(self, in_size, out_size):
        super(FCSmall, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_size),
        )

    def forward(self, x):
        output = self.net(x)
        return output
