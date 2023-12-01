from .lib.base_loss import BaseLoss
from .lib.drawer import OpticalSystemDrawer
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime


class LossBuilder:
    def __init__(self):
        self.losser = BaseLoss()
        self.drawer = OpticalSystemDrawer()
        self.listener = None

    def get_loss(self, sys_param, lens_system):
        all_loss, rays_list, sins_list, surfaces_list = self.losser.get_loss(
            sys_param, lens_system
        )
        self.listener = (rays_list, sins_list, surfaces_list)
        return all_loss

    def show(self, epoch, shotpath):
        self.drawer.show(self.listener, epoch, shotpath)
