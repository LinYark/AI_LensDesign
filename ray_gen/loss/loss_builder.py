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
        ray_trace_detail = self.losser.get_loss(sys_param, lens_system)
        rays_list, sins_list, surfaces_list = (
            ray_trace_detail["rays_list"],
            ray_trace_detail["sins_list"],
            ray_trace_detail["surfaces_list"],
        )

        self.listener = (rays_list, sins_list, surfaces_list)
        return {
            "all_loss": ray_trace_detail["all_loss"],
            "RMS_loss": ray_trace_detail["RMS_loss"],
            "sins_loss": ray_trace_detail["sins_loss"],
            "thick_loss": ray_trace_detail["thick_loss"],
            "na_loss": ray_trace_detail["na_loss"],
        }

    def show(self, epoch, shotpath):
        self.drawer.show(self.listener, epoch, shotpath)

    def backup(self, loss):
        all_loss = loss["all_loss"].cpu().item()
        RMS_loss = loss["RMS_loss"].cpu().item()
        sins_loss = loss["sins_loss"].cpu().item()
        thick_loss = loss["thick_loss"].cpu().item()
        na_loss = loss["na_loss"].cpu().item()
        loss_info = [all_loss, RMS_loss, sins_loss, thick_loss, na_loss]
        return loss_info
