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

    def get_loss(self, sys_param, lens_system):
        ray_trace_detail = self.losser.get_loss(sys_param, lens_system)
        return ray_trace_detail

    def show(self, epoch, shotpath, loss):
        rays_list, sins_list, surfaces_list = (
            loss["rays_list"],
            loss["sins_list"],
            loss["surfaces_list"],
        )
        listener = (rays_list, sins_list, surfaces_list)
        self.drawer.show(listener, epoch, shotpath)

    def backup(self, loss):
        all_loss = loss["all_loss"].cpu().item()
        RMS_loss = loss["RMS_loss"].cpu().item()
        sins_loss = loss["sins_loss"].cpu().item()
        thick_loss = loss["thick_loss"].cpu().item()
        na_loss = loss["na_loss"].cpu().item()
        refraction_loss = loss["refraction_loss"].cpu().item()
        loss_info = [
            all_loss,
            RMS_loss,
            sins_loss,
            thick_loss,
            na_loss,
            refraction_loss,
        ]
        return loss_info
