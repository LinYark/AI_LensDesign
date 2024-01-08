from .lib.base_loss import BaseLoss
from .lib.drawer import OpticalSystemDrawer
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

import numpy as np
import pandas as pd


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
        f_num_loss = loss["f_num_loss"].cpu().item()
        refraction_loss = loss["refraction_loss"].cpu().item()
        loss_info = [
            all_loss,
            RMS_loss,
            sins_loss,
            thick_loss,
            f_num_loss,
            refraction_loss,
        ]
        return loss_info

    def prepare_for_excel(self, loss):
        rms = loss["RMS_loss"].cpu().item()
        f_num_pre = loss["f_num_pre"][0].cpu().item()
        epd, hfov = (loss["sys_param_transed"][0][0], loss["sys_param_transed"][0][1])
        f = epd / f_num_pre
        new_data = {
            "f_num_pre": [f_num_pre],
            "epd": [epd],
            "f": [f],
            "hfov": [hfov],
            "rms": [rms],
            "rms/f": [rms / f],
        }
        df_new = pd.DataFrame(new_data)
        if os.path.exists("./input.xlsx"):
            df = pd.read_excel("./input.xlsx")
            df1 = pd.concat([df, df_new], axis=0, ignore_index=True)
            df1.to_excel("./input.xlsx", index=False)

            lenth = df1.shape[0]
            if lenth > 10000:
                exit()
        else:
            df_new.to_excel("./input.xlsx", index=False)
