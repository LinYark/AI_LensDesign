import torch
import torch.nn as nn
from ...ray_tracing.ray_tracing_builder import RayTracingBuilder

EPSILON = 1e-9


class BaseLoss:
    def __init__(self) -> None:
        pass

    def get_source_data(self, sys_param, lens_system):
        rtb = RayTracingBuilder()
        rays_list, sins_list, surfaces_list = rtb.get_rays_and_surfaces(
            sys_param, lens_system
        )
        return rays_list, sins_list, surfaces_list

    def get_loss(self, sys_param, lens_system):
        rays_list, sins_list, surfaces_list = self.get_source_data(
            sys_param, lens_system
        )
        RMS_loss = self.get_RMS_loss(rays_list, sins_list, surfaces_list)
        sins_loss = self.get_sin_loss(rays_list, sins_list, surfaces_list)
        thick_loss = self.get_thick_loss(lens_system)
        all_loss = RMS_loss + sins_loss + thick_loss * 10
        if torch.isnan(all_loss):
            a = 1
        return all_loss, rays_list, sins_list, surfaces_list

    def get_RMS_loss(self, rays_list, sins_list, surfaces_list):
        bs = len(surfaces_list)
        RMS_loss_list = []
        for i in range(bs):
            surface = surfaces_list[i]
            rays = rays_list[i]
            sins = sins_list[i]
            if len(sins) > 0:
                continue

            z = surface[-1].z
            final_light = rays[-1]

            y = []
            for angle_lights in final_light:
                angle_ys = []

                for single_light in angle_lights:
                    if torch.abs(single_light.u) < EPSILON:
                        single_y = single_light.q
                    else:
                        single_y = torch.tan(single_light.u) * (
                            z
                            - single_light.p
                            + single_light.q / torch.sin(single_light.u)
                        )
                    angle_ys.append(single_y)

                angle_y_tensors = torch.stack(angle_ys)
                y.append(angle_y_tensors)

            y_tensor = torch.stack(y)
            y_var = torch.std(y_tensor, 1)
            RMS_loss = y_var.dot(y_var)
            RMS_loss_list.append(RMS_loss)

        y_loss = torch.tensor(0.0)
        if len(RMS_loss_list) > 0:
            RMS_loss_list_torch = torch.mean(torch.stack(RMS_loss_list))
            y_loss = RMS_loss_list_torch
        if torch.isnan(y_loss):
            a = 1
        return y_loss

    def get_sin_loss(self, rays_list, sins_list, surfaces_list):
        bs = len(surfaces_list)
        sins_loss_list = []
        for i in range(bs):
            surface = surfaces_list[i]
            rays = rays_list[i]
            sins = sins_list[i]
            if len(sins) == 0:
                continue
            first_breaker = sins[0]
            sins_loss = torch.pow(first_breaker, 2)
            sins_loss_list.append(sins_loss)
        y_loss = torch.tensor(0.0)
        if len(sins_loss_list) > 0:
            sins_loss_list_torch = torch.mean(torch.stack(sins_loss_list))
            y_loss = sins_loss_list_torch
        return y_loss

    def get_thick_loss(self, lens_system):
        bs = len(lens_system)
        thick_loss_list = []
        for i in range(bs):
            thicks = lens_system[i][1::2]
            thicks_fail = thicks[torch.where((thicks < -1) + (thicks > 1))]
            if len(thicks_fail) == 0:
                continue
            thick_loss = thicks_fail.dot(thicks_fail - 0.5)
            thick_loss_list.append(thick_loss)
        y_loss = torch.tensor(0.0)
        if len(thick_loss_list) > 0:
            thick_loss_list_torch = torch.mean(torch.stack(thick_loss_list))
            y_loss = thick_loss_list_torch
        return y_loss
