import torch
import torch.nn as nn
from ...ray_tracing.ray_tracing_builder import RayTracingBuilder
from ...config.config import TrainCfg

EPSILON = 1e-9


class BaseLoss:
    def __init__(self) -> None:
        pass

    def get_source_data(self, sys_param, lens_system):
        rtb = RayTracingBuilder()
        info = rtb.get_rays_and_surfaces(sys_param, lens_system)
        return info

    def get_loss(self, sys_param, lens_system):
        (
            rays_list,
            sins_list,
            surfaces_list,
            intersections_tensor,
            f,
            sys_param_transed_list,
        ) = self.get_source_data(sys_param, lens_system)
        RMS_loss = self.get_RMS_loss(rays_list, sins_list, surfaces_list)
        sins_loss = self.get_sin_loss(rays_list, sins_list, surfaces_list)
        thick_loss = self.get_thick_loss(lens_system)
        # na_loss = torch.tensor(0.0)
        refraction_loss = torch.tensor(0.0)
        f_num_loss, f_num_pre_list = torch.tensor(0.0), torch.tensor(0.0)

        na_loss = self.get_na_loss(sys_param, rays_list, sins_list)
        refraction_loss = self.get_reverse_refraction_loss(intersections_tensor)
        # f_num_loss, f_num_pre_list = self.get_f_num_loss(sys_param, f)
        all_loss = (
            RMS_loss * 15
            + sins_loss
            + thick_loss
            + na_loss * 10
            + refraction_loss
            # + f_num_loss
        )
        if torch.isnan(all_loss):
            a = 1
        return {
            "all_loss": all_loss,
            "RMS_loss": RMS_loss,
            "sins_loss": sins_loss,
            "thick_loss": thick_loss,
            "na_loss": na_loss,
            "refraction_loss": refraction_loss,
            "f_num_loss": f_num_loss,
            "rays_list": rays_list,
            "sins_list": sins_list,
            "surfaces_list": surfaces_list,
            #
            "f_num_pre": f_num_pre_list,
            "sys_param_transed": sys_param_transed_list,
        }

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
            RMS_loss = torch.std(y_tensor, 1, unbiased=False)
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
            sins_loss = first_breaker**2
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
            n = len(thicks_fail)
            thicks_fail = torch.abs(thicks_fail) - 0
            thick_loss = thicks_fail.dot(thicks_fail) / n
            thick_loss_list.append(thick_loss)
        y_loss = torch.tensor(0.0)
        if len(thick_loss_list) > 0:
            thick_loss_list_torch = torch.mean(torch.stack(thick_loss_list))
            y_loss = thick_loss_list_torch
        return y_loss

    def get_na_loss(self, sys_param, rays_list, sins_list):
        bs = len(sys_param)
        na_loss_list = []
        for i in range(bs):
            rays = rays_list[i]
            sins = sins_list[i]
            sys = sys_param[i]
            if len(sins) > 0:
                continue

            final_lights = rays[-1][0][-2:]
            for j in final_lights:
                na_loss = (
                    (
                        torch.abs(torch.sin(j.u))
                        - (
                            sys[2] * TrainCfg().input_scale[2][0]
                            + TrainCfg().input_scale[0][1]
                        )
                    )
                    / (torch.abs(torch.sin(j.u)) + EPSILON)
                ) ** 2
                na_loss_list.append(na_loss)

        y_loss = torch.tensor(0.0)
        if len(na_loss_list) > 0:
            na_loss_list_torch = torch.mean(torch.stack(na_loss_list))
            y_loss = na_loss_list_torch
        return y_loss

    def get_reverse_refraction_loss(self, intersections_tensor):
        intersections = intersections_tensor
        cur_intersections = intersections[:, :-1, ...]
        next_intersections = intersections[:, 1:, ...]
        delta = next_intersections - cur_intersections
        picks = delta[torch.where(delta < 0)] - 1

        y_loss = torch.tensor(0.0)
        if picks.numel() > 0:
            y_loss = picks.dot(picks) / len(picks)

        return y_loss

    def get_f_num_loss(self, sys_param, f):
        bs = len(sys_param)
        f_num_list = []
        f_num_pre_list = []
        for i in range(bs):
            f_num_tar = (
                sys_param[i][3] * TrainCfg().input_scale[3][0]
                + TrainCfg().input_scale[3][1]
            )
            epd = (
                sys_param[i][0] * TrainCfg().input_scale[0][0]
                + TrainCfg().input_scale[0][1]
            )
            f_num_pre = epd / f[i]
            f_num_pre_list.append(f_num_pre)
            if torch.isnan(f_num_pre) == False:
                delta = torch.abs(f_num_tar - f_num_pre)
                f_num_list.append(delta)
        y_loss = torch.tensor(0.0)
        if len(f_num_list) > 0:
            thick_loss_list_torch = torch.mean(torch.stack(f_num_list))
            y_loss = thick_loss_list_torch
        return y_loss, f_num_pre_list
