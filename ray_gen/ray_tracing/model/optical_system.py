import torch
import torch.nn as nn
import numpy as np


EPSILON = 1e-9
from .surface import SurfaceModule
from .light import LightModule


class OpticalSystemModule(nn.Module):
    def __init__(self):
        super(OpticalSystemModule, self).__init__()
        self.surfaces = nn.ModuleList()

    def set_system_param(self, EPD, field, stop_position=None, stop_face=None, f=None):
        self.EPD = torch.tensor(EPD, dtype=float)
        self.field = torch.tensor(field / 180 * torch.pi, dtype=float)
        self.stop_position = stop_position
        self.stop_face = stop_face
        if stop_position is None and stop_face is None:
            self.stop_position = 0
        self.f = f

    def add_surface(self, surface):
        c, t, n, h = surface
        self.surfaces.append(SurfaceModule(c, t, n, h))

    def get_surface(self):
        return self.surfaces

    def flash_surface_z_potion(self):
        last_positon = torch.tensor(0, dtype=float)
        for i in range(len(self.surfaces) - 1):
            self.surfaces[i + 1].z = last_positon + 1 / self.surfaces[i].t
            last_positon = self.surfaces[i + 1].z

    def get_cur_entrance_puplil_position(self):
        if self.stop_face is not None:
            self.stop_position = self.surfaces[self.stop_face].z
        return self.stop_position

    def build_reverse_light(self, cur_stop_position):
        u = 0.1 / 180 * torch.pi
        p = cur_stop_position
        c = "g"
        lights = []
        lights.append(LightModule(q=0, u=u, p=p, c=c))
        lights.append(LightModule(q=0, u=-u, p=p, c=c))
        return lights

    def build_forward_light(self, cur_EPD_postion):
        forward_light = []
        step = self.EPD / 6.0
        # field = center
        field_center_light = []
        u = 0 / 180 * torch.pi
        p = 0
        c = "b"
        for i in range(4):
            if i == 0:
                field_center_light.append(LightModule(q=0, u=u, p=p, c=c))
            else:
                field_center_light.append(LightModule(q=step * i, u=u, p=p, c=c))
                field_center_light.append(LightModule(q=-step * i, u=u, p=p, c=c))
        # field = edge
        field_edge_light = []
        u = self.field
        p = 0
        c = "g"
        q_0 = -cur_EPD_postion * torch.sin(u)
        q_step = step * torch.cos(u)
        for i in range(4):
            if i == 0:
                field_edge_light.append(LightModule(q=q_0, u=u, p=p, c=c))
            else:
                field_edge_light.append(LightModule(q=q_0 + q_step * i, u=u, p=p, c=c))
                field_edge_light.append(LightModule(q=q_0 - q_step * i, u=u, p=p, c=c))

        forward_light.append(field_center_light)
        forward_light.append(field_edge_light)
        return forward_light

    def forward_track(self, forward_light):
        all_lights = []
        all_lights.append(forward_light)
        for i, surface in enumerate(self.surfaces):
            c, t_1, n_1, z = surface.c, 1 / surface.t, surface.n, surface.z
            if c == 0:
                c = c + EPSILON
            if i > 0:
                n, t = self.surfaces[i - 1].n, 1 / self.surfaces[i - 1].t
            else:
                n, t = 1, 0
            if c != 0:
                out_lights = []
                for single_angle_lights in all_lights[i]:
                    angle_lights = []
                    for light in single_angle_lights:
                        u, color = light.u, light.c
                        q = light.q + torch.sin(u) * t
                        sinI = q * c + torch.sin(u)
                        sinI_1 = n * sinI / n_1
                        u_1 = u - torch.asin(sinI) + torch.asin(sinI_1)
                        q_1 = (sinI_1 - torch.sin(u_1)) / c
                        angle_lights.append(LightModule(q=q_1, u=u_1, p=z, c=color))
                    out_lights.append(angle_lights)
            else:
                out_lights = []
                for single_angle_lights in all_lights[i]:
                    angle_lights = []
                    for light in single_angle_lights:
                        u, color = light.u, light.c
                        q = light.q + torch.sin(u) * t
                        sinI = torch.sin(u)
                        I_1 = torch.asin(n * sinI / n_1)
                        u_1 = I_1
                        if u == 0:
                            q_1 = 0
                        else:
                            q_1 = torch.cos(u_1) * q / torch.cos(u)
                        if torch.isnan(u_1):
                            a = 1
                        angle_lights.append(LightModule(q=q_1, u=u_1, p=z, c=color))
                    out_lights.append(angle_lights)
            all_lights.append(out_lights)
        return all_lights

    def reverse_track(self, reverse_lights, cur_stop_position):
        front_material = 1
        for i, surface in enumerate(self.surfaces):
            if surface.z >= cur_stop_position:
                break
            else:
                front_material += 1

        reverse_light_counts = front_material
        all_lights = [None] * (reverse_light_counts)
        all_lights[reverse_light_counts - 1] = reverse_lights

        p = reverse_lights[0].p
        reverse_surface_count = reverse_light_counts - 1
        for idx in range(reverse_surface_count):
            surface_1_idx = reverse_surface_count - 1 - idx
            surface_1 = self.surfaces[surface_1_idx]
            c, t, n_1, z = surface_1.c, 1 / surface_1.t, surface_1.n, surface_1.z
            if c == 0:
                c = c + EPSILON
            if surface_1_idx > 0:
                surface = self.surfaces[surface_1_idx - 1]
                n = surface.n
            else:
                n = 1
            out_lights = []
            for light in all_lights[surface_1_idx + 1]:
                q_2, u_1, p_1, color = light.q, light.u, light.p, light.c
                q_1 = q_2 - torch.sin(u_1) * (p_1 - z)
                if c != 0:
                    sinI_1 = c * q_1 + torch.sin(u_1)
                    I_1 = torch.asin(sinI_1)
                    sinI = n_1 * sinI_1 / n
                    I = torch.asin(sinI)
                    u = u_1 + I - I_1
                    q = (sinI - torch.sin(u)) / c
                    out_lights.append(LightModule(q=q, u=u, p=z, c=color))
                else:
                    u = torch.asin(torch.sin(u_1) * n_1 / n)
                    q = q_1 / torch.cos(u_1) * torch.cos(u)
                    out_lights.append(LightModule(q=q, u=u, p=z, c=color))
            all_lights[surface_1_idx] = out_lights
        final_lights = all_lights[0]
        light_counts = len(final_lights)
        z_candidates = []
        for i in range(int(light_counts / 2)):
            l1 = final_lights[i]
            l2 = final_lights[light_counts - i - 1]
            z = (
                torch.tan(l1.u) * (l1.p - l1.q / torch.sin(l1.u))
                - torch.tan(l2.u) * (l2.p - l2.q / torch.sin(l2.u))
            ) / (torch.tan(l1.u) - torch.tan(l2.u))
            y = torch.tan(l2.u) * (z - l2.p + l2.q / torch.sin(l2.u))
            z_candidates.append(z)
        z_c = torch.stack(z_candidates)
        cur_EPD_postion = torch.mean(z_c)
        return cur_EPD_postion

    def forward(self):
        self.flash_surface_z_potion()
        cur_stop_position = self.get_cur_entrance_puplil_position()
        reverser_lights = self.build_reverse_light(cur_stop_position)
        cur_EPD_position = self.reverse_track(reverser_lights, cur_stop_position)
        forward_light = self.build_forward_light(cur_EPD_position)
        light_trace = self.forward_track(forward_light)

        self.light_trace = light_trace
        return light_trace
