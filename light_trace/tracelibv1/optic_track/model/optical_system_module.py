import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
from light_trace.tracelibv1.utils.common import *
class LightModule(nn.Module):
    def __init__(self, q = 10, u = 0, p = 0, c = "r"):
        super(LightModule, self).__init__()
        if torch.is_tensor(q):
            self.q = q
        else:
            self.q = torch.tensor(q,dtype=float)
        if torch.is_tensor(u):
            self.u = u
        else:
            self.u = torch.tensor(u,dtype=float)
        if torch.is_tensor(p):
            self.p = p
        else:
            self.p = torch.tensor(p,dtype=float)
        self.c = c

    def forward(self, light):
        pass

class SurfaceModule(nn.Module):
    def __init__(self, r=torch.inf, t=torch.inf, v=[], n=1,h=torch.inf):
        super(SurfaceModule, self).__init__()
        self.r = nn.Parameter(torch.tensor(r,dtype=float))
        self.t = nn.Parameter(torch.tensor(t,dtype=float))
        self.r.requires_grad = v[0]
        self.t.requires_grad = v[1]

        self.h = torch.tensor(h,dtype=float)
        self.n = torch.tensor(n,dtype=float)
        self.z = torch.tensor(0,dtype=float,requires_grad=True)

    def forward_track(self,):
        pass

    def reverse_track(self,):
        pass


class OpticalSystemModule(nn.Module):
    def __init__(self,):
        super(OpticalSystemModule, self).__init__()
        self.surfaces = nn.ModuleList()

    def set_system_param(self, EPD, field, stop_position=None, stop_face=None, f=None):
        self.EPD = torch.tensor(EPD,dtype=float)
        self.field = torch.tensor(field/180*torch.pi,dtype=float)
        self.stop_position = stop_position
        self.stop_face = stop_face
        if stop_position is None and stop_face is None:
            self.stop_position = 0
        self.f = f

    def add_surface(self, r=torch.inf, t=torch.inf, v=[], n=1, h=torch.inf):
        v = list(map(bool,v))
        self.surfaces.append(SurfaceModule(r,t,v,n,h))

    def get_surface(self,):
        return self.surfaces

    def flash_surface_z_potion(self, ):
        last_positon = torch.tensor(0,dtype=float)
        for i in range(len(self.surfaces)-1):
            self.surfaces[i+1].z = last_positon + self.surfaces[i].t
            last_positon = self.surfaces[i+1].z

    def get_cur_entrance_puplil_position(self):
        if self.stop_face is not None:
            self.stop_position = self.surfaces[self.stop_face].z
        return self.stop_position
    
    def build_reverse_light(self, cur_stop_position):
        u = 0.1/180*torch.pi
        p = cur_stop_position
        c = "g"
        lights = []
        lights.append(LightModule(q=0,  u=u, p=p, c=c))
        lights.append(LightModule(q=0,  u=-u, p=p, c=c))
        return lights

    def build_forward_light(self, cur_EPD_postion):
        forward_light = []
        step = self.EPD /6.0
        # field = center
        field_center_light = []
        u = 0/180*torch.pi
        p = 0
        c = "b"
        for i in range(4):
            if i== 0:
                field_center_light.append(LightModule(q=0, u=u, p=p, c=c))
            else:
                field_center_light.append(LightModule(q= step*i, u=u, p=p, c=c))
                field_center_light.append(LightModule(q=-step*i, u=u, p=p, c=c))
        # field = edge
        field_edge_light = []
        u = self.field
        p = 0
        c = "g"
        q_0 = cur_EPD_postion*torch.sin(u)
        q_step = step*torch.cos(u)
        for i in range(4):
            if i== 0:
                field_edge_light.append(LightModule(q=q_0,  u=u, p=p, c=c))
            else:
                field_edge_light.append(LightModule(q= q_0+q_step*i, u=u, p=p, c=c))
                field_edge_light.append(LightModule(q= q_0-q_step*i, u=u, p=p, c=c))

        forward_light.append(field_center_light)
        forward_light.append(field_edge_light)
        return forward_light

    def forward_track(self,forward_light):
        all_lights = []
        all_lights.append(forward_light)
        for i, surface in enumerate(self.surfaces):
            c, r, t_1, n_1, z = 1 / surface.r, surface.r, surface.t, surface.n, surface.z
            if i > 0:
                n, t = self.surfaces[i-1].n, self.surfaces[i-1].t
            else:
                n, t = 1, 0

            if surface.r != torch.inf:
                out_lights = []
                for single_angle_lights in all_lights[i]:
                    angle_lights = []
                    for light in single_angle_lights:
                        u, color = light.u, light.c
                        q = light.q + torch.sin(u) * t
                        sinI = q * c + torch.sin(u)
                        sinI_1 = n * sinI / n_1
                        u_1 = u - torch.asin(sinI) + torch.asin(sinI_1)
                        q_1 = r * (sinI_1 - torch.sin(u_1))
                        angle_lights.append(LightModule(q=q_1, u=u_1, p=z, c=color))
                    out_lights.append(angle_lights)
            else:
                r, t_1, n_1 = surface.r, surface.t, surface.n 
                out_lights = []
                for single_angle_lights in all_lights[i]:
                    angle_lights = []
                    for light in single_angle_lights:
                        u, color = light.u, light.c
                        q = light.q + sin(u) * t
                        sinI = torch.sin(u)
                        I_1 = torch.asin(n*sinI/n_1)
                        u_1 = I_1
                        if u == 0:
                            q_1=0
                        else:
                            q_1 = torch.cos(u_1)*q/torch.cos(u)
                        angle_lights.append(LightModule(q=q_1, u=u_1, p=z, c=color))
                    out_lights.append(angle_lights)
            all_lights.append(out_lights)
        return all_lights

    def reverse_track(self,reverse_lights,cur_stop_position):
        front_material = 1
        for i,surface in enumerate(self.surfaces):
            if surface.z>=cur_stop_position:
                break
            else:
                front_material += 1

        reverse_light_counts = front_material
        all_lights = [None]*(reverse_light_counts)
        all_lights[reverse_light_counts-1] = reverse_lights

        p = reverse_lights[0].p
        reverse_surface_count = reverse_light_counts - 1
        for idx in range(reverse_surface_count):
            surface_1_idx = reverse_surface_count-1 - idx
            surface_1 = self.surfaces[surface_1_idx]
            c, r, t, n_1, z = 1 / surface_1.r, surface_1.r, surface_1.t, surface_1.n, surface_1.z
            if surface_1_idx >0:
                surface = self.surfaces[surface_1_idx-1]
                n = surface.n
            else:
                n = 1
            out_lights = []
            for light in all_lights[i]:
                q_2, u_1, p_1, color = light.q, light.u, light.p, light.c
                if surface_1.r != torch.inf:
                    q_1 = q_2 - torch.sin(u_1) * (p_1-surface_1.z)
                    sinI_1 = c*q_1+torch.sin(u_1)
                    I_1 = torch.asin(sinI_1)
                    sinI = n_1*sinI_1/n
                    I = torch.asin(sinI)
                    u = u_1+I-I_1
                    q = (sinI-torch.sin(u))*r
                    out_lights.append(LightModule(q=q, u=u, p=z, c=color))
                else:
                    u = torch.asin(torch.sin(u_1)*n_1/n)
                    q = q_1/torch.cos(u_1)*torch.cos(u)
                    out_lights.append(LightModule(q=q, u=u, p=z, c=color))
            all_lights[surface_1_idx] = out_lights
        final_lights = all_lights[0]
        light_counts = len(final_lights)
        z_candidates = []
        for i in range(int(light_counts/2)):
            l1 = final_lights[i]
            l2 = final_lights[light_counts-i-1]
            z = (tan(l1.u) * (l1.p - l1.q / sin(l1.u))
                - tan(l2.u) * (l2.p - l2.q / sin(l2.u))
                ) / (tan(l1.u) - tan(l2.u))
            y = tan(l2.u) * (z - l2.p + l2.q / sin(l2.u))
            z_candidates.append(z)
        z_c = torch.stack(z_candidates)
        cur_EPD_postion = torch.mean(z_c)
        return cur_EPD_postion
    
    def forward(self):
        self.flash_surface_z_potion()
        cur_stop_position = self.get_cur_entrance_puplil_position()
        reverser_lights = self.build_reverse_light(cur_stop_position)
        cur_EPD_postion = self.reverse_track(reverser_lights,cur_stop_position)
        forward_light = self.build_forward_light(cur_EPD_postion)
        light_trace = self.forward_track(forward_light)

        self.light_trace = light_trace
        return light_trace

class OpticalSystemDrawer():
    def __init__(self) -> None:
        pass

    def set_lights(self,lights):
        self.lights = lights

    def set_surfaces(self,surfaces):
        self.surfaces = surfaces

    def draw():
        pass

    def torch2numpy(self,t):
        return t.detach().cpu().numpy()

    def draw_surfaces(self,):
        all_surface_points=[]
        for s in self.surfaces:
            r, h, z  = self.torch2numpy(s.r), self.torch2numpy(s.h), self.torch2numpy(s.z) 
            if r != np.inf:
                center = z + r
                if r>0:
                    theta = np.arcsin(h/r)
                    thetas = np.linspace(np.pi-theta, np.pi+theta, 1000)
                else:
                    theta = np.arcsin(h/r)
                    thetas = np.linspace(theta, -theta, 1000)
                z = center + np.abs(r) * np.cos(thetas)
                y = np.abs(r) * np.sin(thetas)
            else:
                y = np.linspace(h, -h, 1000)
                z = np.full_like(y,z)
            all_surface_points.append([z,y])

        all_edge_points = []
        for i,s in enumerate(self.surfaces):
            r, n, h, z  = self.torch2numpy(s.r), self.torch2numpy(s.n), self.torch2numpy(s.h), self.torch2numpy(s.z) 
            if n != 1:
                z = np.linspace(all_surface_points[i][0][0], all_surface_points[i+1][0][0], 1000)
                y = np.full_like(z, h)
                all_edge_points.append([z,y])
                y = np.full_like(z, -h)
                all_edge_points.append([z,y])

        for points in all_surface_points:
            plt.plot(points[0],points[1],color='black') 
        for points in all_edge_points:
            plt.plot(points[0],points[1],color='black') 
        plt.show()
        return all_surface_points,all_edge_points

    def draw_lights(self,):
        all_cross_points = []
        input_lights = self.lights[0]
        cross_points = []
        for single_angle_lights in input_lights:
            single_angle_cross_points = []
            for l in single_angle_lights:
                u, p, q = self.torch2numpy(l.u), self.torch2numpy(l.p), self.torch2numpy(l.q)
                if u != 0:
                    z = -100
                    y = np.tan(u) * (z - p + q / (np.sin(u)))
                else:
                    z = -100
                    y = q
                single_angle_cross_points.append([z, y, l.c])
            cross_points.append(single_angle_cross_points)
        all_cross_points.append(cross_points)

        all_lights = [1]
        for i in range(len(self.lights) - 2):
            lights_in, lights_out = self.lights[i], self.lights[i + 1]
            cross_points = []
            for j in range(len(lights_in)):
                l1, l2 = lights_in[j], lights_out[j]
                if l1.u != 0 and l2.u != 0 and l1.u != l2.u:
                    z = (
                        tan(l1.u) * (l1.p - l1.q / sin(l1.u))
                        - tan(l2.u) * (l2.p - l2.q / sin(l2.u))
                    ) / (tan(l1.u) - tan(l2.u))
                    y = tan(l2.u) * (z - l2.p + l2.q / sin(l2.u))
                elif l1.u == 0 and l2.u != 0:
                    z = l1.q / tan(l2.u) + l2.p - l2.q / sin(l2.u)
                    y = l1.q
                elif l1.u != 0 and l2.u == 0:
                    z = l2.q / tan(l1.u) + l1.p - l1.q / sin(l1.u)
                    y = l2.q
                elif l1.u == 0 and l2.u == 0:
                    z = l2.p
                    y = l2.q
                else:
                    z = l2.p
                    y = tan(l2.u) * (z - l2.p + l2.q / sin(l2.u))
                cross_points.append([z, y, l1.c])
            all_cross_points.append(cross_points)

        input_lights = all_lights[-1]
        cross_points = []
        for l in input_lights:
            if l.u != 0:
                z = l.p
                y = tan(l.u) * (z - l.p + l.q / (sin(l.u)))
            else:
                z = l.p
                y = l.q
            cross_points.append([z, y, l.c])
        all_cross_points.append(cross_points)



if __name__=="__main__":
    # surfaces = [
    #     surface_lib(r=200, h=100, t=50, n=1.5168),
    #     surface_lib(r=-200, h=100, t=60, n=1),
    #     surface_lib(r="inf", h=20, t="inf"),
    # ]
    osm = OpticalSystemModule()
    osm.add_surface(200,50,[True,True],1.5168,50)
    osm.add_surface(-200,60,[True,True],1,50)
    osm.add_surface(torch.inf,torch.inf,[False,False],1,30)
    osm.set_system_param(10,5,stop_face=1)
    l = osm()

    osd = OpticalSystemDrawer()
    osd.set_surfaces(osm.get_surface())
    osd.set_lights(l)

    osd.draw_surfaces()
    osd.draw_lights()
    osd.draw()

    a = 1