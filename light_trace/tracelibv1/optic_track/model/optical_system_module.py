import torch
import torch.nn as nn
import numpy as np

class SurfaceModule(nn.Module):
    def __init__(self, r=np.inf, t=np.inf, v=[], n=1,h=np.inf):
        super(SurfaceModule, self).__init__()

        self.r = nn.Parameter(torch.tensor(r,dtype=float))
        self.t = nn.Parameter(torch.tensor(t,dtype=float))
        self.r.requires_grad = v[0]
        self.t.requires_grad = v[1]

        self.h = torch.tensor(h,dtype=float)
        self.n = torch.tensor(n,dtype=float)
        self.z = torch.tensor(0,dtype=float)

    def forward_track(self,):
        pass

    def reverse_track(self,):
        pass

class LightModule(nn.Module):
    def __init__(self, q = 10, u = 0, p = 0, c = "r"):
        super(LightModule, self).__init__()
        self.q = torch.tensor(q,dtype=float)
        self.u = torch.tensor(u,dtype=float)
        self.p = torch.tensor(p,dtype=float)
        self.c = "r"

    def forward(self, light):
        pass

class OpticalSystemModule(nn.Module):
    def __init__(self,):
        super(OpticalSystemModule, self).__init__()
        self.surfaces = nn.ModuleList()
        self.stop_position

    def set_system_param(self, EPD, field, stop_position=None, stop_face=None, f=None):
        self.EPD = EPD
        self.field = field
        self.stop_position = stop_position
        self.stop_face = stop_face
        if stop_position is None and stop_face is None:
            self.stop_position = 0
        self.f = f

    def add_surface(self, r=np.inf, t=np.inf, v=[], n=1, h=np.inf):
        v = list(map(bool,v))
        self.surfaces.append(SurfaceModule(r,t,v,n,h))

    def flash_surface_z_potion(self, ):
        last_positon = torch.tensor(0,dtype=float)
        for i in range(len(self.surfaces)-1):
            self.surfaces[i+1].z = last_positon + self.surfaces[i].h
            last_positon = self.surfaces[i+1].z

    def get_cur_entrance_puplil_position(self):
        if self.stop_face is not None:
            self.stop_position = self.surfaces[self.stop_face].z
        return self.stop_position
    
    def build_reverse_light(self, cur_stop_position):
        u = 3/180*np.pi
        c = "g"
        p = cur_stop_position
        lights = nn.ModuleList()
        lights.append(LightModule(q=0,  u=u, p=p, c=c))
        lights.append(LightModule(q=0,  u=-u, p=p, c=c))
        return lights

    def build_forward_light(self,):
        pass

    def forward_track(self,):
        pass

    def reverse_track(self,reverser_lights):
        last_n, last_t, sum_t = 1, 0, 0
        all_lights = []
        all_lights.append(reverser_lights)
        p = reverser_lights[0].p
        
        for i, surface in enumerate(surfaces):
            if surface.r != "inf":
                c, r, t, n_1 = 1 / surface.r, surface.r, surface.t, surface.n
                out_lights = []
                for light in all_lights[i]:
                    u, color = light.u, light.c
                    q = light.q + sin(u) * last_t
                    sinI = q * c + sin(u)
                    sinI_1 = last_n * sinI / n_1
                    u_1 = u - asin(sinI) + asin(sinI_1)
                    q_1 = r * (sinI_1 - sin(u_1))
                    out_lights.append(light_lib(q=q_1, u=u_1, p=sum_t, c=color))
            else:
                r, t, n_1 = surface.r, surface.t, surface.n 
                out_lights = []
                for light in all_lights[i]:
                    u, color = light.u, light.c
                    q = light.q + sin(u) * last_t
                    sinI = sin(u)
                    I_1 = asin(last_n*sinI/n_1)
                    u_1 = I_1
                    if u == 0:
                        q_1=0
                    else:
                        q_1 = sin(u_1)*tan(u)*q/sin(u)/tan(u_1)
                    out_lights.append(light_lib(q=q_1, u=u_1, p=sum_t, c=color))
            if t != "inf":
                sum_t = sum_t + t
            last_n, last_t = surface.n, surface.t
            all_lights.append(out_lights)

    def forward(self):
        self.flash_surface_z_potion()
        cur_stop_position = self.get_cur_entrance_puplil_position()
        reverser_lights = self.build_reverse_light(cur_stop_position)
        cur_EPD_postion = self.reverse_track(reverser_light)
        forward_light = self.build_forward_light(cur_EPD_postion)
        light_trace = self.forward_track(forward_light)

        self.light_trace = light_trace
        return light_trace


if __name__=="__main__":
    os = OpticalSystemModule()
    os.add(100,100,[True,True],1,50)
    
    a = 1