from ..utils.common import *
import torch

class OpticalLoss():
    def __init__(self) -> None:
        pass

    def get_RMS_loss(self,surface,light):
        z = surface[-1].z
        final_light = light[-1]
        # final_light = [final_light[0]]

        y = []
        for angle_lights in final_light:
            angle_ys = []
            
            for single_light in angle_lights:
                if torch.abs(single_light.u) < EPSILON:
                    single_y = single_light.q
                else:
                    single_y = torch.tan(single_light.u)*(z - single_light.p +single_light.q/torch.sin(single_light.u))
                angle_ys.append(single_y)

            angle_y_tensors = torch.stack(angle_ys)
            y.append(angle_y_tensors)
        
        y_tensor = torch.stack(y)
        y_var = torch.std(y_tensor,1)
        RMS_loss = y_var.dot(y_var)

        t_loss = []
        air_thick_thr, glass_thick_thr = 100,20
        for single_face in surface:
            t, n = 1/single_face.t, single_face.n
            if t>air_thick_thr and t != torch.inf and n==1:
                t_loss.append((t-air_thick_thr)**2)
            if t>glass_thick_thr and t != torch.inf and n!=1:
                t_loss.append((t-glass_thick_thr)**2)
        if len(t_loss)>0:
            t_tensors = torch.stack(t_loss)
            t_loss = t_tensors.dot(t_tensors)
        else:
            t_loss = torch.tensor(0.)
        # u, u_1 = final_light[0][-2].u, final_light[0][-1].u
        # y_u = torch.abs(torch.abs(u)-0.42)
        # y_u1 = torch.abs(torch.abs(u_1)-0.42)
        # + y_u*5 + y_u1*5
        y_loss = RMS_loss  + t_loss
        # print(f"u_loss,{final_light[0][-2].u:8.2f}")
        return y_loss , 0 #, u


