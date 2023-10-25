from ..utils.common import *
import torch

class OpticalLoss():
    def __init__(self) -> None:
        pass

    def get_RMS_loss(self,surface,light):
        z = surface[-1].z
        final_light = light[-1]

        y = []
        for angle_lights in final_light:
            angle_ys = []
            
            for single_light in angle_lights:
                if single_light.u < EPSILON:
                    single_y = single_light.q
                else:
                    single_y = torch.tan(single_light.u)*(z - single_light.p +single_light.q/torch.sin(single_light.u))
                    angle_ys.append(single_y)

            angle_y_tensors = torch.stack(angle_ys)
            y.append(angle_y_tensors)
        
        y_tensor = torch.stack(y)
        y_loss = torch.mean(torch.var(y_tensor,1))
        return y_loss


