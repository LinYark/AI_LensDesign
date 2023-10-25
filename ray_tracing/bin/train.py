import os 
import sys
sys.path.append(os.getcwd())
import time
import torch
from ray_tracing.optic_track.model.optical_system import OpticalSystemModule
from ray_tracing.optic_track.loss.loss import OpticalLoss
from ray_tracing.optic_track.model.drawer import OpticalSystemDrawer


if __name__=="__main__":

    osm = OpticalSystemModule()
    osm.add_surface(133,50,[True,False],1.5168,40)
    osm.add_surface(-100,60,[True,False],2.02204,40)
    osm.add_surface(torch.inf,100,[False,False],1,40)
    osm.add_surface(100,50,[False,False],1.5168,50)
    osm.add_surface(-100,60,[False,False],1,50)

    osm.add_surface(torch.inf,torch.inf,[False,False],1,30)
    osm.set_system_param(40,5,stop_face=2)
    # osm.cuda()
    draw_flag = 1
    osd = OpticalSystemDrawer(draw_flag)
    osd.set_start_z(-100)
    optim = torch.optim.AdamW(osm.parameters(), lr=0.1,betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    optical_loss = OpticalLoss()

    first_tick = time.time()
    tick, tock = first_tick, first_tick
    for i in range(999999):
            
        light_trace = osm()
        loss = optical_loss.get_RMS_loss(osm.get_surface(),light_trace)
        for idx in range(len(light_trace[-1][0])):
            loss += torch.abs(light_trace[-1][0][idx].q)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if i % 100 ==0:
            tock = tick
            tick = time.time()
            print(f"\n[i loss c_t a_t], [{i:<8} {loss.item():<8.4f} {tick-tock:<8.2f} {tick-first_tick:<8.2f}]") #i loss cost_time all_time 
            osd.print_surface(osm.get_surface())
        if i % 100 ==0:
            osd.set_surfaces(osm.get_surface())
            osd.set_lights(light_trace)
            osd.draw()

    a= 1