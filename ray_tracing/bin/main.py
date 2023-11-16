import os 
import sys
sys.path.append(os.getcwd())
import time
import torch
import torch.nn as nn
from ray_tracing.optic_track.model.optical_system import OpticalSystemModule
from ray_tracing.optic_track.loss.loss import OpticalLoss
from ray_tracing.optic_track.model.drawer import OpticalSystemDrawer
from ray_tracing.optic_track.config.cfg_1027 import config


if __name__=="__main__":

    osm = OpticalSystemModule()
    for surface in config["surfaces"]:
        osm.add_surface(*surface)

    osm.set_system_param(40,5,stop_face=2)
    # osm.cuda()
    draw_flag = 1
    osd = OpticalSystemDrawer(draw_flag)
    osd.set_start_z(-100)
    optim = torch.optim.Adam(osm.parameters(), lr=1e-5,betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
    optical_loss = OpticalLoss()
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optim,base_lr=1e-6, max_lr=1e-4, step_size_up=10, mode="triangular",cycle_momentum=False)

    first_tick = time.time()
    tick, tock = first_tick, first_tick
    flash_hz = 100
    stop  = 1
    for epoch in range(99999):
        iter = 100
        scheduler.step()
        for i in range(iter):
            i = epoch*iter + i

            light_trace = osm()
            if i % flash_hz ==0:
                print("============================================")
                osd.set_surfaces(osm.get_surface())
                osd.set_lights(light_trace)
                osd.draw(draw_flag)
                osd.print_surface(osm.get_surface())
            loss, u = optical_loss.get_RMS_loss(osm.get_surface(),light_trace)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(osm.parameters(), max_norm=20, norm_type=2)
            optim.step()
            if i % flash_hz ==0:
                tock = tick
                tick = time.time()
                print(f"[i loss c_t a_t], [{i:<8} {loss.item():<8.6f} {tick-tock:<8.2f} {tick-first_tick:<8.2f}  {u:<8.2f}]") #i loss cost_time all_time 
            if stop == 1:
                time.sleep(5)
                stop = 0
    a= 1