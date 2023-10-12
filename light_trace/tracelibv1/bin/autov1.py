from light_trace.tracelibv1.light import LightModule
from light_trace.tracelibv1.surface import OpticalSystemModule
from light_trace.tracelibv1.utils.common import *
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.getcwd())
sys.dont_write_bytecode = True


def run():
    osm = OpticalSystemModule()
    # add surface
    sequence = [
        [133,   50,    [1, 1],  1.5],
        [-100,  60,    [1, 1],  1.9],
        [np.inf, 100,   [0, 1],  1],
        [100,   50,    [1, 1],  1.5],
        [-100,  60,    [1, 0],  1],
        ["end"],
    ]
    for i in sequence:
        if len(i) != 1:
            osm.add_surface(r=i[0], t=i[1], v=i[2], n=i[3])
        else:
            osm.add_surface(r=np.inf, t=np.inf, v=[0, 0], n=np.inf)

    osm.set_system_param(stop=0, EPD=10, field=5)

    def loss(self,RMS):
        y = 0
        return y 

if __name__ == "__main__":
    run()
