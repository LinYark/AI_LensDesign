import os
import sys

sys.path.append(os.getcwd())
sys.dont_write_bytecode = True

import numpy as np
import matplotlib.pyplot as plt

from light_trace.tracelib.common import *

from light_trace.tracelib.surface import surface_lib

surfaces = [
    surface_lib(r=100, h=50, t=50, n=1.1168),
    surface_lib(r=100, h=50, t=60, n=1.9),
    surface_lib(r="inf", h=50, t=160, n=1),
    surface_lib(r="inf", h=50, t="inf", n=1),
]

from light_trace.tracelib.light import light_lib

lights = [
    light_lib(q=10, u=0, p=0),
    light_lib(q=0, u=0, p=0),
    light_lib(q=-10, u=0, p=0),
]

# trace
last_n, last_t, sum_t = 1, 0, 0
all_lights = []
all_lights.append(lights)
for i, surface in enumerate(surfaces):
    c, r, t, n_1 = 1 / surface.r, surface.r, surface.t, surface.n
    out_lights = []
    for light in all_lights[i]:
        u = light.u
        q = light.q + sin(u) * last_t
        sinI = q * c + sin(u)
        sinI_1 = last_n * sinI / n_1
        u_1 = u - asin(sinI) + asin(sinI_1)
        q_1 = r * (sinI_1 - sin(u_1))
        out_lights.append(light_lib(q=q_1, u=u_1, p=sum_t))
    if t != "inf":
        sum_t = sum_t + t
    last_n, last_t = surface.n, surface.t
    all_lights.append(out_lights)

print("final check")
for i, lights in enumerate(all_lights):
    for l in lights:
        print(f"[{i:2}, p q u], [{l.p:5} {l.q:8.3f} {l.u:8.3f}]")
    print()


# draw
def get_cross_point(all_lights):
    all_cross_points = []

    input_lights = all_lights[0]
    cross_points = []
    for l in input_lights:
        if l.u != 0:
            z = 0
            y = tan(l.u) * (z - l.p + l.q / (sin(l.u)))
        else:
            z = 0
            y = l.q
        cross_points.append([z, y])
    all_cross_points.append(cross_points)

    
    for i in range(len(all_lights) - 2):
        lights_in, lights_out = all_lights[i], all_lights[i + 1]
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
            cross_points.append([z, y])
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
        cross_points.append([z, y])
    all_cross_points.append(cross_points)

    return all_cross_points

all_cross_points = get_cross_point(all_lights)
points = np.array(all_cross_points)
_,num_lights,_ = points.shape
for i in range(num_lights):
    plt.plot(points[:,i,0],points[:,i,1]) #color='b',linestyle = 'dotted',
plt.show()

a = 1
