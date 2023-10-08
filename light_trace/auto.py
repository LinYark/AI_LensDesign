import os
import sys

sys.path.append(os.getcwd())
sys.dont_write_bytecode = True

import numpy as np
import matplotlib.pyplot as plt
import torch

from light_trace.tracelib.common import *
from light_trace.tracelib.surface import surface_lib
from light_trace.tracelib.light import light_lib

surfaces = [
    surface_lib(r=133, h=40, t=50, n=1.5168),
    surface_lib(r=-100, h=40, t=60, n=1.9),
    surface_lib(r="inf", h=40, t=100, n=1),
    surface_lib(r=100, h=50, t=50, n=1.5168),
    surface_lib(r=-100, h=50, t=60, n=1),
    surface_lib(r="inf", h=20, t="inf"),
]

u = 0/180*np.pi
c = "b"
p = 0
lights1 = [
    light_lib(q=30,  u=u, p=p, c=c),
    light_lib(q=20,  u=u, p=p, c=c),
    light_lib(q=10,  u=u, p=p, c=c),
    light_lib(q=0,   u=u, p=p, c=c),
    light_lib(q=-10, u=u, p=p, c=c),
    light_lib(q=-20, u=u, p=p, c=c),
    light_lib(q=-30, u=u, p=p, c=c),
]
u = 5/180*np.pi
c = "g"
p = 0
lights = [
    light_lib(q=30,  u=u, p=p, c=c),
    light_lib(q=20,  u=u, p=p, c=c),
    light_lib(q=10,  u=u, p=p, c=c),
    light_lib(q=0,   u=u, p=p, c=c),
    light_lib(q=-10, u=u, p=p, c=c),
    light_lib(q=-20, u=u, p=p, c=c),
    light_lib(q=-30, u=u, p=p, c=c),
]
light=lights.extend(lights1)
# trace
last_n, last_t, sum_t = 1, 0, 0
all_lights = []
all_lights.append(lights)
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

# print("final check")
# for i, lights in enumerate(all_lights):
#     for l in lights:
#         print(f"[{i:2}, p q u], [{l.p:5} {l.q:8.3f} {l.u:8.3f}]")
#     print()


# draw_lights
def get_cross_point(all_lights):
    all_cross_points = []

    input_lights = all_lights[0]
    cross_points = []
    for l in input_lights:
        if l.u != 0:
            z = -100
            y = tan(l.u) * (z - l.p + l.q / (sin(l.u)))
        else:
            z = -100
            y = l.q
        cross_points.append([z, y, l.c])
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

    return all_cross_points

all_cross_points = get_cross_point(all_lights)
points = np.array(all_cross_points)
_,num_lights,_ = points.shape
for i in range(num_lights):
    z= np.array(points[:,i,0],dtype='float64')
    y = np.array(points[:,i,1],dtype='float64')
    c = points[0,i,2]
    plt.plot(z,y,color=c) #color='b',linestyle = 'dotted',

def get_surface_point(surfaces):
    all_surface_points=[]
    cur_p = 0
    for s in surfaces:
        if s.r != "inf":
            center = cur_p+s.r
            if s.r>0:
                theta = asin(s.h/s.r)
                thetas = np.append(np.arange(np.pi-theta, np.pi+theta, 0.1), np.pi+theta)
                thetas = np.flip(thetas) 
            else:
                theta = asin(s.h/s.r)
                thetas = np.append(np.arange(theta, -theta, 0.1), -theta)
            z = center + abs(s.r) * np.cos(thetas)
            y = 0 + abs(s.r) * np.sin(thetas)
        else:
            y = np.arange(-s.h, s.h, 0.1)
            z = np.full_like(y,cur_p)
        all_surface_points.append([z,y])
        if s.t != "inf":
            cur_p += s.t
    all_edge_points = []
    for i,s in enumerate(surfaces):
        if s.n != 1:
            z = np.arange(all_surface_points[i][0][0], all_surface_points[i+1][0][0], 0.1)
            y = np.full_like(z,s.h)
            all_edge_points.append([z,y])
            y = np.full_like(z,-s.h)
            all_edge_points.append([z,y])
    return all_surface_points,all_edge_points
all_surface_points,all_edge_points = get_surface_point(surfaces)

for points in all_surface_points:
    plt.plot(points[0],points[1],color='black') 
for points in all_edge_points:
    plt.plot(points[0],points[1],color='black') 

plt.show()

a = 1
