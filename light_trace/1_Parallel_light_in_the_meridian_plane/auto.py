import os 
import sys
sys.path.append(os.getcwd())

import math

from tracelib.surface import surface_lib
surfaces = [
    surface_lib(r=100, h=50, t=50, n=1.5168),
    surface_lib(r='inf',h=50, t=160, n=1),
    surface_lib(r='inf',h=50, t='inf', n=1)
]

from tracelib.light import light_lib
lights = [
    light_lib(q=10, u=0),
    light_lib(q=0,  u=0),
    light_lib(q=-10,u=0),
]

last_n, last_t, sum_t = 1, 0, 0
last_lights = lights
for i,surface in enumerate(surfaces):
    c, r, t, n_1 = 1/surface.r, surface.r, surface.t, surface.n
    out_lights = []
    for light in last_lights:
        u = light.u
        q = light.q + math.sin(u)*last_t
        sinI = q*c + math.sin(u)
        sinI_1 = last_n * sinI / n_1
        u_1 = u - math.asin(sinI) + math.asin(sinI_1)
        q_1 = r * (sinI_1 - math.sin(u_1))
        out_lights.append(light_lib(q=q_1, u=u_1, p=sum_t, idx=i+1))
    if t != 'inf':
        sum_t = sum_t + t

    last_n, last_t = surface.n, surface.t
    last_lights = out_lights
    for l in out_lights:
        print(f"[p {l.p:6}], [q {l.q:8.3f}], [u {l.u:8.3f}], [idx {l.idx:4}]")
    print("finish one surface")

print("final check")
for l in last_lights:
    print(f"[p {l.p:6}], [q {l.q:8.3f}], [u {l.u:8.3f}]")
a = 1