import os 
import sys
sys.path.append(os.getcwd())

import math

from tracelib.surface import surface_lib
surfaces = [
    surface_lib(radius=100, h=50, t=50, material=1.5),
    surface_lib(radius='inf',h=50, t=200, material=1),
    surface_lib(t='inf')
]

from tracelib.light import light_lib
lights = [
    light_lib(q=10, u=0),
    light_lib(q=0,  u=0),
    light_lib(q=-10,u=0),
]

last_n, last_t = 1, 0
last_lights = lights
for i,surface in enumerate(surfaces):
    out_lights = []
    for light in last_lights:
        q = light.q + math.sin(u)*last_t,
        u, n_1=  light.u, surface.n
        c, r = 1/surface.r, surface.r

        sinI = q*c + math.sin(u)
        sinI_1 = last_n * sinI / n_1
        u_1 = u - math.asin(sinI) + math.asin(sinI_1)
        q_1 = r * (sinI_1 - math.sin(u_1))
        out_lights.append(light_lib(q=q_1, u=u_1))

    last_n, last_t = surface.n, surface.t
    last_lights = out_lights