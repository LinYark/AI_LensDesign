import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import os
import datetime

EPSILON = 1e-9


class OpticalSystemDrawer:
    def __init__(self) -> None:
        self.set_start_z(-100)

    def __del__(self):
        plt.ioff()

    def set_lights(self, lights):
        self.lights = lights

    def set_surfaces(self, surfaces):
        self.surfaces = surfaces

    def set_start_z(self, z):
        self.start_z = z

    def print_surface(self, surfaces=None, epoch=0, shotpath=None):
        if surfaces is not None:
            self.surfaces = surfaces

        os.makedirs(shotpath, exist_ok=True)
        with open(f"{shotpath}/test.txt", "a") as file:
            gap = "========================"
            print(gap)
            print(gap, file=file)
            for i in self.surfaces:
                if i.c != 0:
                    content = f"{epoch} [r t h n z], [{(1/i.c).item():8.2f} {1/i.t.item():8.2f} {i.h.item():8.2f} {i.n.item():8.2f} {i.z.item():8.2f}]"
                else:
                    t = (1 / i.t).item()
                    content = f"{epoch} [r t h n z], [{np.inf:8.2f} {(1/i.t).item():8.2f} {i.h.item():8.2f} {i.n.item():8.2f} {i.z.item():8.2f}]"
                print(content, file=file)
                print(content)

    def show(self, listener, epoch, shotpath):
        bs = len(listener[0])
        bs = np.clip(bs, 0, 6)

        plt.cla()
        plt.figure(figsize=(36, 20), dpi=80)
        for idx in range(bs):
            # i, j = idx / 2, idx % j
            plt.subplot(3, 2, idx + 1)
            self.set_surfaces(listener[2][idx])
            self.set_lights(listener[0][idx])
            self.draw()
            self.print_surface(listener[2][idx], epoch, shotpath)

        # plt.get_current_fig_manager().full_screen_toggle()
        current_time = (
            str(datetime.datetime.now())
            .replace(".", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )
        target_dir = f"{shotpath}/img"
        os.makedirs(target_dir, exist_ok=True)
        target_path = f"{target_dir}/{current_time}.png"
        plt.savefig(target_path, bbox_inches="tight")

        # plt.show(block=False)
        # plt.pause(5)
        # plt.close("all")

    def draw(self):
        all_surface_points, all_edge_points = self.draw_surfaces()
        for points in all_surface_points:
            plt.plot(points[0], points[1], color="black")
        for points in all_edge_points:
            plt.plot(points[0], points[1], color="black")

        all_cross_points = np.array(self.draw_lights())
        num_face, num_angles, num_q, _ = all_cross_points.shape
        for i in range(num_angles):
            for j in range(num_q):
                z = np.array(all_cross_points[:, i, j, 0], dtype="float64")
                y = np.array(all_cross_points[:, i, j, 1], dtype="float64")
                c = all_cross_points[0, i, j, 2]
                plt.plot(z, y, color=c)

    def torch2numpy(self, t):
        if torch.is_tensor(t):
            t = t.detach().cpu().numpy()
        return t

    def draw_surfaces(self):
        all_surface_points = []
        for s in self.surfaces:
            c, h, z = (
                self.torch2numpy(s.c),
                self.torch2numpy(s.h),
                self.torch2numpy(s.z),
            )
            if c != 0:
                r = 1 / self.torch2numpy(s.c)
            else:
                r = np.inf
            if abs(h) > abs(r):
                h = abs(r)
            if r != np.inf:
                center = z + r
                if r > 0:
                    theta = np.arcsin(h / r)
                    thetas = np.linspace(np.pi - theta, np.pi + theta, 1000)
                else:
                    theta = np.arcsin(h / r)
                    thetas = np.linspace(theta, -theta, 1000)
                z = center + np.abs(r) * np.cos(thetas)
                y = np.abs(r) * np.sin(thetas)
            else:
                y = np.linspace(h, -h, 1000)
                z = np.full_like(y, z)
            all_surface_points.append([z, y])

        all_edge_points = []
        for i, s in enumerate(self.surfaces):
            c, n, h, z = (
                self.torch2numpy(s.c),
                self.torch2numpy(s.n),
                self.torch2numpy(s.h),
                self.torch2numpy(s.z),
            )
            if n != 1:
                z = np.linspace(
                    all_surface_points[i][0][0], all_surface_points[i + 1][0][0], 1000
                )
                y = np.full_like(z, h)
                all_edge_points.append([z, y])
                y = np.full_like(z, -h)
                all_edge_points.append([z, y])
        return all_surface_points, all_edge_points

    def draw_lights(self):
        all_cross_points = []

        input_lights = self.lights[0]
        cross_points = []
        start_z = self.start_z
        for angle_lights in input_lights:
            angle_cross_points = []
            for l in angle_lights:
                u, p, q = (
                    self.torch2numpy(l.u),
                    self.torch2numpy(l.p),
                    self.torch2numpy(l.q),
                )
                z = np.array(start_z)
                if abs(u) > EPSILON:
                    y = np.tan(u) * (z - p + q / (np.sin(u)))
                else:
                    y = q
                angle_cross_points.append([z, y, l.c])
            cross_points.append(angle_cross_points)
        all_cross_points.append(cross_points)

        for space_idx in range(len(self.lights) - 1):
            lights_in, lights_out = self.lights[space_idx], self.lights[space_idx + 1]
            cross_points = []

            for angle_idx in range(len(lights_in)):
                angle_cross_points = []
                for q_idx in range(len(lights_in[angle_idx])):
                    l, l_1 = lights_in[angle_idx][q_idx], lights_out[angle_idx][q_idx]
                    u, p, q = (
                        self.torch2numpy(l.u),
                        self.torch2numpy(l.p),
                        self.torch2numpy(l.q),
                    )
                    u_1, p_1, q_1 = (
                        self.torch2numpy(l_1.u),
                        self.torch2numpy(l_1.p),
                        self.torch2numpy(l_1.q),
                    )
                    if abs(u * u_1) > EPSILON and abs(u - u_1) > EPSILON:
                        delta = u - u_1
                        z = (
                            np.tan(u) * (p - q / np.sin(u))
                            - np.tan(u_1) * (p_1 - q_1 / np.sin(u_1))
                        ) / (np.tan(u) - np.tan(u_1))
                        y = np.tan(u_1) * (z - p_1 + q_1 / np.sin(u_1))
                    elif abs(u) < EPSILON and abs(u_1) > EPSILON:
                        z = q / np.tan(u_1) + p_1 - q_1 / np.sin(u_1)
                        y = q
                    elif abs(u) > EPSILON and abs(u_1) < EPSILON:
                        z = q_1 / np.tan(u) + p - q / np.sin(u)
                        y = q_1
                    elif abs(u) < EPSILON and abs(u_1) < EPSILON:
                        z = p_1
                        y = q_1
                    else:
                        z = p_1
                        y = np.tan(u_1) * (z - p_1 + q_1 / np.sin(u_1))
                    angle_cross_points.append([z, y, l.c])
                cross_points.append(angle_cross_points)
            all_cross_points.append(cross_points)
        return all_cross_points
