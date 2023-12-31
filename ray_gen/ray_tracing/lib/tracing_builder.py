import torch
from ..model.optical_system import OpticalSystemModule


class TracingBuilder:
    def __init__(self) -> None:
        pass

    def g_thick_map(self, input):
        out = (input * 15) + 20  # 5-25
        return out

    def a_thick_map(self, input):
        out = (input * 300) + 310  # 10-110
        return out

    def get_config_list(self, lens_system):
        config_list = []
        for i, bs in enumerate(lens_system):
            config = [
                (bs[0], self.g_thick_map(bs[1]), 1.5168, 60),
                (bs[2], self.a_thick_map(bs[3]), 1, 60),
                (bs[4], self.g_thick_map(bs[5]), 1.5168, 60),
                (bs[6], self.a_thick_map(bs[7]), 1, 60),
                (0, torch.inf, 1, 60),
            ]
            config_list.append(config)
        return config_list

    def get_model_list(self, sys_param, config_list):
        osm_list = []
        for i, batch in enumerate(config_list):
            osm = OpticalSystemModule()  # .cuda()
            epd, field = (
                sys_param[i][0].item() * 20 + 30,
                sys_param[i][1].item() * 15 + 15,
            )
            osm.set_system_param(epd, field, stop_face=0)
            for surface in batch:
                osm.add_surface(surface)
            osm_list.append(osm)
        return osm_list

    def get_rays_and_surfaces(self, sys_param, lens_system):
        config_list = self.get_config_list(lens_system)
        osm_list = self.get_model_list(sys_param, config_list)
        rays_list, sins_list, surfaces_list, intersections_list = [], [], [], []
        for osm in osm_list:
            rays, sins, intersections = osm()
            surfaces = osm.get_surface()
            rays_list.append(rays)
            sins_list.append(sins)
            surfaces_list.append(surfaces)
            intersections_list.append(intersections)

        intersections_tensor = torch.stack(intersections_list)
        return rays_list, sins_list, surfaces_list, intersections_tensor
