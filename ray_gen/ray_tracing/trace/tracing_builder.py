import torch
from ..model.optical_system import OpticalSystemModule
from ..model.drawer import OpticalSystemDrawer


class TracingBuilder:
    def __init__(self) -> None:
        self.drawer = OpticalSystemDrawer()
        self.drawer.set_start_z(-100)

    def g_thick_map(self, input):
        out = (input * 10) + 15  # 5-25
        return out

    def a_thick_map(self, input):
        out = (input * 50) + 60  # 10-110
        return out

    def get_config_list(self, input):
        config_list = []
        for i, bs in enumerate(input):
            config = [
                (bs[0], self.g_thick_map(bs[1]), 1.5168, 40),
                (bs[2], self.a_thick_map(bs[3]), 1, 40),
                (torch.inf, torch.inf, 1, 40),
            ]
            config_list.append(config)
        return config_list

    def get_model_list(self, config_list):
        osm_list = []
        for i, batch in config_list:
            osm = OpticalSystemModule()
            for surface in batch:
                osm.add_surface(surface)
            osm_list.append(osm)
        return osm_list

    def forward(self, input):
        config_list = self.get_config_list(input)
        osm_list = self.get_model_list(config_list)
        output_list = []
        for i, osm in osm_list:
            rays = osm()
            surfaces = osm.get_surface()
            output_list.append([rays, surfaces])
        return output_list
