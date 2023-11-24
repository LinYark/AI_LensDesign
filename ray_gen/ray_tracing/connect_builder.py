from .connet.config_builder import ConfigBuilder
from .model.optical_system import OpticalSystemModule


class ConnectBuilder:
    def __init__(self) -> None:
        self.config_builder = ConfigBuilder()

    def get_config_list(self, input):
        return self.config_builder.connect(input)

    def get_model(self, config_list):
        osm = OpticalSystemModule()
        for surface in config_list:
            osm.add_surface(surface)
