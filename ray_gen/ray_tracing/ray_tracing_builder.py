from .lib.tracing_builder import TracingBuilder


class RayTracingBuilder:
    def __init__(self) -> None:
        self.config_builder = TracingBuilder()

    def get_rays_and_surfaces(self, sys_param, lens_system):
        infos = self.config_builder.get_rays_and_surfaces(sys_param, lens_system)
        return infos
