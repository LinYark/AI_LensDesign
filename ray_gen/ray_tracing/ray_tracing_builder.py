from .lib.tracing_builder import TracingBuilder


class RayTracingBuilder:
    def __init__(self) -> None:
        self.config_builder = TracingBuilder()

    def get_rays_and_surfaces(self, lens_system):
        rays_list, surfaces_list = self.config_builder.get_rays_and_surfaces(
            lens_system
        )
        return rays_list, surfaces_list
