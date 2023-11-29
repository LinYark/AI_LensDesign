from .trace.tracing_builder import TracingBuilder


class RayTracingBuilder:
    def __init__(self) -> None:
        self.config_builder = TracingBuilder()
