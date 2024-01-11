import torch
from ..model.optical_system import OpticalSystemModule
from ...config.config import TrainCfg
import multiprocessing
from multiprocessing import Pool
from joblib import Parallel, delayed


def worker(osm):
    rays, sins, intersections = osm()
    f = osm.get_f()
    surfaces = osm.get_surface()
    return rays, sins, intersections, f, surfaces


class TracingBuilder:
    def __init__(self) -> None:
        pass

    def g_thick_map(self, input):
        out = input * TrainCfg().output_scale[0][0] + TrainCfg().output_scale[0][1]
        return out

    def a_thick_map(self, input):
        out = input * TrainCfg().output_scale[1][0] + TrainCfg().output_scale[1][1]
        return out

    def get_config_list(self, lens_system):
        config_list = []
        for i, bs in enumerate(lens_system):
            config = [
                (bs[0], self.g_thick_map(bs[1]), 1.5168, 60),
                (bs[2], self.a_thick_map(bs[3]), 1, 60),
                (bs[4], self.g_thick_map(bs[5]), 1.5168, 60),
                (bs[6], self.a_thick_map(bs[7]), 1, 60),  #
                # (bs[8], self.g_thick_map(bs[9]), 1.5168, 60),
                # (bs[10], self.a_thick_map(bs[11]), 1, 60),
                # (bs[12], self.g_thick_map(bs[13]), 1.5168, 60),
                # (bs[14], self.a_thick_map(bs[15]), 1, 60),
                # (bs[16], self.g_thick_map(bs[17]), 1.5168, 60),
                # (bs[18], self.a_thick_map(bs[19]), 1, 60),
                (0, torch.inf, 1, 60),
            ]
            config_list.append(config)
        return config_list

    def get_model_list(self, sys_param, config_list):
        osm_list = []
        sys_param_transed = []
        for i, batch in enumerate(config_list):
            osm = OpticalSystemModule()
            epd, field = (
                sys_param[i][0].item() * TrainCfg().input_scale[0][0]
                + TrainCfg().input_scale[0][1],
                sys_param[i][1].item() * TrainCfg().input_scale[1][0]
                + TrainCfg().input_scale[1][1],
            )
            sys_param_transed.append([epd, field])
            osm.set_system_param(epd, field, stop_face=0)
            for surface in batch:
                osm.add_surface(surface)
            # osm.to_cuda()
            osm_list.append(osm)
        return osm_list, sys_param_transed

    def get_rays_and_surfaces(self, sys_param, lens_system):
        config_list = self.get_config_list(lens_system)
        osm_list, sys_param_transed = self.get_model_list(sys_param, config_list)
        rays_list, sins_list, surfaces_list, intersections_list = [], [], [], []
        f_list = []

        for osm in osm_list:
            rays, sins, intersections = osm()
            f = osm.get_f()
            surfaces = osm.get_surface()
            rays_list.append(rays)
            sins_list.append(sins)
            surfaces_list.append(surfaces)
            intersections_list.append(intersections)
            f_list.append(f)

        # results = Parallel(n_jobs=4, backend="loky")(
        #     delayed(worker)(osm) for osm in osm_list
        # )
        # for i in results:
        #     rays, sins, intersections, f, surfaces = i
        #     rays_list.append(rays)
        #     sins_list.append(sins)
        #     surfaces_list.append(surfaces)
        #     intersections_list.append(intersections)
        #     f_list.append(f)
        # rays, sins, intersections, f, surfaces
        # a = torch.mean(torch.stack(results))

        intersections_tensor = torch.stack(intersections_list)
        return (
            rays_list,
            sins_list,
            surfaces_list,
            intersections_tensor,
            f_list,
            sys_param_transed,
        )

    # with Pool(processes=8) as pool:
    #     for x in pool.imap(worker, osm_list):
    #         rays_list.append(rays)
    #         sins_list.append(sins)
    #         surfaces_list.append(surfaces)
    #         intersections_list.append(intersections)
    #         f_list.append(f)
    # print("check done!")
