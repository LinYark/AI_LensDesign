from .optim.optim import Optim


class OptimBuilder:
    def __init__(self) -> None:
        self.my_optim = Optim()

    def build_optim_and_scheduler(self, model, resume_flg):
        optim = self.my_optim.build_optim(model)
        scheduler = self.my_optim.build_scheduler(optim, resume_flg)
        return optim, scheduler
