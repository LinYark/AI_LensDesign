import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler


class Optim:
    def __init__(self) -> None:
        pass

    def eval(self, model):
        model.eval()

    def train(self, model):
        model.train()

    def build_optim(self, model):
        optim = torch.optim.Adam(
            model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3
        )
        return optim

    def build_scheduler(self, optim):
        one_turn = 100
        scheduler1 = lr_scheduler.LinearLR(
            optim, start_factor=1e-3, end_factor=1e-5, total_iters=one_turn
        )
        scheduler2 = lr_scheduler.CosineAnnealingLR(optim, T_max=one_turn, eta_min=1e-6)
        milestones = [one_turn]
        schedulers = [scheduler1, scheduler2]

        scheduler = lr_scheduler.SequentialLR(optim, schedulers, milestones)
        return scheduler
