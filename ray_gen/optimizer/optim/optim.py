import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler


class Optim:
    def __init__(self) -> None:
        self.first_turn, self.second_turn = 10, 20

    def build_optim(self, model):
        optim = torch.optim.Adam(
            model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3
        )
        return optim

    def build_scheduler_start(self, optim):
        def rule(epoch):
            lamda = self.first_turn * min(1, 2.0 * epoch / self.first_turn)
            return lamda

        scheduler1 = torch.optim.lr_scheduler.LambdaLR(
            optim, lr_lambda=rule, verbose=True
        )
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.second_turn, eta_min=1e-5, verbose=True
        )
        milestones = [self.first_turn]
        schedulers = [scheduler1, scheduler2]

        scheduler = torch.optim.lr_scheduler.SequentialLR(optim, schedulers, milestones)
        scheduler.step()
        return scheduler

    def build_scheduler_continue(self, optim):
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.second_turn, eta_min=1e-5, verbose=True
        )
        return scheduler2

    def build_scheduler(self, optim, resume_flg):
        if resume_flg == False:
            scheduler = self.build_scheduler_start(optim)
        else:
            scheduler = self.build_scheduler_continue(optim)
        return scheduler

    def eval(self, model):
        model.eval()

    def train(self, model):
        model.train()
