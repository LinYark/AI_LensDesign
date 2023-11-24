import torch
import torch.nn as nn


class Optim:
    def __init__(self) -> None:
        pass

    def eval(self, model):
        model.eval()

    def train(self, model):
        model.train()

    def build_optim(self, model):
        optim = torch.optim.Adam(
            model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3
        )
        return optim

    def build_scheduler(self, optim):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim, T_0=10, T_mult=1, eta_min=1e-5, last_epoch=-1, verbose=True
        )
        return scheduler
