import torch
import torch.nn as nn


class Optim:
    def __init__(self) -> None:
        pass

    def eval(model):
        model.eval()

    def train(model):
        model.train()

    def build_optim(model):
        optim = torch.optim.Adam(
            model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3
        )
        return optim

    def build_scheduler(optim):
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optim,
            base_lr=1e-6,
            max_lr=1e-4,
            step_size_up=10,
            mode="triangular",
            cycle_momentum=False,
        )
        return scheduler
