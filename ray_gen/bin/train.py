import os
import sys

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
import numpy as np
import os
import torch

from ray_gen.model.model_builder import ModelBuilder
from ray_gen.data.data_loader import DataLoadBuilder
from ray_gen.optimizer.optim_builder import OptimBuilder
from ray_gen.loss.loss_builder import LossBuilder
from ray_gen.utils.common import seed_torch
from ray_gen.loss.loss_builder import LossBuilder


def train():
    all_epoch = 999
    seed_torch()

    model = ModelBuilder().train()  # .cuda().train()
    train_data_loader = DataLoadBuilder().build_train_loader()
    val_data_loader = DataLoadBuilder().build_valid_loader()
    # a = next(iter(train_data_loader))
    my_optim = OptimBuilder()
    optim, scheduler = my_optim.build_optim_and_scheduler(model)
    loss_obj = LossBuilder().get_loss_instance()
    for epoch in range(all_epoch):
        train_loss = []
        for i, sys_param in enumerate(train_data_loader):
            # sys_param = sys_param.cuda()
            lens_system = model(sys_param)

            loss = loss_obj.get_loss(sys_param, lens_system)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss.append(loss.item())

        scheduler.step()
        if epoch > 999:
            shotpath = "./ray_gen/snapshot/step1"
            os.makedirs(shotpath, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "next_epoch": epoch + 1,
                    "next_lr": optim.param_groups[-1]["lr"],
                },
                shotpath + "/step_{}.pth".format(epoch),
            )

        train_loss = np.mean(train_loss, 0)
        train_loss_info = "train mean loss = {} ".format(train_loss)
        print(train_loss_info)
        print(lens_system[-1])
