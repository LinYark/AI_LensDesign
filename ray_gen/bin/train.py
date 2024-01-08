import os
import sys

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
import numpy as np
import os
import torch
from tqdm import tqdm
from ray_gen.model.model_builder import ModelBuilder
from ray_gen.data.data_loader import DataLoadBuilder
from ray_gen.optimizer.optim_builder import OptimBuilder
from ray_gen.loss.loss_builder import LossBuilder
from ray_gen.loss.loss_builder import LossBuilder
from ray_gen.bin.sup import (
    seed_torch,
    save_epoch,
    print_epoch,
    print_item,
    resume_training,
)


def train():
    all_epoch = 99999
    print_hz = 100
    seed_torch()

    model = ModelBuilder().cuda().train()  # .cuda().train()
    # resume_training(model, "./workspace/snapshot/step1/step_100.pth")

    train_data_loader = DataLoadBuilder().build_train_loader()
    val_data_loader = DataLoadBuilder().build_valid_loader()

    my_optim = OptimBuilder()
    optim, scheduler = my_optim.build_optim_and_scheduler(model)
    loss_obj = LossBuilder()
    shotpath = "./workspace/snapshot/step0108"

    for epoch in range(all_epoch):
        train_loss = []
        for i, sys_param in enumerate(train_data_loader):
            sys_param = sys_param.cuda()
            lens_system = model(sys_param).cpu()

            loss = loss_obj.get_loss(sys_param, lens_system)
            optim.zero_grad()
            loss["all_loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=10, norm_type=2
            )
            optim.step()
            with torch.no_grad():
                loss_info = loss_obj.backup(loss)
                train_loss.append(loss_info)
                if i % print_hz == 0:
                    print_item([loss_info], shotpath, epoch, i)
                    # loss_obj.show(epoch, shotpath, loss)
        scheduler.step()
        loss_obj.show(epoch, shotpath, loss)
        print_epoch(train_loss, shotpath, epoch)

        if epoch % 20 == 0 and epoch != 0:
            save_epoch(epoch, shotpath, model, optim)
