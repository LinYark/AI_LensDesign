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


def eval():
    with torch.no_grad():
        all_epoch = 99999
        print_hz = 100
        seed_torch()

        model = ModelBuilder().cuda().eval()  # .cuda().train()
        resume_training(model, "./workspace/snapshot/step1/step_20.pth")

        train_data_loader = DataLoadBuilder().build_train_loader()
        val_data_loader = DataLoadBuilder().build_valid_loader()

        my_optim = OptimBuilder()
        optim, scheduler = my_optim.build_optim_and_scheduler(model)
        loss_obj = LossBuilder()
        shotpath = "./workspace/snapshot/step1"

        for epoch in range(all_epoch):
            train_loss = []
            for i, sys_param in enumerate(val_data_loader):
                sys_param = sys_param.cuda()
                lens_system = model(sys_param).cpu()

                loss = loss_obj.get_loss(sys_param, lens_system)
                excel_item = loss_obj.prepare_for_excel(loss)
            loss_obj.show(epoch, shotpath, loss)
            print_epoch(train_loss, shotpath, epoch)
