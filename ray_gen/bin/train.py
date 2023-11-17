from ..model.model_builder import ModelBuilder
from ..data.data_loader import DataLoadBuilder
from ..optimizer.optim_builder import OptimBuilder
from ..utils.common import seed_torch
import numpy as np
import os
import torch

def train():
    all_epoch = 999
    seed_torch()

    model = ModelBuilder.cuda().train()
    train_data_loader = DataLoadBuilder().build_train_loader()
    val_data_loader = DataLoadBuilder().build_valid_loader()
    my_optim = OptimBuilder()
    optim,scheduler = my_optim.build_optim_and_scheduler()

    for epoch in range(all_epoch):
        cur_lr = optim.param_groups[-1]['lr']
        epoch_info = f"1.epoch={epoch},. current lr={cur_lr}"
        print(epoch_info)

        train_loss = []
        for i, data in enumerate(train_data_loader):
            outputs = model(data)
            loss = outputs['total_loss']

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss.append(loss.item())

        train_loss = np.mean(train_loss,0)
        train_loss_info = "2. train  mean loss = {}   ".format(train_loss)
        print(train_loss_info)

        scheduler.step()
        if epoch > 9:
            shotpath= "./snapshot/step1"
            if not os.path.exists(shotpath):
                os.makedirs(shotpath)
            torch.save({'state_dict':model.state_dict(),
                        'next_epoch':epoch+1,
                        'next_lr': optim.param_groups[-1]['lr']
                        }, shotpath+"/step1_{}.pth".format(epoch))



    
            































