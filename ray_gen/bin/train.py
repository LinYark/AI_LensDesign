import os 
import sys
sys.path.append(os.getcwd())
import numpy as np
import os
import torch

from ray_gen.model.model_builder import ModelBuilder
from ray_gen.data.data_loader import DataLoadBuilder
from ray_gen.optimizer.optim_builder import OptimBuilder
from ray_gen.utils.common import seed_torch

def train():
    all_epoch = 999
    seed_torch()

    model = ModelBuilder()#.cuda().train()
    train_data_loader = DataLoadBuilder().build_train_loader()
    val_data_loader = DataLoadBuilder().build_valid_loader()
    # a = next(iter(train_data_loader))
    my_optim = OptimBuilder()
    optim,scheduler = my_optim.build_optim_and_scheduler(model)

    for epoch in range(all_epoch):
        cur_lr = optim.param_groups[-1]['lr']
        epoch_info = f"\n\nepoch={epoch},. current lr={cur_lr}"
        print(epoch_info)

        train_loss = []
        for i, data in enumerate(train_data_loader):
            outputs = model(data)
            loss = outputs[:,0]#['total_loss']

            loss = (torch.sum(loss)-4)**2
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss.append(loss.item())

        train_loss = np.mean(train_loss,0)
        train_loss_info = "train  mean loss = {}   ".format(train_loss)
        print(train_loss_info)

        scheduler.step()
        if epoch > 999:
            shotpath= "./snapshot/step1"
            os.makedirs(shotpath,exist_ok=True)
            torch.save({'state_dict':model.state_dict(),
                        'next_epoch':epoch+1,
                        'next_lr': optim.param_groups[-1]['lr']
                        }, shotpath+"/step1_{}.pth".format(epoch))



    
            































