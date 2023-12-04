import os
import torch
import numpy as np
import random


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_epoch(epoch, shotpath, model, optim):
    os.makedirs(shotpath, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "next_epoch": epoch + 1,
            "next_lr": optim.param_groups[-1]["lr"],
        },
        shotpath + "/step_{}.pth".format(epoch),
    )
    print(f"save epoch {epoch}")


def print_epoch(train_loss, shotpath):
    train_loss = np.mean(train_loss, 0)
    train_loss_info = "train mean loss = {} ".format(train_loss)
    print(train_loss_info)
    os.makedirs(shotpath, exist_ok=True)
    with open(f"{shotpath}/test.txt", "a") as file:
        print(train_loss_info, file=file)
