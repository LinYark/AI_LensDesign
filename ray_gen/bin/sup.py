import os
import time
import torch
import numpy as np
import random

start_time_epoch = time.time()
start_time_item = time.time()


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


def print_epoch(train_loss, shotpath, epoch):
    global start_time_epoch
    train_loss = np.mean(train_loss, 0)
    end_time_epoch = time.time()
    delta_time = end_time_epoch - start_time_epoch
    start_time_epoch = end_time_epoch
    np.set_printoptions(6)
    train_loss_info = f"{delta_time:6.1f}s, {epoch}, train_loss = {train_loss} "
    print(train_loss_info)
    os.makedirs(shotpath, exist_ok=True)
    with open(f"{shotpath}/test.txt", "a") as file:
        print(train_loss_info, file=file)


def print_item(train_loss, shotpath, epoch, idx):
    global start_time_item
    train_loss = np.mean(train_loss, 0)
    end_time_item = time.time()
    delta_time = end_time_item - start_time_item
    start_time_item = end_time_item
    np.set_printoptions(5)
    train_loss_info = f"{delta_time:3.1f}s, {epoch}, {idx}, train_loss = {train_loss} "
    print(train_loss_info)
    os.makedirs(shotpath, exist_ok=True)
    with open(f"{shotpath}/test.txt", "a") as file:
        print(train_loss_info, file=file)


def resume_training(model, path):
    load_data = torch.load(path)

    model_dict = model.state_dict()
    state_dict = load_data["state_dict"]
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
