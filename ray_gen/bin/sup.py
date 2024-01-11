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

    # print(torch.get_num_threads())
    # print(torch.__config__.parallel_info())
    # print(*torch.__config__.show().split("\n"), sep="\n")
    # os.environ["OMP_NUM_THREADS"] = "8"  # 设置OpenMP计算库的线程数
    # os.environ["MKL_NUM_THREADS"] = "8"  # 设置MKL-DNN CPU加速库的线程数。
    # torch.set_num_threads(8)


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
    np.set_printoptions(
        floatmode="fixed",
        precision=3,
        formatter={"float": lambda x: "{:10.2f}".format(x)},
    )
    train_loss_info = f"{delta_time:6.1f}s, {epoch},  loss = {train_loss.T} "
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
    np.set_printoptions(
        floatmode="fixed",
        precision=3,
        formatter={"float": lambda x: "{:5.3f}".format(x)},
    )
    train_loss_info = (
        f"{delta_time:6.1f}s, {epoch}, {idx}, [ArStnRfR], e_loss = {train_loss.T} "
    )
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
