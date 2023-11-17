import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

"""生成随机样本"""
def get_samples(batch_size=100, n_inputs=2):
    # return torch.rand((batch_size, n_inputs), requires_grad=True)     
    return torch.rand((batch_size, n_inputs))

"""建立模型"""
class Sine(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(x)

mlp = nn.Sequential(
 nn.Linear(2, 100),
 Sine(), 
 nn.Linear(100, 100), 
 Sine(), 
 nn.Linear(100, 1)
)

"""初始情况(t=0), 两个变量:x,t=0"""
def get_samples_ini(batch_size=100):
    x = torch.rand(batch_size)
    p0 = torch.sin(2.*np.pi*x).unsqueeze(1)
    return torch.stack([  
        x,
        torch.zeros(batch_size) # t=0
    ], axis=-1), p0

"""考虑边界情况(x=0,x=1)"""
def get_samples_boco(batch_size=100):
    t = torch.rand(batch_size)
    X0 = torch.stack([  
        torch.zeros(batch_size), # x=0
        t
    ], axis=-1)
    X1 = torch.stack([  
        torch.ones(batch_size), # x=1
        t
    ], axis=-1)
    return X0, X1

"""训练模型"""
BATCH_SIZE = 100
N_STEPS = 5000
U = 1

optimizer = torch.optim.Adam(mlp.parameters())
criterion = torch.nn.MSELoss()

hist = []
log_each = 500
for step in range(1, N_STEPS+1):

    # 1 PDE 优化
    X = get_samples(BATCH_SIZE, 2) # N, (X, T)
    X.requires_grad_(True)
    y_hat = mlp(X) # N, P 
    grads, = torch.autograd.grad(y_hat, X, grad_outputs=y_hat.data.new(y_hat.shape).fill_(1), create_graph=True, only_inputs=True)
    dpdx, dpdt = grads[:,0], grads[:,1] # x的梯度，t的梯度
    pde_loss = criterion(dpdt, - U*dpdx) # PDE微分函数

    # 2 初始条件优化
    X, p0 = get_samples_ini(BATCH_SIZE)
    y_hat = mlp(X) # N, P0 
    ini_loss = criterion(y_hat, p0)  # NN的输出=(t=0)的初始值
    
    # 3 边界情况优化
    X0, X1 = get_samples_boco(BATCH_SIZE)
    y_0 = mlp(X0) 
    y_1 = mlp(X1)
    bound_loss = criterion(y_0, y_1) # x=0,x=1两者目标值相等

    # 4 权重更新
    optimizer.zero_grad()
    loss = pde_loss + ini_loss + bound_loss # 三个损失相加
    loss.backward()
    optimizer.step()
   
    hist.append(loss.item())
    if step % log_each == 0:
        print(f'{step}/{N_STEPS} pde_loss {pde_loss.item():.5f} ini_loss {ini_loss.item():.5f} bound_loss {bound_loss.item():.5f}')