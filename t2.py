import numpy as np  
import matplotlib.pyplot as plt  
  
# 光线的初始位置和方向  
x0, y0 = 0.0, 0.0  
dx, dy = 1.0, 0.0  
  
# 光线的最大传播距离  
max_distance = 10.0  
  
# 光线的折射率  
n = 1.5  
  
# 将光线分成很多小段，每段长度为dx  
dx = 0.01  
num_steps = int(max_distance / dx)  
  
# 存储光线的位置和方向的列表  
positions = []  
directions = []  
  
# 光线追迹循环  
for i in range(num_steps):  
    # 更新光线的位置  
    x = x0 + i * dx  
    y = y0 + n * dx * (1 - (x / max_distance) ** 2) ** 0.5  
    positions.append((x, y))  
      
    # 更新光线的方向，考虑折射效应  
    if i < num_steps - 1:  
        nx = (x / max_distance) ** 2  
        ny = (y / max_distance) ** 2  
        k = (n ** 2 * ny - nx) / (n ** 2 * nx + ny)  
        dx_new = dx * (1 - k ** 2) ** 0.5 * np.cos(k * np.pi / 2)  # 根据折射定律计算新的方向  
        dy_new = dx * (1 - k ** 2) ** 0.5 * np.sin(k * np.pi / 2)  # 根据折射定律计算新的方向  
        directions.append((dx_new, dy_new))  
    else:  
        directions.append((0.0, 0.0))  # 光线到达终点，方向设置为零  
          
# 绘制结果  
plt.figure()  
plt.plot([pos[0] for pos in positions], [pos[1] for pos in positions], 'r-')  
plt.scatter([pos[0] for pos in positions], [pos[1] for pos in positions], s=50, c='b')  
plt.quiver(positions[:-1][0], positions[:-1][1], directions[:-1][0], directions[:-1][1])  # 使用更新后的方向绘制箭头  
plt.xlim(-max_distance, max_distance)  
plt.ylim(-max_distance, max_distance)  
plt.grid()  
plt.show()