
import matplotlib.pyplot as plt
import numpy as np
a = np.append(np.arange(2,3,0.1),3)
thetas = np.flip(a) 

center = (1, 1)
radius = 5
theta = np.radians(np.arange(-90, 90, 0.1))
x = center[0] + radius * np.cos(theta)
y = center[1] + radius * np.sin(theta)
plt.plot(x, y)
plt.show()