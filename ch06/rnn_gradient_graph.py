import numpy as np
import matplotlib.pyplot as plt

N = 2
H = 3
T = 20

dh = np.ones((N, H))
np.random.seed(3)
Wh = np.random.randn(H, H)
norm_list = []
for t in range(T):
  dh = np.dot(dh, Wh.T)
  norm = np.sqrt(np.sum(dh**2)) / N
  norm_list.append(norm)

x = np.arange(T)
y = np.array(norm_list)
plt.plot(x, y)
plt.show()