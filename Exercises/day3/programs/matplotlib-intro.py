import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')

x = np.linspace(-5,5, 50)
y = 1/(1+np.exp(-x))
fig, ax = plt.subplots()
ax.plot(x,y)
plt.show()
