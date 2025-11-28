import matplotlib.pyplot as plt
import numpy as np
### Comment next 2 lines if NOT running directly from cmd line:
import matplotlib
matplotlib.use("TkAgg")

x = np.linspace(-5,5, 50)
y = 1/(1+np.exp(-x))
fig, ax = plt.subplots()
ax.plot(x,y)
plt.show()
### comment line above, uncomment line below to save figure (exercise 2)
#plt.savefig("matplotlib-ex1", format='svg')
### rename file as desired
