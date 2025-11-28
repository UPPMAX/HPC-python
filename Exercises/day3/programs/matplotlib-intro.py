import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')

x = np.linspace(-5,5, 50)
y = 1/(1+np.exp(-x))
fig, ax = plt.subplots()
ax.plot(x,y)
plt.show()
# comment line above & uncomment line below to save figure; file name up to you
#plt.savefig("matplotlib-ex1", format='svg')
