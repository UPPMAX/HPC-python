import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import linregress

matplotlib.use('TkAgg')

# Example from https://ourcodingclub.github.io/tutorials/pandas-python-intro/ 
dataframe = pd.read_csv("scottish_hills.csv")

x = dataframe.Height
y = dataframe.Latitude

stats = linregress(x, y)

m = stats.slope
b = stats.intercept

plt.scatter(x, y)
plt.plot(x, m * x + b, color="red")   # I've added a color argument here

plt.show()
