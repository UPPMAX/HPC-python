import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv("scottish_hills.csv")
x = dataframe.Height
y = dataframe.Latitude
plt.scatter(x, y)
plt.show()  # or plt.savefig("name.png")
