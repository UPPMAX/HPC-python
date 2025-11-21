import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mtcars = pd.read_csv('mtcars.csv')
mtcars.head()
mtcars.info()
mtcars.shape
res = sns.barplot(x='cyl', y='carb', data=mtcars)
sns.heatmap(mtcars.corr(),cbar=True,linewidths=0.5)

