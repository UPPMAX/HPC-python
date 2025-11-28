import numpy as np
import pandas as pd

df = pd.DataFrame( np.arange(1,13).reshape((4,3)), index=['w','x','y','z'], columns=['a','b','c'] )
print(df)
