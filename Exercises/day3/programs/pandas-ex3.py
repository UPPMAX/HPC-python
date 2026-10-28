import pandas as pd
import numpy as np

df = pd.read_csv('./docs/day3/exoplanets_5250_EarthUnits_fixed.csv',index_col=0)
print("Before:\n", df['planet_type'].memory_usage(deep=True), '\n')

# Convert planet_type to Categorical
df['planet_type']=df['planet_type'].astype('category')
print("After:\n", df['planet_type'].memory_usage(deep=True))
