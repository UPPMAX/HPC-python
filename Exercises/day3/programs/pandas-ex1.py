import pandas as pd

df = pd.read_csv('exoplanets_5250_EarthUnits_fixed.csv',index_col=0)
df.to_csv('./docs/day3/exoplanets_5250_EarthUnits.txt', sep='\t',index=True)
