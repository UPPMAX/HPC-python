import pandas as pd
import numpy as np

loss_sum = 0
for chunk in pd.read_csv('./docs/day3/global_disaster_response_2018-2024.csv',
                         chunksize=10000):
    loss_sum+=chunk['economic_loss_usd'].sum()
print('total loss over all disasters in this database: $',
      np.round(loss_sum/10**9,2), 'billion USD')  
