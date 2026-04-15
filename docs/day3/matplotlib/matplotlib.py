import pandas as pd

import os
if not os.path.exists("diamonds.csv"):
    # Download the file
    from urllib.request import urlretrieve    
    urlretrieve("https://uppmax.github.io/HPC-python/_downloads/2c3f44b9035e3effc9e3b2854f37f1f0/diamonds.csv", "diamonds.csv")

if not os.path.exists("diamonds.csv"):
    sys.exit("Failed to download 'diamonds.csv'")

table = pd.read_csv("diamonds.csv")
df = pd.DataFrame(table)


import matplotlib.pyplot as plt

fig, ax = plt.subplots()             # Create a figure containing a single Axes.
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot some data on the Axes.

plt.show()    
plt.figure().savefig('matplotlib.png')
