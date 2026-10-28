#!/bin/env python
#
# Naming this file 'matplotlib.py' will cause problems
#
import pandas as pd
import os

if not os.path.exists("dem_score.csv"):
    # Download the file
    from urllib.request import urlretrieve

    urlretrieve(
        "https://moderndive.com/data/dem_score.csv",
        "dem_score.csv",
    )

if not os.path.exists("dem_score.csv"):
    os.sys.exit("Failed to download 'dem_score.csv'")

table = pd.read_csv("dem_score.csv")
table = table.melt(id_vars = ["country"])
table.rename(columns = {"variable": "year", "value": "democratic_score"}, inplace = True)
table.to_csv("tidy_dem_scores.csv", index = False)
