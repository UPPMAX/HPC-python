#!/bin/env python
#
# Naming this file 'matplotlib.py' will cause problems
#
import pandas
import pandas as pd
import os

################################################################################
# Minimal code
################################################################################
print(pandas.__version__)

################################################################################
# Diamonds dataset
################################################################################

if not os.path.exists("diamonds.csv"):
    # Download the file
    from urllib.request import urlretrieve

    urlretrieve(
        "https://uppmax.github.io/HPC-python/_downloads/2c3f44b9035e3effc9e3b2854f37f1f0/diamonds.csv",
        "diamonds.csv",
    )

if not os.path.exists("diamonds.csv"):
    os.sys.exit("Failed to download 'diamonds.csv'")

table = pd.read_csv("diamonds.csv")
print(table)

table.to_csv("my_new_file_without_index.csv", index = False)
table.to_csv("my_new_file_with_index.csv", index = True)

################################################################################
# Democratic scores dataset
################################################################################

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
print(table)

table = table.melt(id_vars = ["country"])

table.rename(columns = {"variable": "year", "value": "democratic_score"}, inplace = True)

print(table)

table.to_csv("tidy_dem_scores.csv", index = False)
