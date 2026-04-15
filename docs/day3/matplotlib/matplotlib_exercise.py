#!/bin/env python
#
# Naming this file 'matplotlib.py' will cause problems
#
import pandas as pd
import matplotlib.pyplot as plt
import os

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


plt.scatter(table["carat"], table["price"])
plt.savefig("matplotlib_exercise.png")
