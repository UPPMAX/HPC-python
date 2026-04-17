#!/bin/env python
#
# Naming this file 'matplotlib.py' will cause problems
#
import pandas as pd

table = pd.read_csv("diamonds.csv")
print(table)
table.to_csv("pandas_exercise_2.csv")
# table.to_csv("pandas_exercise_2.csv", index = False)

