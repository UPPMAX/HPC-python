#!/bin/env python
#
# Naming this file 'seaborn.py' will cause problems
#
import seaborn as sns
y = [0, 1, 4, 9, 16]
sns.lineplot(x = range(len(y)), y = y).figure.savefig("seaborn_exercise_2.png")
