#!/bin/env python
#
# Naming this file 'seaborn.py' will cause problems
#
import seaborn as sns

y = [0, 1, 4, 9, 16]
sns.set_theme()


# sns.lineplot(x = range(len(y)), y = y).figure.show()

sns.lineplot(x = range(len(y)), y = y).figure.savefig("what_is_seaborn.png")
