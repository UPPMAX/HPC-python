
<!-- From https://docs.readthedocs.com/platform/stable/guides/cross-referencing-with-sphinx.html#explicit-targets -->
(seaborn)=

# Seaborn

:::{admonition} Learning outcomes
:class: note

At the end of this sessions, learners ...

- understand why Seaborn is important
- have run Python code that uses Seaborn

:::

## Loading Seaborn

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|How to load Seaborn
-----------|-----------------------------------------------------------------------------------------------------------------------
Alvis      |`module load Seaborn/0.13.2-gfbf-2024a`
Kebnekaise |`module load GCC/13.2.0 Seaborn/0.13.2`
COSMOS     |`module load GCC/13.2.0 Seaborn/0.13.2`
Pelle      |`module load Seaborn/0.13.2-gfbf-2024a`
Tetralith  |`module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11; pip install seaborn`
Dardel     |`module load cray-python/3.11.7 PDCOLD/23.12 matplotlib/3.8.2-cpeGNU-23.12`

<!-- markdownlint-enable MD013 -->

## Exercises

```python
import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline
import numpy as np
import pandas as pd

# Create some data
rng = np.random.RandomState(0)
x = np.linspace(0, 10, 500)
y = np.cumsum(rng.randn(500, 6), 0)

# Plot the data with Matplotlib defaults
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');


import seaborn as sns
sns.set()


# same plotting code as above!
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');
```

```python
data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])

# Two overlaid density plots
for col in 'xy':
    sns.kdeplot(data[col], shade=True)

# Density plot
sns.kdeplot(data);
```

## External links

- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
