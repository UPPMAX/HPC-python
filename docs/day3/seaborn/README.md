
<!-- From https://docs.readthedocs.com/platform/stable/guides/cross-referencing-with-sphinx.html#explicit-targets -->
(seaborn)=

# Seaborn

![The seaborn logo](logo-wide-lightbg.svg)

> [The `seaborn` logo](https://seaborn.pydata.org/citing.html)

:::{admonition} Learning outcomes
:class: note

At the end of this sessions, learners ...

- have practiced using the documentation of favorite HPC cluster
- understand why Pandas is important
- understand why Seaborn is important
- have run Python code that uses `seaborn`
- have run Python code that uses `seaborn` to display data from a `pandas` table
:::

## Why Seaborn is important


## Loading Seaborn

- Use the documentation of the HPC cluster you work on

:::{admonition} Answer: where is your documentation?
:class: dropdown

Sorted by HPC cluster:

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC center |HPC cluster|HPC cluster-specific documentation
-----------|-----------|------------------------------------------------------------
C3SE       |Alvis      |[Documentation](https://www.c3se.chalmers.se)
UPPMAX     |Bianca     |[Documentation](https://docs.uppmax.uu.se)
LUNARC     |COSMOS     |[Documentation](https://lunarc-documentation.readthedocs.io)
PDC        |Dardel     |[Documentation](https://support.pdc.kth.se)
HPC2N      |Kebnekaise |[Documentation](https://docs.hpc2n.umu.se)
UPPMAX     |Pelle      |[Documentation](https://docs.uppmax.uu.se)
NSC        |Tetralith  |[Documentation](https://www.nsc.liu.se)

<!-- markdownlint-enable MD013 -->

:::

- In that documentation, find the software module to load the package.
  If you know how, you may also use the module system

:::{admonition} Answer: where is the `seaborn` documentation?
:class: dropdown

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|HPC cluster-specific `seaborn` documentation
-----------|-------------------------------------------------------------------------------------------------------------------
Alvis      |[`seaborn` documentation](https://www.c3se.chalmers.se/documentation/module_system/python/#numpy-and-scipy)
Bianca     |[`seaborn` documentation](https://docs.uppmax.uu.se/software/python/) [Only shows module name]
COSMOS     |[`seaborn` documentation](https://lunarc-documentation.readthedocs.io/en/latest/software/installed_software/) [Only shows module name]
Dardel     |[`seaborn` documentation](https://support.pdc.kth.se/doc/applications/python/)
Kebnekaise |[`seaborn` documentation](https://docs.hpc2n.umu.se/software/libs/Seaborn/)
Pelle      |[`seaborn` documentation](https://docs.uppmax.uu.se/software/python_bundles/#pytorch) No relevant documentation
Tetralith  |[`seaborn` documentation](https://www.nsc.liu.se/software/python) No relevant documentation

<!-- markdownlint-enable MD013 -->

:::


- Load the software module to use `seaborn`

:::{admonition} Answer: how to load the `seaborn` software module
:class: dropdown

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|How to load Seaborn
-----------|-----------------------------------------------------------------------------------------------------------------------
Alvis      |`module load Seaborn/0.13.2-gfbf-2024a`
COSMOS     |`module load GCC/13.2.0 Seaborn/0.13.2`
Dardel     |`module load cray-python/3.11.7 PDCOLD/23.12 matplotlib/3.8.2-cpeGNU-23.12`
Kebnekaise |`module load GCC/13.2.0 Seaborn/0.13.2`
Pelle      |`module load Seaborn/0.13.2-gfbf-2024a` :-)
Tetralith  |`module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11; pip install seaborn` :-)


<!-- markdownlint-enable MD013 -->

:::

## Exercises

## Exercise 1: a minimal `seaborn` program

Run the following code, that is copied from
[the `seaborn` page 'An introduction to `seaborn`'](https://seaborn.pydata.org/tutorial/introduction.html)
and combined with
[this StackOverflow post to save it to a file](https://stackoverflow.com/a/39482402/3364162)

```python
# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()

# Load an example dataset
tips = sns.load_dataset("tips")

# Create a visualization and save it to file
my_plot = sns.relplot(
    data = tips,
    x = "total_bill", y = "tip", col = "time",
    hue = "smoker", style = "smoker", size = "size",
)
fig = my_plot.get_figure()
fig.savefig("out.png") 
```

- Run the script
- Check that the figure is created

## (optional) Exercise 2: displaying a `pandas` table

In this exercise, we will again use 
[the 'diamonds' dataset (as a comma-separated file)](diamonds.csv):
a dataset about diamonds.

This dataset contains information about more than fifty thousand diamonds.
Two such features are the weight (in carats) and the price (in USD).
Here we want to use an image to display the relationship between these two.

- Use `pandas` to read the dataset and use `seaborn`
  to create a scatter plot from that data. Put the diamond weight
  on the x-axis and the diamond price on the y-axis.

:::{admonition} Answer
:class: dropdown

Here is a simple solution
(simplified from [this script](seaborn_exercise.py)):

```python
import pandas as pd
import seaborn as sns
table = pd.read_csv("diamonds.csv")

scatter_plot = sns.relplot(
    data = table, x = "carat", y = "price"
)
scatter_plot.savefig("seaborn_exercise.png") 
```

This will look like this:

![](seaborn_exercise.png)

:::


## (optional) Exercise 3: making the plot pretty

## External links

- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
