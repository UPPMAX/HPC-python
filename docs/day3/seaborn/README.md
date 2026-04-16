
<!-- From https://docs.readthedocs.com/platform/stable/guides/cross-referencing-with-sphinx.html#explicit-targets -->
(seaborn)=

# `seaborn`

![The seaborn logo](logo-wide-lightbg.svg)

> [The `seaborn` logo](https://seaborn.pydata.org/citing.html)

:::{admonition} Learning outcomes
:class: note

At the end of this sessions, learners ...

- have practiced using the documentation of favorite HPC cluster
- understand why `seaborn` is important
- have run Python code that uses `seaborn`
- have run Python code that uses `seaborn` to display data from a `pandas` table
:::

## What is `seaborn`?

`seaborn` allows you to create figures:

```python
import seaborn as sns

y = [0, 1, 4, 9, 16]
sns.set_theme()
sns.lineplot(x = range(len(y)), y = y).figure.show()
```

Which shows:

![A minimal `seaborn` plot](what_is_seaborn.png)

:::{admonition} Why is `seaborn` imported as `sns`?
:class: dropdown

From [the `seaborn` FAQ](https://seaborn.pydata.org/faq.html#why-is-seaborn-imported-as-sns):

> This is an obscure reference to
> [the namesake](https://pbs.twimg.com/media/C3C6q1ZUYAALXX0.jpg)
> of the library, but you can also think of it as “seaborn name space”.

:::

## Why `seaborn` is important

`seaborn` is one of the most popular Python plotting libraries.
It can be used to create publication-quality figures and
[the `seaborn` plot gallery](https://seaborn.pydata.org/examples/index.html)
shows that most plot types are present.

## Exercises

## Loading `seaborn`

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

Create a script called `seaborn_exercise_1.py`,
with the following content:

```python
import seaborn as sns
y = [0, 1, 4, 9, 16]
sns.lineplot(x = range(len(y)), y = y).figure.savefig("seaborn_exercise_1.png")
```

Run the script.

:::{admonition} Answer: how to run the script
:class: dropdown

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|How to run the script
-----------|-----------------------------------------------------------------------------------------------------------------------
Alvis      |`python seaborn_exercise_1.py`
COSMOS     |`python seaborn_exercise_1.py`
Dardel     |`python3 seaborn_exercise_1.py`
Kebnekaise |`python seaborn_exercise_1.py`
Pelle      |`python seaborn_exercise_1.py`
Tetralith  |`python seaborn_exercise_1.py`

<!-- markdownlint-enable MD013 -->

:::

Check that the figure is created.

![`seaborn` exercise 1](seaborn_exercise_1.png)

:::{admonition} Answer: how to check that the figure is created
:class: dropdown

There are many ways to do so:

- **Download the file to your local computer**:
  You can download the file to your local computer.
  Then use your favorite way to view this image.
- **View in a remote desktop environment**:
  Your favorite HPC cluster has a remote desktop environment, which
  is a visual/graphical environment that is intuitive to use.
  There, for example, use the file explorer to find
  the file, then double-click it to do display it
- **View from a console environment that has X-forwarding enabled**:
  Use the same procedure as on
  [the 'HPC Python' course Day 1: view a plot](https://uppmax.github.io/naiss_intro_python/sessions/working_with_graphics/#exercise-2-optional-view-the-plot).

<!-- markdownlint-enable MD013 -->

:::

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

![`seaborn` exercise result](seaborn_exercise.png)

:::

## (optional) Exercise 3: making the plot pretty

Use [the `seaborn` documentation](https://seaborn.pydata.org/)
to improve the plot, for example:

- Add a title
- Add titles to the axes
- Add a linear trendline
- Whatever you like

## External links

- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
