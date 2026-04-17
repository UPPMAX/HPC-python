
<!-- From https://docs.readthedocs.com/platform/stable/guides/cross-referencing-with-sphinx.html#explicit-targets -->
(matplotlib)=

# `matplotlib`

![The matplotlib logo](matplotlib_logo.png)

> [The `matplotlib` logo](https://matplotlib.org/stable/gallery/misc/logos2.html)

:::{admonition} Learning outcomes
:class: note

At the end of this sessions, learners ...

- understand why `matplotlib` is important
- have run Python code that uses `matplotlib`
- have created a plot with `matplotlib`
- (optional) have created a plot with `matplotlib` from a `pandas` table

:::

## What is `matplotlib`?

`matplotlib` allows you to create figures:

```python
import matplotlib.pyplot as plt
plt.plot([0, 1, 4, 9, 16])
plt.show()
```

Which shows:

![A minimal `matplotlib` plot](what_is_matplotlib.png)

## Why `matplotlib` is important

`matplotlib` is one of the most popular Python plotting libraries.
It can be used to create publication-quality figures and
[the `matplotlib` plot types overview](https://matplotlib.org/stable/plot_types/index.html)
shows that most plot types are present.

:::{admonition} How popular is `matplotlib`?
:class: dropdown

`matplotlib` is not popular enough to be in
[the `PyPI` top 20](https://pypistats.org/top).

However, at
[the `matplotlib` PyPI statistics page](https://pypistats.org/packages/matplotlib)
we see that it has around 200 million downloads per month. 
As the number 20 package has around 800 million downloads per month,
we can infer that it is not all too unpopular.

:::

## Exercises

## Exercise 1: minimal code

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

:::{admonition} Answer: where is the `matplotlib` documentation?
:class: dropdown

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|HPC cluster-specific `matplotlib` documentation
-----------|-------------------------------------------------------------------------------------------------------------------
Alvis      |[`matplotlib` documentation](https://www.c3se.chalmers.se/documentation/module_system/modules/#to-search-for-keywords-in-a-module-such-as-extensions-in-bundles) [No relevant documentation]
Bianca     |[`matplotlib` documentation](https://docs.uppmax.uu.se/software/python_bundles/#matplotlib)
COSMOS     |[`matplotlib` documentation](https://lunarc-documentation.readthedocs.io/en/latest/manual/manual_modules/#loading-packages) [No relevant documentation]
Dardel     |[`matplotlib` documentation](https://support.pdc.kth.se/doc/basics/quickstart/#the-lmod-module-system) [No relevant documentation]
Kebnekaise |[`matplotlib` documentation](https://docs.hpc2n.umu.se/software/libs/matplotlib/)
Pelle      |[`matplotlib` documentation](https://docs.uppmax.uu.se/software/python_bundles/#matplotlib)
Tetralith  |[`matplotlib` documentation](https://www.nsc.liu.se/software/catalogue/tetralith/modules/python.html)

<!-- markdownlint-enable MD013 -->

:::

- Load the software module to use `matplotlib`

:::{admonition} Answer: how to load the `matplotlib` software module
:class: dropdown

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|How to load Matplotlib
-----------|-------------------------------------------------------------------------------------------------------------------
Alvis      |`module load matplotlib/3.9.2-gfbf-2024a` <!-- TODO -->
COSMOS     |`module load matplotlib/3.8.2` (avoid version `3.9.2`!) <!-- :-) 2026-04-17 -->
Dardel     |`module load PDC/23.12 cray-python/3.11.5 matplotlib/3.8.2-cpeGNU-23.12` <!-- TODO -->
Kebnekaise |`module load matplotlib/3.8.2` <!-- TODO -->
Pelle      |`module load matplotlib/3.9.2-gfbf-2024a` <!-- :-) 2026-04-17 -->
Tetralith  |`module load Python/3.10.4-env-hpc1-gcc-2022a-eb` <!-- :-) 2026-04-17 -->

<!-- markdownlint-enable MD013 -->

:::

Create a script called `matplotlib_exercise.py`
with the following code:

```python
import matplotlib
print(matplotlib.__version__)
```

Load the modules needed and run the script.

What do you see?

:::{admonition} Answer: how does that look like?
:class: dropdown

The output looks similar to this:

```text
3.6.3
```

:::

Even though the code shows nothing directly useful,
why is this a useful exercise anyways?

:::{admonition} Answer
:class: dropdown

This is a useful exercise,
because it proves that you have successfully loaded/installed
`matplotlib`.

:::

## Exercise 2: a minimal plot


- Create a script with the following code:

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

# plt.show()
plt.figure().savefig('my_figure.png')
```

- Run the script
- Check that the figure is created

## (optional) Exercise 3: displaying a `pandas` table

In this exercise, we will again use
[the 'diamonds' dataset (as a comma-separated file)](../pandas/diamonds.csv):
a dataset about diamonds.

This dataset contains information about more than fifty thousand diamonds.
Two such features are the weight (in carats) and the price (in USD).
Here we want to use an image to display the relationship between these two.

- Use `pandas` to read the dataset and use `matplotlib`
  to create a scatter plot from that data. Put the diamond weight
  on the x-axis and the diamond price on the y-axis.

:::{admonition} Answer
:class: dropdown

Here is a simple solution
(simplified from [this script](matplotlib_exercise.py)):

```python
import pandas as pd
import matplotlib.pyplot as plt
table = pd.read_csv("diamonds.csv")

plt.scatter(table["carat"], table["price"])
plt.savefig("matplotlib_exercise.png")
```

This will look like this:

![Result of the `matplotlib` exercise](matplotlib_exercise.png)

:::

## (optional) Exercise 4: making the plot pretty

Use [the `matplotlib` documentation](https://matplotlib.org/stable/index.html)
to improve the plot, for example:

- Add a title
- Add titles to the axes
- Add a linear trendline
- Whatever you like

## (optional) Exercise 5: should I use `matplotlib` or `seaborn`?

Search the academic literature to answer the question
if you should use `matplotlib` or `seaborn`,
for example
[by searching Google Scholar for 'matplotlib versus seaborn'](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=matplotlib+versus+seaborn&btnG=).

Which paper will you find?

:::{admonition} Answer
:class: dropdown

You will find the paper `[Sial et al., 2021]`
(see below for the complete reference)

:::

What does the paper conclude, regarding using `matplotlib` or `seaborn`?

:::{admonition} Answer
:class: dropdown

Here is a quote from the conclusion of `[Sial et al., 2021]`:

> It has been identified that if a data scientist
> wants to visualize the large chunks of datasets then seaborn
> will be a better option, but if you are looking for basic
> visualization patterns then matplotlib would be a better
> choice for beginners and starters in the field of data
> visualization & computational modelling

:::

## Done?

Go to [the session about `seaborn`](../seaborn/README.md)

## External links

- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

## References

- `[Sial et al., 2021]` Sial, Ali Hassan, Syed Yahya Shah Rashdi, and Abdul Hafeez Khan. "Comparative analysis of data visualization libraries Matplotlib and Seaborn in Python." International Journal 10.1 (2021): 277-281.
