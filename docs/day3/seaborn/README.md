
<!-- From https://docs.readthedocs.com/platform/stable/guides/cross-referencing-with-sphinx.html#explicit-targets -->
(seaborn)=

# `seaborn`

![The seaborn logo](logo-wide-lightbg.svg)

[The `seaborn` logo](https://seaborn.pydata.org/citing.html)


:::{admonition} Learning outcomes
:class: note

At the end of this sessions, learners ...

- have practiced using the documentation of favorite HPC cluster
- understand why `seaborn` is important
- have created a plot with `seaborn`
- (optional) have created a plot with `seaborn` from a `pandas` table
:::

## What is `seaborn`?

`seaborn` allows you to create figures:

```python
import seaborn as sns
y = [0, 1, 4, 9, 16]
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

:::{admonition} Why does this plot look identical to the `matplotlib` plot?
:class: dropdown

Because it is!

This plot is identical to the `matplotlib` plot in
[the session about `matplotlib`](../matplotlib/README.md),
because `seaborn` is built on top of `matplotlib`.
Hence, `seaborn` uses `matplotlib` for plotting.

:::

## Why `seaborn` is important

`seaborn` is one of the most popular Python plotting libraries.
It can be used to create publication-quality figures and
[the `seaborn` plot gallery](https://seaborn.pydata.org/examples/index.html)
shows that most plot types are present.

:::{admonition} How popular is `seaborn`?
:class: dropdown

`seaborn` is not popular enough to be in
[the `PyPI` top 20](https://pypistats.org/top).

However, at
[the `seaborn` PyPI statistics page](https://pypistats.org/packages/seaborn)
we see that it has around 20 million downloads per month. 
As the number 20 package has around 800 million downloads per month,
we can infer that it is not all too unpopular.

:::

## Exercises

:::{admonition} Want to see the answers as a video?
:class: dropdown

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|YouTube video
-----------|-------------------------------------------------------------------------------------------------------------------
Alvis      |.
Bianca     |.
COSMOS     |.
Dardel     |[YouTube video](https://youtu.be/ykSkToSHF_Y)
Kebnekaise |.
Pelle      |[YouTube video](https://youtu.be/xD56QJ0O_jM)
Tetralith  |.


<!-- markdownlint-enable MD013 -->

:::

## Exercise 1: a minimal `seaborn` program

Use the documentation of the HPC cluster you work on.

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

In that documentation, find the software module to load the package.

:::{admonition} Answer: where is the `seaborn` documentation?
:class: dropdown

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|HPC cluster-specific `seaborn` documentation
-----------|-------------------------------------------------------------------------------------------------------------------
Alvis      |[`seaborn` documentation](https://www.c3se.chalmers.se/documentation/module_system/python/#numpy-and-scipy)
Bianca     |[`seaborn` documentation](https://docs.uppmax.uu.se/software/python/) [Only shows module name]
COSMOS     |[`seaborn` documentation](https://lunarc-documentation.readthedocs.io/en/latest/software/installed_software/) [Only shows module name]
Dardel     |No documentation
Kebnekaise |[`seaborn` documentation](https://docs.hpc2n.umu.se/software/libs/Seaborn/)
Pelle      |[`seaborn` documentation](https://docs.uppmax.uu.se/software/seaborn/)
Tetralith  |[`seaborn` documentation](https://www.nsc.liu.se/software/python) No relevant documentation

<!-- markdownlint-enable MD013 -->

:::

In a terminal (on your HPC cluster), load the software module to use `seaborn`.

:::{admonition} Answer: how to load the `seaborn` software module
:class: dropdown

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|How to load Seaborn
-----------|-----------------------------------------------------------------------------------------------------------------------
Alvis      |`module load Seaborn/0.13.2-gfbf-2024a`<!-- TODO -->
COSMOS     |`module load GCC/13.2.0 Seaborn/0.13.2`<!-- 2026-04-17 -->
Dardel     |`module load python/3.12.3 ; pip3 install seaborn` <!-- 2026-04-17 -->
Kebnekaise |`module load GCC/13.2.0 Seaborn/0.13.2`<!-- :-) 2026-04-21 -->
Pelle      |`module load Seaborn/0.13.2-gfbf-2024a` <!-- :-) 2026-05-17 -->
Tetralith  |`module load Python/3.10.4-env-hpc1-gcc-2022a-eb ; pip install seaborn` <!-- :-) 2026-05-17 -->

<!-- markdownlint-enable MD013 -->

:::

On your HPC cluster, create a script called `seaborn_exercise_1.py`
with the following code:

```python
import seaborn
print(seaborn.__version__)
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

What do you see?

:::{admonition} Answer: how does that look like?
:class: dropdown

The output looks similar to this:

```text
0.13.2
```

:::

Even though the code shows nothing directly useful,
why is this a useful exercise anyways?

:::{admonition} Answer
:class: dropdown

This is a useful exercise,
because it proves that you have successfully loaded/installed
`seaborn`.

:::

## Exercise 2: a minimal plot

On your HPC cluster,
create a script called `seaborn_exercise_2.py`,
with the following content:

```python
import seaborn as sns
y = [0, 1, 4, 9, 16]
sns.lineplot(x = range(len(y)), y = y).figure.savefig("seaborn_exercise_2.png")
```

Run the script.

:::{admonition} Answer: how to run the script
:class: dropdown

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|How to run the script
-----------|-----------------------------------------------------------------------------------------------------------------------
Alvis      |`python seaborn_exercise_2.py`
COSMOS     |`python seaborn_exercise_2.py`
Dardel     |`python3 seaborn_exercise_2.py`
Kebnekaise |`python seaborn_exercise_2.py`
Pelle      |`python seaborn_exercise_2.py`
Tetralith  |`python seaborn_exercise_2.py`

<!-- markdownlint-enable MD013 -->

:::

Check that the figure is created.

![`seaborn` exercise 2](seaborn_exercise_2.png)

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

## (optional) Exercise 3: displaying a `pandas` table

In this exercise, we will again use
the 'diamonds' dataset (as a comma-separated file), [click to download](https://uppmax.github.io/HPC-python/_downloads/2c3f44b9035e3effc9e3b2854f37f1f0/diamonds.csv):
a dataset about diamonds.

This dataset contains information about more than fifty thousand diamonds.
Two such features are the weight (in carats) and the price (in USD).
Here we want to use an image to display the relationship between these two.

On your HPC cluster,
create a script called `seaborn_exercise_3.py`. In that script:

- Use `pandas` to read the dataset
- Use `seaborn` to create a scatter plot from that data.
  Put the diamond weight
  on the x-axis and the diamond price on the y-axis.
  Use
  [the `seaborn` documentation](https://seaborn.pydata.org/),
  a search engine or an AI chatbot for the answer.
- save the plot as `seaborn_exercise_3.png`
  Use
  [the `matplotlib` documentation](https://seaborn.pydata.org/),
  a search engine or an AI chatbot for the answer.

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

## (optional) Exercise 4: making the plot pretty

Use [the `seaborn` documentation](https://seaborn.pydata.org/)
to improve the plot, for example:

- Add a title
- Add titles to the axes
- Add a linear trendline
- Whatever you like

## External links

- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
