
<!-- From https://docs.readthedocs.com/platform/stable/guides/cross-referencing-with-sphinx.html#explicit-targets -->
(pandas)=

# Pandas

![The pandas logo](pandas.svg)

> [The `pandas` logo](https://pandas.pydata.org/about/citing.html) 

:::{admonition} Learning outcomes
:class: note

At the end of this sessions, learners ...

- have practiced using the documentation of favorite HPC cluster
- understand why Pandas is important
- have run Python code that uses Pandas
- have read a comma-separated file using Pandas
- have saved a table as a comma-separated file using Pandas
- have seen the effect of the `index` argument when saving a table
- have tried out some of the operation at the
  [the pandas page '10 minutes to pandas'](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html)
:::

## Why `pandas` is important

From [the pandas homepage](https://pandas.pydata.org/):

> pandas is a fast, powerful, flexible and easy to use
> open source data analysis and manipulation tool,
> built on top of the Python programming language.



## Loading Pandas


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

:::{admonition} Answer: where is the `pandas` documentation?
:class: dropdown

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|HPC cluster-specific `pandas` documentation
-----------|-------------------------------------------------------------------------------------------------------------------
Alvis      |[`pandas` documentation](https://www.c3se.chalmers.se/documentation/module_system/python/#numpy-and-scipy)
Bianca     |[`pandas` documentation](https://docs.uppmax.uu.se/software/tensorflow/#tensorflow-as-a-python-package-for-cpu)
COSMOS     |[`pandas` documentation](https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python)
Dardel     |[`pandas` documentation](https://support.pdc.kth.se/doc/applications/tensorflow) (irrelevant)
Kebnekaise |[`pandas` documentation](https://docs.hpc2n.umu.se/software/apps/#scipy)
Pelle      |[`pandas` documentation](https://docs.uppmax.uu.se/software/python_bundles/#pytorch)
Tetralith  |[`pandas` documentation](https://www.nsc.liu.se/software/python)

<!-- markdownlint-enable MD013 -->

:::


- Load the software module to use `pandas`

:::{admonition} Answer: how to load the `pandas` software module
:class: dropdown

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|How to load the `pandas` software module
-----------|-------------------------------------------------------------------------------------------------------------------
Alvis      |`module load matplotlib/3.9.2-gfbf-2024a`
COSMOS     |`module load GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 matplotlib/3.8.2`
Dardel     |`module load cray-python/3.11.7`
Kebnekaise |`module load GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 Tkinter/3.11.3`
Pelle      |`module load SciPy-bundle/2024.05-gfbf-2024a` :-)
Tetralith  |`module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 JupyterLab/4.2.0` :-)
<!-- markdownlint-enable MD013 -->

:::


## Exercises

## Exercise 1: minimal code

Get this code to run:

```python
import pandas
print(pandas.__version__)
```

What do you see?

:::{admonition} Answer: how does that look like?
:class: dropdown

```python
TODO: text output here
```

:::

## Exercise 2: reading and saving a comma-separated file

In this exercise, we will first read
[the 'diamonds' dataset (as a comma-separated file)](diamonds.csv):
a dataset about diamonds.

Download this file to the same folder as where you are running your Python code.

Run and read the following code:

```python
import pandas as pd

table = pd.read_csv("diamonds.csv")
print(table)
```

:::{admonition} Answer: how does that look like?
:class: dropdown

```python
TODO: text output here
```
:::


Great, you've read a comma-separated file! Next step is to save it.
Saving it is straightforward, except for one thing: there is a function
argument called `index`. Here we'll find out what it is.

Add the following code to this Python script:

```python
table.to_csv("my_new_file_without_index.csv", index = False)
table.to_csv("my_new_file_with_index.csv", index = True)
```

Run the code.

:::{admonition} Answer: how does that look like?
:class: dropdown

```python
TODO: text output here
```
:::

What is the difference between the two created files?

:::{admonition} Answer: what is the difference between the two created files?
:class: dropdown

```python
TODO: text output here
```
:::

## Exercise 3: working with tabular data

There are many things one can do with tabular data.
Pandas has an overview at
[the pandas page '10 minutes to pandas'](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html).

## Done?

Go to [the session about `matplotlib`](../matplotlib/README.md)

## External links

- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
