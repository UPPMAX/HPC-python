
<!-- From https://docs.readthedocs.com/platform/stable/guides/cross-referencing-with-sphinx.html#explicit-targets -->
(pandas)=

# Pandas

:::{admonition} Learning outcomes
:class: note

At the end of this sessions, learners ...

- understand why Pandas is important
- have run Python code that uses Pandas
- have read a comma-separated file using Pandas
- have saved a table as a comma-separated file using Pandas
- have seen the effect of the `index` argument when saving a table
- have tried out some of the operation at the
  [the pandas page '10 minutes to pandas'](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html)
:::

## Loading Pandas

- Use the documentation of the HPC cluster you work on


- Find the software module to load the package. Use either
  the documentation of the HPC center, or use the module system

:::{admonition} Answer: where is the pandas documentation?
:class: dropdown

+------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| HPC cluster|URL to documentation                                                                                                                          |
+============+==============================================================================================================================================+
| Alvis      |`Here <https://www.c3se.chalmers.se/documentation/module_system/python/#numpy-and-scipy>`__                                                   |
+------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| Bianca     |`Here <https://docs.uppmax.uu.se/software/tensorflow/#tensorflow-as-a-python-package-for-cpu>`__                                              |
+------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| COSMOS     |`Here <https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/>`__                                                  |
+------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| Dardel     |`Here <https://support.pdc.kth.se/doc/applications/tensorflow/>`__, but it is irrelevant                                                      |
+------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| Kebnekaise |`Here <https://docs.hpc2n.umu.se/software/apps/#scipy>`__                                                                                     |
+------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| LUMI       |`Has no software modules <https://docs.lumi-supercomputer.eu/software/installing/python/#use-an-existing-container>`__                        |
+            +----------------------------------------------------------------------------------------------------------------------------------------------+
|            |`Use the thanard/matplotlib container <https://hub.docker.com/r/thanard/matplotlib>`__                                                        |
+------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| Pelle      |`Python bundles <https://docs.uppmax.uu.se/software/python_bundles/#pytorch>`__                                                               |
+------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| Tetralith  |`Here <https://www.nsc.liu.se/software/python/>`__                                                                                            |
+------------+----------------------------------------------------------------------------------------------------------------------------------------------+

:::


- Load the software module to use pandas

:::{admonition} Answer
:class: dropdown

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|How to load Pandas
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

Great, you've read a comma-separated file! Next step is to save it.
Saving it is straightforward, except for one thing: there is a function
argument called `index`. Here we'll find out what it is.

Add the following code to this Python script:

```python
table.to_csv("my_new_file_without_index.csv", index = False)
table.to_csv("my_new_file_with_index.csv", index = True)
```

Run the code. What is the difference between the two created files?

## Exercise 3: working with tabular data

There are many things one can do with tabular data.
Pandas has an overview at
[the pandas page '10 minutes to pandas'](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html).

```python
data = pd.Series([0.25, 0.5, 0.75, 1.0])
data = pd.Series({2:'a', 1:'b', 3:'c'})
print(data.values)
```

Data frame:

```python
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
population
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
area
states = pd.DataFrame({'population': population,
                       'area': area})
states
```

## Done?

Go to [the session about Matplotlib](../matplotlib/README.md)

## External links

- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
