
<!-- From https://docs.readthedocs.com/platform/stable/guides/cross-referencing-with-sphinx.html#explicit-targets -->
(pandas)=

# Pandas

:::{admonition} Learning outcomes
:class: note

At the end of this sessions, learners ...

- understand why Pandas is important
- have run Python code that uses Pandas
:::

## Loading Pandas

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|How to load Pandas
-----------|-------------------------------------------------------------------------------------------------------------------
Alvis      |`module load matplotlib/3.9.2-gfbf-2024a`
COSMOS     |`module load GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 matplotlib/3.8.2`
Dardel     |`module load cray-python/3.11.7`
Kebnekaise |`module load GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 Tkinter/3.11.3`
Pelle      |`module load SciPy-bundle/2024.05-gfbf-2024a`
Tetralith  |`module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 JupyterLab/4.2.0`

<!-- markdownlint-enable MD013 -->

## Exercises

## Exercise 1: minimal code

Get this code to run:

```python
import pandas
print(pandas.__version__)
```

## Exercise 2: reading a comma-seperated file

Download


```python
pd.read_csv("diamonds.csv")
# 
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
