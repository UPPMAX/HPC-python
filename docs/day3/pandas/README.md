
<!-- From https://docs.readthedocs.com/platform/stable/guides/cross-referencing-with-sphinx.html#explicit-targets -->
(pandas)=

# `pandas`

![The pandas logo](pandas.svg)

> [The `pandas` logo](https://pandas.pydata.org/about/citing.html)

:::{admonition} Learning outcomes
:class: note

At the end of this sessions, learners ...

- have practiced using the documentation of favorite HPC cluster
- understand why `pandas` is important
- have run Python code that uses `pandas`
- (optional) have read a comma-separated file using `pandas`
- (optional) have saved a table as a comma-separated file using `pandas`
- (optional) have seen the effect of the `index` argument when saving a table
- (optional) have tried out some of the operation at the
  [the `pandas` page '10 minutes to pandas'](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html)
:::

## What is `pandas`?

From [the `pandas` homepage](https://pandas.pydata.org/):

> *pandas* is [an] [...]
> open source data analysis and manipulation tool [...]

It allows you to do work with/on data, for example,
you can turn this messy data ...

Country  |1952|1957|1962
---------|----|----|----
Albania  |-9  |-9  |-9
Argentina|-9  |-1  |-1

using this `pandas` code ...

```python
table = pd.read_csv("dem_score.csv")
table = table.melt(id_vars = ["country"])
```

into this tidy data, which is easier to work with:

Country  |Year|Democracy level
---------|----|---------------
Albania  |1952|-9
Albania  |1957|-9
Albania  |1962|-9
Argentina|1952|-9
Argentina|1957|-1
Argentina|1962|-1

`pandas` can do many other things, such as reshaping data (from
[the `pandas` cheat sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)):

![Reshaping functions, from the `pandas` cheat sheet](pandas_cheat_sheet_reshape.png)

## Why `pandas` is important

`pandas` is a popular Python package that allows you
to work with data and it gives you a *vocabulary*
(and the Python functions) to do so.

:::{admonition} How popular is `pandas`?
:class: dropdown

`pandas` is not popular enough to be in
[the `PyPI` top 20](https://pypistats.org/top).

However, at
[the `pandas` PyPI statistics page](https://pypistats.org/packages/pandas)
we see that it has more than 600 million downloads per month. 
As the number 20 package has around 800 million downloads per month,
we can infer that it is not all too unpopular.

:::

## Exercises

:::{admonition} Want to see the answers as a video?
:class: dropdown

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|YouTube video
-----------|----------------------------------------------
Alvis      |[YouTube video](https://youtu.be/ogPddCKCUuA)
Bianca     |.
COSMOS     |[YouTube video](https://youtu.be/Lj2osOWK3WU)
Dardel     |[YouTube video](https://youtu.be/pfql_NmZuIc)
Kebnekaise |.
Pelle      |[YouTube video](https://youtu.be/oaRV1M9mwPE)
Tetralith  |[YouTube video](https://youtu.be/C1LLyXHwtx4)


<!-- markdownlint-enable MD013 -->

:::

## Exercise 1: minimal code

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

In that documentation, find the software module to load
the `pandas` Python package.

:::{admonition} Answer: where is the `pandas` documentation?
:class: dropdown

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|HPC cluster-specific `pandas` documentation
-----------|-------------------------------------------------------------------------------------------------------------------
Alvis      |Has no documentation on how to load `pandas`
Bianca     |[`pandas` documentation](https://docs.uppmax.uu.se/software/python_bundles/#loading-the-pandas-module)
COSMOS     |Has no documentation on how to load `pandas`
Dardel     |Has no documentation on how to load `pandas`
Kebnekaise |[`pandas` documentation](https://docs.hpc2n.umu.se/software/apps/#scipy)
Pelle      |[`pandas` documentation](https://docs.uppmax.uu.se/software/python_bundles/#loading-the-pandas-module)
Tetralith  |[`pandas` documentation](https://www.nsc.liu.se/software/python)

<!-- markdownlint-enable MD013 -->

:::

In a terminal (on your HPC cluster), load the software module to use `pandas`.

:::{admonition} Answer: how to load the `pandas` software module
:class: dropdown

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|How to load the `pandas` software module
-----------|-------------------------------------------------------------------------------------------------------------------
Alvis      |`module load matplotlib/3.9.2-gfbf-2024a`
COSMOS     |`module load GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 matplotlib/3.8.2`
Dardel     |`module load python/3.12.3 ; pip3 install pandas`
Kebnekaise |`module load GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 Tkinter/3.11.3`
Pelle      |`module load SciPy-bundle/2024.05-gfbf-2024a`
Tetralith  |`module load Python/3.10.4-env-hpc1-gcc-2022a-eb`
<!-- markdownlint-enable MD013 -->

:::

On your HPC cluster, create a script called `pandas_exercise_1.py`
with the following code:

```python
import pandas
print(pandas.__version__)
```

Run the script.

:::{admonition} Answer: how to run the script
:class: dropdown

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|How to run the script
-----------|-----------------------------------------------------------------------------------------------------------------------
Alvis      |`python pandas_exercise_1.py`
COSMOS     |`python pandas_exercise_1.py`
Dardel     |`python3 pandas_exercise_1.py`
Kebnekaise |`python pandas_exercise_1.py`
Pelle      |`python pandas_exercise_1.py`
Tetralith  |`python pandas_exercise_1.py`

<!-- markdownlint-enable MD013 -->

:::

What do you see?

:::{admonition} Answer: how does that look like?
:class: dropdown

The output looks similar to this:

```text
3.0.1
```

:::

Even though the code shows nothing directly useful,
why is this a useful exercise anyways?

:::{admonition} Answer
:class: dropdown

This is a useful exercise,
because it proves that you have successfully loaded/installed
`pandas`.

:::


## (optional) Exercise 2: reading and saving a comma-separated file

In this exercise, we will first read
[the 'diamonds' dataset (as a comma-separated file)](diamonds.csv):
a dataset about diamonds.
It is described
[in the `ggplot2` (an R package) documentation](https://ggplot2.tidyverse.org/reference/diamonds.html).

Download this file to the same folder as where you are running your Python code.

:::{admonition} How do I do that?
:class: dropdown

There are many ways:

- Click on [the 'diamonds' dataset (as a comma-separated file)](diamonds.csv).
  This will take you to a webpage with the data.
  Right-click and do 'Save as' to save this file to your computer
- Download the file from the command-line:

```bash
wget https://raw.githubusercontent.com/UPPMAX/HPC-python/refs/heads/main/docs/day3/pandas/diamonds.csv
```

- Your favorite alternative way

:::


On your HPC cluster, create a script called `pandas_exercise_2.py`
with the following code:

```python
import pandas as pd

table = pd.read_csv("diamonds.csv")
print(table)
```

Run the script `pandas_exercise_2.py`.

:::{admonition} Answer: how to run the script
:class: dropdown

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|How to run the script
-----------|-----------------------------------------------------------------------------------------------------------------------
Alvis      |`python pandas_exercise_2.py`
COSMOS     |`python pandas_exercise_2.py`
Dardel     |`python3 pandas_exercise_2.py`
Kebnekaise |`python pandas_exercise_2.py`
Pelle      |`python pandas_exercise_2.py`
Tetralith  |`python pandas_exercise_2.py`

<!-- markdownlint-enable MD013 -->

:::

What does the script `pandas_exercise_2.py` do?

:::{admonition} Answer
:class: dropdown

It reads a comma-separated file into memory.

:::

Next step is to save it. Add the following code to `pandas_exercise_2.py`:

```python
table.to_csv("pandas_exercise_2.csv")
```

Again, run the script `pandas_exercise_2.py`.

:::{admonition} Answer: how to run the script
:class: dropdown

<!-- markdownlint-disable MD013 --><!-- Tables cannot be split up over lines, hence will break 80 characters per line -->

HPC cluster|How to run the script
-----------|-----------------------------------------------------------------------------------------------------------------------
Alvis      |`python pandas_exercise_2.py`
COSMOS     |`python pandas_exercise_2.py`
Dardel     |`python3 pandas_exercise_2.py`
Kebnekaise |`python pandas_exercise_2.py`
Pelle      |`python pandas_exercise_2.py`
Tetralith  |`python pandas_exercise_2.py`

<!-- markdownlint-enable MD013 -->

:::

Take a look at the file `pandas_exercise_2.csv`.
What has been added to the data?

:::{admonition} Answer
:class: dropdown

Of each row in the data, there has been an index added:

```text
,carat,cut,color,clarity,depth,table,price,x,y,z
0,0.23,Ideal,E,SI2,61.5,55.0,326,3.95,3.98,2.43
1,0.21,Premium,E,SI1,59.8,61.0,326,3.89,3.84,2.31
2,0.23,Good,E,VS1,56.9,65.0,327,4.05,4.07,2.31
3,0.29,Premium,I,VS2,62.4,58.0,334,4.2,4.23,2.63
4,0.31,Good,J,SI2,63.3,58.0,335,4.34,4.35,2.75
```

:::

In `pandas_exercise_2.py`, replace the last line by this version:

```python
table.to_csv("pandas_exercise_2.csv", index = False)
```

Run `pandas_exercise_2.py`. How does the data look like now?

:::{admonition} Answer
:class: dropdown

Now, the file looks like shown below, where there is no indexing anymore.

```text
carat,cut,color,clarity,depth,table,price,x,y,z
0.23,Ideal,E,SI2,61.5,55.0,326,3.95,3.98,2.43
0.21,Premium,E,SI1,59.8,61.0,326,3.89,3.84,2.31
0.23,Good,E,VS1,56.9,65.0,327,4.05,4.07,2.31
0.29,Premium,I,VS2,62.4,58.0,334,4.2,4.23,2.63
0.31,Good,J,SI2,63.3,58.0,335,4.34,4.35,2.75
```

:::

What seems to be the most useful way to save: with or without indexing?

:::{admonition} Answer
:class: dropdown

Typically, you will want to save without indexing.

:::

Why would `pandas` supply this option, to save with/without indexing?

:::{admonition} Answer
:class: dropdown

For backwards compatibility.

Indexing was a useful feature in the field
`pandas` was initially developed in,
so `pandas` always used indexing, with no way to disable this feature.

However, later it was found that indexing is not useful in other fields.

There were two options:

- Remove indexing from `pandas`
- Allow users to disable indexing

Removing indexing would cause old code to break, so this was decided
against.
Instead, it was decided to allow users to disable indexing when needed.

:::

## (optional) Exercise 3: tidy data

`pandas` shines when the data is tidy.

Search the web for 'What is tidy data?'. Is the `diamonds` dataset tidy? Why?

:::{admonition} Answer
:class: dropdown

I found the definition below from
[a `tidyr` (an R package) article](https://tidyr.tidyverse.org/articles/tidy-data.html#tidy-data):

In tidy data:

- Each variable is a column; each column is a variable.
- Each observation is a row; each row is an observation.
- Each value is a cell; each cell is a single value.

The `diamonds` dataset is tidy, because:

- Each feature of each single diamond has a column.
  Each feature is observed at more-or-less the same time
- Each diamond has its own row
- Each value in the table is indeed one value

:::

Now take a look at a dataset from [this book](https://moderndive.com)
called [`dem_score.csv`](https://moderndive.com/data/dem_score.csv).
This dataset shows the ratings of the level of democracy in
different countries spanning 1952 to 1992, where the minimum value of -10
corresponds to a highly autocratic nation whereas a value of 10 corresponds
to a highly democratic nation. Here is how it looks like:


```text
country,1952,1957,1962,1967,1972,1977,1982,1987,1992
Albania,-9,-9,-9,-9,-9,-9,-9,-9,5
Argentina,-9,-1,-1,-9,-9,-9,-8,8,7
Armenia,-9,-7,-7,-7,-7,-7,-7,-7,7
Australia,10,10,10,10,10,10,10,10,10
```

Is the `dem_score` dataset tidy? Why?

:::{admonition} Answer
:class: dropdown

I found the definition below from
[a `tidyr` (an R package) article](https://tidyr.tidyverse.org/articles/tidy-data.html#tidy-data):

In tidy data:

- Each variable is a column; each column is a variable.
- Each observation is a row; each row is an observation.
- Each value is a cell; each cell is a single value.

The [`dem_score.csv`](https://moderndive.com/data/dem_score.csv)
dataset is **not** tidy, because:

- For all expect the first column, these columns are *values*: they are
  values for the year the measurement was done.
- Each row contains multiple observations: per country, it shows
  the democratic index of 1952, the democratic index of 1953, etc.
- Each value in the table is indeed one value

:::

How would this data look like, would it be tidy?

:::{admonition} Answer
:class: dropdown

Here is how this data would look like, would it be tidy:

```text
country,year,democracy_level
Albania,1952,-9
Albania,1953,-9
Albania,1954,-9
Albania,1955,-9
```
:::

Create a Python script called `pandas_exercise_3.py`.
In that script, use `pandas` to read
[the `dem_score.csv` dataset](https://moderndive.com/data/dem_score.csv),
convert it to tidy data
and save it as `tidy_dem_scores.csv`.

For this use:

- [the `pandas` cheat sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf).
  Tip: the function you will need is [this one](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.melt.html#pandas.DataFrame.melt).
- Your favorite web search engine
- Your favorite AI

:::{admonition} Answer
:class: dropdown

Here is the minimal code to do so:

```python
import pandas as pd
table = pd.read_csv("dem_score.csv")
table = table.melt(id_vars = ["country"])
table.rename(columns = {"variable": "year", "value": "democratic_score"}, inplace = True)
table.to_csv("tidy_dem_scores.csv", index = False)
```

:::

## (optional) Exercise 4: what does `pandas` mean?

The word `pandas` is actually a shortened version of something.
Search the internet for what it stands for.
In which field did `pandas` originate?

:::{admonition} Answer
:class: dropdown

`pandas` is short for 'panel data'.
Panel data is a type of data set used in econometrics.
Econometrics is the field where `pandas` originated.
:::

## Done?

Go to [the session about `matplotlib`](../matplotlib/README.md)

## External links

- [The `pandas` cheat sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf).
- [the `pandas` page '10 minutes to `pandas`'](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html).
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
