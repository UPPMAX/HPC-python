###############
Intro to Pandas
###############

**Pandas**, short for PANel Data AnalysiS, is a Python data library for cleaning, organizing, and statistically analyzing moderately large ($lesssim3$ GiB) data sets. It was originally developed for analyzing and modelling financial records (panel data) over time, and has since expanded into a package rivaling SciPy in the number and complexity of available functions. Pandas offers:

* Explicit, automatic data alignment: all entries have corresponding row and column labels/indexes.
* Easy methods to add, remove, transform, compare, broadcast, and aggregate data within and across data structures.
* Data structures that support any mix of numerical, string, list, Boolean, and datetime datatypes.
* I/O interfaces that support a wide variety of text, binary, and database formats, including Excel, JSON, HDF5, NetCDF, and SQLite.
* Hundreds of built-in functions for cleaning, organizing, and statistical analysis, plus support for user-defined functions.
* A simple interface with the Seaborn plotting library, and increasingly also Matplotlib.
* Easy multi-threading with Numba.

**Limitations.** Pandas alone has somewhat limited support for parallelization, N-dimensional data structures, and datasets much larger than 3 GiB. Fortunately, there are packages like ``dask`` and ``polars`` that can help. In partcular, ``dask`` will be covered in a later lecture in this workshop. There is also the ``xarray`` package that provides many similar functions to Pandas for higher-dimensional data structures, but that is outside the scope of this workshop.

Load and Run
------------

.. tabs::

  .. tab:: HPC2N
     
     Pandas, like NumPy, has been part of the SciPy-bundle module since 2020. Use ``ml spider SciPy-bundle`` to see which versions are available and how to load them.

     .. important::
    
        Pandas requires Python 3.8.x and newer. Do not use SciPy-bundles for Python 2.7.x!


     As of 27-11-2024, the output of ``ml spider SciPy-bundle`` on Kebnekaise is:

     .. code-block:: console

        ----------------------------------------------------------------------------
          SciPy-bundle:
        ----------------------------------------------------------------------------
            Description:
              Bundle of Python packages for scientific software
        
             Versions:
                SciPy-bundle/2019.03
                SciPy-bundle/2019.10-Python-2.7.16
                SciPy-bundle/2019.10-Python-3.7.4
                SciPy-bundle/2020.03-Python-2.7.18
                SciPy-bundle/2020.03-Python-3.8.2
                SciPy-bundle/2020.11-Python-2.7.18
                SciPy-bundle/2020.11
                SciPy-bundle/2021.05
                SciPy-bundle/2021.10-Python-2.7.18
                SciPy-bundle/2021.10
                SciPy-bundle/2022.05
                SciPy-bundle/2023.02
                SciPy-bundle/2023.07-Python-3.8.6
                SciPy-bundle/2023.07
                SciPy-bundle/2023.11
          ----------------------------------------------------------------------------
            For detailed information about a specific "SciPy-bundle" package (including how to load the modules) use the module's full name.
            Note that names that have a trailing (E) are extensions provided by other modules.
            For example:
          
               $ module spider SciPy-bundle/2023.11
          ----------------------------------------------------------------------------


  .. tab:: LUNARC

     On the LUNARC HPC Desktop, all versions of Jupyter and Spyder load Pandas, NumPy, SciPy, Matplotlib, Seaborn, and many other Python packages automatically, so you don't need to load any modules. 

     If you choose to work at the command line and opt not to use Anaconda3, you will need to load a SciPy-bundle to access Pandas. Use ``ml spider SciPy-bundle`` to see which versions are available, which Python versions they depend on, and how to load them.

     .. important::
    
        Pandas requires Python 3.8.x and newer. Do not use SciPy-bundles for Python 2.7.x!

     As of 27-11-2024, the output of ``ml spider SciPy-bundle`` on Cosmos is:

     .. code-block:: console

        ----------------------------------------------------------------------------
          SciPy-bundle:
        ----------------------------------------------------------------------------
            Description:
              Bundle of Python packages for scientific software
        
             Versions:
                SciPy-bundle/2020.11-Python-2.7.18
                SciPy-bundle/2020.11
                SciPy-bundle/2021.05
                SciPy-bundle/2021.10-Python-2.7.18
                SciPy-bundle/2021.10
                SciPy-bundle/2022.05
                SciPy-bundle/2023.02
                SciPy-bundle/2023.07
                SciPy-bundle/2023.11
                SciPy-bundle/2024.05
        
        ----------------------------------------------------------------------------
          For detailed information about a specific "SciPy-bundle" package (including ho
        w to load the modules) use the module's full name.
          Note that names that have a trailing (E) are extensions provided by other modu
        les.
          For example:
        
             $ module spider SciPy-bundle/2024.05
        ----------------------------------------------------------------------------


  .. tab:: UPPMAX

     On Rackham, Python versions 3.8 and newer include NumPy, Pandas, and Matplotlib. There is no need to load additional modules after loading your preferred Python version.

To know if Pandas is the right tool for your job, you can consult the flowchart below.

.. image:: ../img/when-to-use-pandas.png
   :width: 600 px


.. objectives:: You will learn...

   * What are the basic object classes, data types, and their most important attributes and methods
   * How to input/output Pandas data
   * How to inspect, clean, and sort data for later operations
   * How to perform basic operations - statistics, binary operators, vectorized math and string methods
   * What are GroupBy objects and their uses
   * How to compare data, implement complex and/or user-defined functions, and perform windowed operations
   * How to use or create time series data (if time allows)
   * Advanced topics (if time allows) - How to prep for ML/AI, what are memory-saving data types


We will also have a short sesion after this on plotting with Seaborn, a package for easily making publication-ready statistical plots with Pandas data structures.


Basic Data Types and Object Classes
------------------------------------

The main object classes of Pandas are ``Series`` and ``DataFrame``. There is also a separate object class called ``Index`` for the row indexes/labels and column labels, if applicable. Data that you load from file will mainly be loaded into either Series or DataFrames. Indexes are typically extracted later.

* ``pandas.Series(data, index=None, name=None, ...)`` instantiates a 1D array with customizable indexes (labels) attached to every entry for easy access, and optionally a name for later addition to a DataFrame as a column.

  - Indexes can be numbers (integer or float), strings, datetime objects, or even tuples. The default is 0-based integer indexing. Indexes are also themselves a Pandas data type.

* ``pandas.DataFrame(data, columns=None, index=None, ...)`` instantiates a 2D array where every column is a Series. All entries are accessible by column and row labels/indexes.

  - Any function that works with a DataFrame will work with a Series unless the function specifically requires column or index arguments.
  - Column labels and row indexes/labels can be safely (re)assigned as needed.

For the rest of this lesson, example DataFrames will be abbreviated as ``df`` and example Series will be abbreviated as ``ser`` in code snippets.


.. admonition:: Important Attributes

   The API reference in the `official Pandas documentation <https://pandas.pydata.org/docs/user_guide/index.html>`_ shows hundreds of methods and attributes for Series and DataFrames. The following is a list of the most important attributes and what they output.
   
   - ``df.index`` returns a list of row labels as an array of Pandas datatype ``Index``
   - ``df.columns`` returns a list of column labels as an array of Pandas datatype ``Index``
   - ``df.dtypes`` lists datatypes by column
   - ``df.shape`` gives a tuple of the number of rows and columns in ``df``
   - ``df.values`` returns ``df`` converted to a NumPy array (also applicable to ``df.columns`` and ``df.index``)
   

Pandas assigns the data in a Series and each column of a DataFrame a datatype based on built-in or NumPy datatypes or other formatting cues. The main Pandas datatypes are as follows.

* Numerical data are stored as ``float64`` or ``int64``. You can convert to 32-, 16-, and even 8-bit versions of either to save memory.
* The ``object`` datatype stores any of the built-in types ``str``, ``Bool``, ``list``, ``tuple``, and mixed data types. Malformed data are also often designated as ``object`` type.

  - A common indication that you need to clean your data is finding a column that you expected to be numeric assigned a datatype of ``object``.


A significant fraction of Pandas functions are devoted to time series data in particular, so there are several datatypes based on NumPy datetimes and timedeltas, as well as calendar functions from the ``datetime`` module. Unfortunately, we won't have time to cover those at any length.

Finally, there are some specialized datatypes for, e.g. saving on memory or performing windowed operations, including

* ``Categorical`` is a set-like datatype for non-numeric data with few unique values. The unique values are stored in the attribute ``.categories``, that are mapped to a number of low-bit-size integers, and those integers replace the actual values in the DataFrame as it is stored in memory, which can save a lot on memory usage.

* ``Interval`` is a datatype for tuples of bin edges, all of which must be open or closed on the same sides, usually output by Pandas discretizing functions.

* ``Sparse[float64, <omitted>]`` is a datatype based on the SciPy sparse matrices, where ``<omitted>`` can be NaN, 0, or any other missing value placeholder. This placeholder value is stored in the datatype, and the DataFrame itself is compressed in memory by not storing anything at the coordinates of the missing values. 

This is far from an exhaustive list.


.. note:: Index-Class Objects
   :class: dropdown
   
   Index-class objects, like those returned by ``df.columns`` and ``df.index``, are immutable, hashable sequences used to align data for easy access. All of the previously mentioned categorical, interval, and time series data types have a corresponding Index subclass. Indexes have many Series-like attributes and set-operation methods, but Index methods only return copies, whereas the same methods for DataFrames and Series might return either copies or views into the original depending on the method.
  

.. warning:: Nomenclature for Row and Column Labels
   
   Pandas documentation has inconsistent nomenclature for row and column labels/indexes: 
   
   - "Indexes" usually refer to just the row labels, but may sometimes refer to both row and column labels.
   - "Columns" may refer to the labels and contents of columns collectively, or only the labels.
   - Column labels, and rarely also row indexes, are sometimes called “Keys”, particularly in commands designed to mimic SQL functions.
   - A column label may be called a “name”, after the optional Series label.


Input/Output and Making DataFrames from Scratch
-----------------------------------------------

Most of the time, Series and DataFrames will be loaded from files, not made from scratch. The following table lists I/O functions for the most common data formats. Input and output functions are sometimes called readers and writers, respectively. The ``read_csv()`` is by far the most commonly used since it can read any text file with a specified delimiter (comma, tab, or otherwise). 

======  ========================================  ===================================================  =================================
Type    Data Description                          Reader                                               Writer
======  ========================================  ===================================================  =================================
text    CSV / ASCII text with standard delimiter  ``read_csv(path_or_url, sep=',', **kwargs)``         ``to_csv()``
text    Fixed-Width Text File                     ``read_fwf()``                                       N/A
text    JSON                                      ``read_json()``                                      ``to_json()``
text    HTML                                      ``read_html()``                                      ``to_html()``
text    LaTeX                                     N/A                                                  ``Styler.to_latex()``
text    XML                                       ``read_xml()``                                       ``to_xml()``
text    Local clipboard                           ``read_clipboard()``                                 ``to_clipboard()``
SQL     SQLite table or query                     ``read_sql()``                                       ``to_sql()``
SQL     Google BigQuery                           ``read_gbq()``                                       ``to_gbq()``
binary  Python Pickle Format                      ``read_pickle()``                                    ``to_pickle()``
binary  MS Excel                                  ``read_excel(path_or_url, sheet_name=0, **kwargs)``  ``to_excel(path, sheet_name=...)``
binary  OpenDocument                              ``read_excel(path_or_url, sheet_name=0, **kwargs)``  ``to_excel(path, engine="odf")``
binary  HDF5 Format                               ``read_hdf()``                                       ``to_hdf()``
binary  Apache Parquet                            ``read_parquet()``                                   ``to_parquet()``
======  ========================================  ===================================================  =================================

This is not a complete list, and most of these functions have several dozen possible kwargs. It is left to the reader to determine what kwargs are needed. As with NumPy's ``genfromtxt()`` function, most of the *text* readers above, and the excel reader, have kwargs that let you choose to load only some of the data.

As an example, if there was a CSV file called "exoplanets_5250_EarthUnits.csv" in your home directory, it could be opened as follows 

.. code-block:: python

    >>> import pandas as pd
    >>> df = pd.read_csv('exoplanets_5250_EarthUnits.csv',index_col=0)

The ``index_col=0`` part sets the first column as the row labels, and the reader functions take the first row as the list of column names by default. If you forget to set a column as the list of row indexes during import, you can do it later with ``df.set_index('column_name')``.

Building a DataFrame or Series from scratch is also easy. Lists and arrays can be converted directly to Series and DataFrames, respectively.

* Both ``pd.Series()`` and ``pd.DataFrame()`` have an ``index`` kwarg to assign a list of numbers, names, times, or other hashable keys to each row. 
* You can use the ``columns`` kwarg in ``pd.DataFrame()`` to assign a list of names to the columns of the table. The equivalent for ``pd.Series()`` is just ``name``, which only takes a single value and doesn't do anything unless you plan to join that Series to a larger DataFrame.
* Dictionaries and record arrays can be converted to DataFrames with ``pd.DataFrame.from_dict(myDict)`` and ``pd.DataFrame.from_records(myRecArray)``, respectively, and the keys will automatically be converted to column labels.

**Example**

.. jupyter-execute::

    import numpy as np
    import pandas as pd
    df = pd.DataFrame( np.random.randint(0,100, size=(4,4)), columns=['a','b','c','d'], index=['w','x','y','z'] )
    print(df)

It is also possible to convert DataFrames and Series to NumPy arrays (with or without the indexes), dictionaries, record arrays, or strings with the methods ``.to_numpy()``, ``.to_dict()``, ``to_records()``, and ``to_string()``.


Inspection, Cleaning, and Sorting
---------------------------------

Inspection
^^^^^^^^^^

The main data inspection functions for DataFrames (and Series) are as follows.

``df.head()`` prints first 5 rows of data with row and column labels.  ``df.tail()`` does same for last 5 rows. Both accept and integer argument to print a different number of rows.

``df.info()`` prints the number of rows with their first and last index values; titles, index numbers, valid data counts, and datatypes of columns; and the estimated size of ``df`` in memory. Don't rely on this memory estimate; it is only accurate for numerical columns.

``df.describe()`` prints summary statistics for all the numerical columns in ``df``.

``df.nunique()`` prints counts of the unique values in each column.

``df.value_counts()`` prints each unique value and the number of of occurrences for every combination of row and column values for as many of each as are selected (usually applied to just a couple of columns at a time at most)

``df.sample()`` randomly selects a given number of rows ``n=nrows``, or a decimal fraction ``frac`` of the total number of rows.

``df.nlargest(n, columns)`` and ``df.nsmallest(n, columns)`` take an integer ``n`` and a column name or list of column names to sort the table by, and then return the ``n`` rows with the largest or smallest values in the columns used for sorting. These functions do not return ``df`` sorted.

.. important:: The ``memory_usage()`` function

   ``df.memory_usage(deep=False)`` returns the estimated memory usage of each column. With the default ``deep=False``, the sum of the estimated memory size of all columns is the same as what is included with ``df.info()``, which is not accurate. However, with ``deep=True``, the sizes of strings and other non-numeric data are factored in, giving a much better estimate of the total size of ``df`` in memory.
  
   This is because numeric columns are fixed width in memory and can be stored contiguously, but object-type columns are variable in size, so only pointers can be stored at the location of the main DataFrame in memory. The strings that those pointers refer to are kept elsewhere. When ``deep=False``, or when the memory usage is estimated with df.info()``, the memory estimate includes all the numeric data but only the pointers to non-numeric data.


Data Selection Syntax
^^^^^^^^^^^^^^^^^^^^^

Here is a table of the syntax for how to select different subsets or cross-sections of a DataFrame

===================================  =================================================================================================================================
To Access...                         Syntax
===================================  =================================================================================================================================
1 column                             ``df['col_name']`` or ``df.col_name``
1 named row                          ``df.loc['row_name']``
1 row by index                       ``df.iloc[index]``
1 column by index (rarely used)      ``df.iloc[:,index]``
subset of columns                    ``df[['col0', 'col1', 'col2']]``
subset of named rows                 ``df.loc[['rowA','rowB','rowC']]``
subset of rows by index              ``df.iloc[i_m:i_n]`` or ``df.take([i_m, ..., i_n])`` where ``i_m`` and ``i_n`` are the m :sup:`th` and n :sup:`th` integer indexes
1 or more rows and columns by name   ``df.loc['row','col']`` or ``df.loc[['rowA','rowB', ...],['col0', 'col1', ...]]``
2 or more rows and columns by index  ``df.iloc[i_m:i_n, j_p:j_q]`` where i and j are row and column indexes, respectively
columns by name and rows by index    **You can mix ``.loc[]`` and ``.iloc[]`` for selection, but NOT assignment!**
===================================  ==================================================================================================================================

To select by conditions, any binary comparison operator (>, <, ==, =>, =<, !=) and most logical operators can be used inside the square brackets of ``df[...]``, ``df.loc[...]``, and ``df.iloc[...]`` with some conditions.

* Bitwise logical operators (``&``, ``|``, ``^``, ``~``) must be used in lieu of plain-English counterparts (``and``, ``or``, ``xor``, ``not``)
* When 2 or more conditions are specified, **each individual condition must be bracketed by parentheses** or code will raise TypeError
* The "is" operator does not work within ``.loc[]``. Use ``.isna()`` or ``.notna()`` to check for invalid data, and ``.isin()``, ``.notin()``, or ``.str.contains()`` to check for the presence of substrings.
