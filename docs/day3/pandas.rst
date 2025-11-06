###############
Intro to Pandas
###############

**Pandas**, short for PANel Data AnalysiS, is a Python data library for cleaning, organizing, and statistically analyzing moderately large (:math:`\lesssim3` GiB) data sets. It was originally developed for analyzing and modelling financial records (panel data) over time, and has since expanded into a package rivaling SciPy in the number and complexity of available functions. Pandas offers:

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

Pandas has been part of the SciPy-bundle module (which also contains NumPy) since 2020, so at most HPC resources, you should use ``ml spider SciPy-bundle`` to see which versions are available and how to load them.

.. important::

   Pandas requires Python 3.8.x and newer. Do not use SciPy-bundles for Python 2.7.x!

Some facilities also have Anaconda, which typically includes Pandas, JupyterLab, NumPy, SciPy, and many other popular packages. However, if there is a Python package you want that is not included, you will typically have to build your own environment to install it, and extra steps may be required to use that conda environment in a development tool like Jupyter Lab.


.. tabs::

   .. tab:: HPC2N
     
      .. important::

         For this session, you should load

         .. code-block:: console
        
            ml GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 Tkinter/3.11.3

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

      .. important::

         For this session, you should load

         .. code-block:: console
        
            ml GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 matplotlib/3.8.2
     
      On the LUNARC HPC Desktop, all versions of Jupyter and Spyder load Pandas, NumPy, SciPy, Matplotlib, Seaborn, and many other Python packages automatically, so you don't need to load any modules. 

      If you work at the command line and choose not to use Anaconda3, you will need to load a SciPy-bundle to access Pandas. Use ``ml spider SciPy-bundle`` to see which versions are available, which Python versions they depend on, and how to load them.

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

      .. important::

         For this session, you should load

         .. code-block:: console
        
            module load python/3.11.8
     
      On Rackham, Python versions 3.8 and newer include NumPy, Pandas, and Matplotlib. There is no need to load additional modules after loading your preferred Python version.


   .. tab:: Tetralith (NSC)
     
      .. important::

         For this session, you should load

         .. code-block:: console
        
            module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 JupyterLab/4.2.0
         
         Pandas, like NumPy, has typically been part of the SciPy-bundle module since 2020. Use ``ml spider SciPy-bundle`` to see which versions are available and how to load them.


   .. tab:: Dardel (PDC)

      - Jupyter Lab is only available on Dardel via ThinLinc. 
      - As there are only 30 ThinLinc licenses available at this time, we recommend that you work on the exercises with a local installation on a personal computer. 
      - Do not trust that a ThinLinc session will be available or that On-Demand applications run therein will start in time for you to keep up (it is not unusual for wait times to be longer than the requested walltime). 
      - The exercises were written to work on a regular laptop. If you must work on Dardel, follow the steps below, and view the `exercises <https://github.com/UPPMAX/HPC-python/blob/main/docs/day3/HPC-Pandas-exercises.ipynb>`_ and `solutions <https://github.com/UPPMAX/HPC-python/blob/main/docs/day3/HPC-Pandas-exercises-solutions.ipynb>`_ in the GitHub repository (they should render correctly).

      .. important::

         For this session, you could load

         .. code-block:: console
        
            ml cray-python/3.11.7
     
      On Dardel, all cray-python versions include NumPy, SciPy, and Pandas, and do not require any prerequisites. Matplotlib is separate and will have to be loaded using ``ml PDC/23.12 matplotlib/3.8.2-cpeGNU-23.12``, where PDC/23.12 is a prerequisite. The versions available for for both cray-python and matplotlib are limited because it is generally assumed that most users will build their own environments, but the installed versions are fine for this course.


     - ALTERNATIVE IF THINLINC IS AVAILABLE
     - Start Jupyter from the Menu and it will work! 

          - Default Anaconda 3 has all packages needed for this lesson

     - OR USE SPYDER:
          - start interactive session

          .. code-block:: console 

             salloc --ntasks=4 -t 0:30:00 -p shared --qos=normal -A naiss2025-22-403
             salloc: Pending job allocation 9102757
             salloc: job 9102757 queued and waiting for resources
             salloc: job 9102757 has been allocated resources
             salloc: Granted job allocation 9102757
             salloc: Waiting for resource configuration
             salloc: Nodes nid001057 are ready for job

          We need to ssh to the specific node, like

          .. code-block:: console 

             ssh nid001057


          Use the conda env you created in Exercise 2 in `Use isolated environemnts <https://uppmax.github.io/HPC-python/day2/use_isolated_environments.html#exercises>`_

          .. code-block:: console

             ml PDC/23.12
             ml miniconda3/24.7.1-0-cpeGNU-23.12
             export CONDA_ENVS_PATH="/cfs/klemming/projects/supr/hpc-python-spring-naiss/$USER/"
             export CONDA_PKG_DIRS="/cfs/klemming/projects/supr/hpc-python-spring-naiss/$USER/"
             source activate spyder-env
             conda install matplotlib pandas seaborn
             spyder %


To know if Pandas is the right tool for your job, you can consult the flowchart below.

.. image:: ../img/when-to-use-pandas.png
   :width: 600 px


.. objectives:: You will learn...

   * TK

We will also have a short session after this on plotting with Seaborn, a package for easily making publication-ready statistical plots with Pandas data structures.

Basic Data Types and Object Classes
-----------------------------------

The main object classes of Pandas are ``Series`` and ``DataFrame``. There is also a separate object class called ``Index`` for the row indexes/labels and column labels, if applicable. Data that you load from file will mainly be loaded into either Series or DataFrames. Indexes are typically extracted later.

* ``pandas.Series(data, index=None, name=None, ...)`` instantiates a 1D array with customizable indexes (labels) attached to every entry for easy access, and optionally a name for later addition to a DataFrame as a column.

  - Indexes can be numbers (integer or float), strings, datetime objects, or even tuples. The default is 0-based integer indexing. Indexes are also a Pandas data type (the data type of the row and column labels)

* ``pandas.DataFrame(data, columns=None, index=None, ...)`` instantiates a 2D array where every column is a Series. All entries are accessible by column and row labels/indexes.

  - Any function that works with a DataFrame will work with a Series unless the function specifically requires column arguments.
  - Column labels and row indexes/labels can be safely (re)assigned as needed.

For the rest of this lesson, example DataFrames will be abbreviated as ``df`` in code snippets (and example Series, if they appear, will be abbreviated as ``ser``).

.. admonition:: **Important Attributes**

   The API reference in the `official Pandas documentation <https://pandas.pydata.org/docs/user_guide/index.html>`_ shows hundreds of methods and attributes for Series and DataFrames. The following is a very brief list of the most important attributes and what they output.
   
   - ``df.index`` returns a list of **row labels** as an array of Pandas datatype ``Index``
   - ``df.columns`` returns a list of **column labels** as an array of Pandas datatype ``Index``
   - ``df.dtypes`` lists datatypes by column
   - ``df.shape`` gives a tuple of the number of rows and columns in ``df``
   - ``df.values`` returns ``df`` converted to a NumPy array (also applicable to ``df.columns`` and ``df.index``)
   
Pandas assigns the data in a Series and each column of a DataFrame a datatype based on built-in or NumPy datatypes or other formatting cues. Important Pandas datatypes include the following.

* Numerical data are stored as ``float64`` or ``int64``. You can convert to 32-, 16-, and even 8-bit versions of either to save memory.
* The ``object`` datatype stores any of the built-in types ``str``, ``Bool``, ``list``, ``tuple``, and mixed data types. Malformed data are also often designated as ``object`` type.

  - A common indication that you need to clean your data is finding a column that you expected to be numeric assigned a datatype of ``object``.

* Pandas has many functions devoted to time series, so there are several datatypes---``datetime``, ``timedelta``, and ``period``. The first two are based on `NumPy data types of the same name <https://numpy.org/devdocs/reference/arrays.datetime.html>`_ , and ``period`` is a time-interval type specified by a starting datetime and a recurrence rate. Unfortunately, we won't have time to cover these at depth.

There are also specialized datatypes for, e.g. saving on memory or performing windowed operations, including

* ``Categorical`` is a set-like datatype for non-numeric data with few unique values. The unique values are stored in the attribute ``.categories``, that are mapped to a number of low-bit-size integers, and those integers replace the actual values in the DataFrame as it is stored in memory, which can save a lot on memory usage.
* ``Interval`` is a datatype for tuples of bin edges, all of which must be open or closed on the same sides, usually output by Pandas discretizing functions.
* ``Sparse[float64, <omitted>]`` is a datatype based on the SciPy sparse matrices, where ``<omitted>`` can be NaN, 0, or any other missing value placeholder. This placeholder value is stored in the datatype, and the DataFrame itself is compressed in memory by not storing anything at the coordinates of the missing values. 

This is far from an exhaustive list.

Input/Output and Making DataFrames from Scratch
-----------------------------------------------

Most of the time, Series and DataFrames will be loaded from files, not made from scratch. The following table lists I/O functions for a few of the most common data formats. Input and output functions are sometimes called readers and writers, respectively. The ``read_csv()`` is by far the most commonly used since it can read any text file with a specified delimiter (comma, tab, or otherwise). 

======  ============================================  ===================================================  =================================
Typ1e    Data Description                              Reader                                               Writer
======  ============================================  ===================================================  =================================
text    **CSV / ASCII text with standard delimiter**  ``read_csv(path_or_url, sep=',', **kwargs)``         ``to_csv(path, **kwargs)``
text    Fixed-Width Text File                         ``read_fwf()``                                       N/A
text    JSON                                          ``read_json()``                                      ``to_json(path, **kwargs)``
SQL     SQLite table or query                         ``read_sql()``                                       ``to_sql(path, **kwargs)``
binary  **MS Excel**/**OpenDocument**                 ``read_excel(path_or_url, sheet_name=0, **kwargs)``  ``to_excel(path, **kwargs)``
binary  HDF5 Format                                   ``read_hdf()``                                       ``to_hdf(path, **kwargs)``
======  ============================================  ===================================================  =================================

This is far from a complete list, and most of these functions have several dozen possible kwargs. It is left to the reader to determine what kwargs are needed. As with NumPy's ``genfromtxt()`` function, most of the *text* readers above, and the excel reader, have kwargs that let you choose to load only some of the data.

In the example below, a CSV file called "exoplanets_5250_EarthUnits.csv" in the current working directory is read into the DataFrame ``df`` and then written out to a plain text file where decimals are rendered with commas, the delimiter is the pipe character, and the indexes are preserved as the first column.


.. hint:: 

   Try it yourself!
   
   .. code-block:: python
   
      import pandas as pd
      df = pd.read_csv('exoplanets_5250_EarthUnits.csv',index_col=0)
      df.to_csv('./docs/day3/exoplanets_5250_EarthUnits.txt', sep='|',decimal=',', index=True)

In most reader functions, including ``index_col=0`` sets the first column as the row labels, and the first row is assumed to contain the list of column names by default. If you forget to set one of the columns as the list of row indexes during import, you can do it later with ``df.set_index('column_name')``.

Building a DataFrame or Series from scratch is also easy. Lists and arrays can be converted directly to Series and DataFrames, respectively.

* Both ``pd.Series()`` and ``pd.DataFrame()`` have an ``index`` kwarg to assign a list of numbers, names, times, or other hashable keys to each row. 
* You can use the ``columns`` kwarg in ``pd.DataFrame()`` to assign a list of names to the columns of the table. The equivalent for ``pd.Series()`` is just ``name``, which only takes a single value and doesn't do anything unless you plan to join that Series to a larger DataFrame.
* Dictionaries and record arrays can be converted to DataFrames with ``pd.DataFrame.from_dict(myDict)`` and ``pd.DataFrame.from_records(myRecArray)``, respectively, and the keys will automatically be converted to column labels.

**Example**

.. hint:: 

   Try it yourself!

   .. jupyter-execute::
   
       import numpy as np
       import pandas as pd
       df = pd.DataFrame( np.random.randint(0,100, size=(4,4)), columns=['a','b','c','d'], index=['w','x','y','z'] )
       print(df)

It is also possible to convert DataFrames and Series to NumPy arrays (with or without the indexes), dictionaries, record arrays, or strings with the methods ``.to_numpy()``, ``.to_dict()``, ``to_records()``, and ``to_string()``, respectively.



