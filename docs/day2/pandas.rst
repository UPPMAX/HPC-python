===============
Intro to Pandas
===============

**Pandas**, short for PANel Data AnalysiS, is a Python data library for cleaning, organizing, and statistically analyzing moderately large ($lesssim3$ GiB) data sets. It was originally developed for analyzing and modelling financial records (panel data) over time, and has since expanded into a package rivaling SciPy in the number and complexity of available functions. Pandas offers:
- Explicit, automatic data alignment: all entries have corresponding row and column labels/indexes.
- Easy methods to add, remove, transform, compare, broadcast, and aggregate data within and across data structures.
- Data structures that support any mix of numerical, string, list, Boolean, and datetime datatypes.
- I/O interfaces that support a wide variety of text, binary, and database formats, including Excel, JSON, HDF5, NetCDF, and SQLite.
- Hundreds of built-in functions for cleaning, organizing, and statistical analysis, plus support for user-defined functions.
- A simple interface with the Seaborn plotting library, and increasingly also Matplotlib.
- Easy multi-threading with Numba.

**Limitations.** Pandas alone has somewhat limited support for parallelization, N-dimensional data structures, and datasets much larger than 3 GiB. Fortunately, there are packages like ``dask`` and ``polars`` that can help. In partcular, ``dask`` will be covered in a later lecture in this workshop. There is also the ``xarray`` package that provides many similar functions to Pandas for higher-dimensional data structures, but that is outside the scope of this workshop.

.. tabs:: How is Pandas Provided?

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


.. hint:: Flow-Chart: Should I use Pandas?

  .. image:: ../img/when-to-use-pandas.png
     :width: 600 px

.. objectives:: You will learn...

  - What are the basic object classes, data types, and their most important attributes and methods
  - How to input/output work
  - How to inspect, clean, and sort data for later operations
  - How to perform basic operations: statistics, binary operators, vectorized math and string methods
  - What are GroupBy objects and their uses
  - How to compare data, implement complex and/or user-defined functions, and perform windowed operations
  - How to use or create time series data (if time allows)
  - Advanced topics (if time allows): How to prep for ML/AI, what are memory-saving data types

  We will also have a short sesion after this on plotting with Seaborn, a package for easily making publication-ready statistical plots.


Basic Object Classes and Data Types
-----------------------------------

The main object classes of Pandas are ``Series and ``DataFrame``.
- ``pandas.Series(data, index=None, name=None, **kwargs)`` instantiates a 1D array with customizable indexes (labels) attached to every entry for easy access, and optionally a name for later addition to a DataFrame as a column
  - Indexes can be numbers (integer or float), strings, datetime objects, or even tuples; the default is 0-based integer indexing
- ``pandas.DataFrame(data, columns=None, index=None, **kwargs)`` instantiates a 2D array where every column is a Series: all entries are accessible by column and row labels
  - Any function that works with a DataFrame will work with a Series unless the function specifically requires column or index arguments
  - Column labels and row indexes/labels can be safely (re)assigned as needed

The API reference in the [official Pandas documentation](https://pandas.pydata.org/docs/user_guide/index.html) shows *hundreds* of methods and attributes for each.



