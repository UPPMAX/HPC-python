.. _big-data:

Big data with Python
====================

.. admonition:: "Learning outcomes"

   Learners 

   - can decide on useful file formats
   - can allocate resources sufficient to data size
   - can use data-chunking as technique
   - know where to learn more
    
.. admonition:: "For teacher"

   Preliminary timings

   - Intro 10 min
   - Files 5
   - Exercise files 10
   - Memory 5
   - Exercise Allocation 10
   - Dask 10
   - Exercise Dask 30

High-Performance Data Analytics (HPDA)
--------------------------------------

.. admonition:: What is it?
   :class: dropdown

   - **High-performace data analytics (HPDA)**, a subset of high-performance computing which focuses on working with **large data**.

         - The data can come from either computer models and simulations or from experiments and observations, and the goal is to preprocess, analyse and visualise it to generate scientific results.

   - “**Big data** refers to data sets that are too large or complex to be dealt with by traditional data-processing application software. […] 

         - Big data analysis challenges include capturing data, data storage, data analysis, search, sharing, transfer, visualization, querying, updating, information privacy, and data source.” (from Wikipedia)

Why we need to take special actions
-----------------------------------

.. discussion:: 

   - What can limit us?

.. admonition:: What do we need to cover??
   :class: dropdown

   - File formats
   - Methods
   - RAM allocation 
      - chunking 

scenario
::::::::

- use dataset (10 GB)
- fails in pandas or is slow
- Load with dask + xarray

What the constraints are
------------------------

- storage
- reading into memory


Memory, nodes

Solutions and tools
-------------------

- Choose file format for reading and writing
- Allocate enough RAM
- Choose the Python package
- Is chunking suitable?

File formats
------------

Bit and Byte
............

- The smallest building block of storage and memory (RAM) in the computer is a bit, which stores either a 0 or 1. 
- Normally a number of 8 bits are combined in a group to make a byte. 
- One byte (8 bits) can represent/hold at most 2^8 distinct values. Organising bytes in different ways can represent different types of information, i.e. data.

.. admonition:: Numerical data
   :class: dropdown

   Different numerical data types (e.g. integer and floating-point numbers) can be represented by bytes. The more bytes we use for each value, the larger is the range or precision we get, but more bytes require more memory.

   - DataTypes
       - 16-bit: short (integer)
       - 32-bit: Int (integer)
       - 32-bit: Single (floating point)
       - 64-bit: Double (floating point)

   - For some use cases, the precision is not that important, 1% error, or so, is not that crucial. Faster and less data storage!

.. admonition:: Text data
   :class: dropdown

   - DataTypes
        - 8-bit: char

   .. admonition:: more
      :class: dropdown

      - When it comes to text data, the simplest character encoding is ASCII (American Standard Code for Information Interchange) and was the most common character encodings until 2008 when UTF-8 took over.
      - The original ASCII uses only 7 bits for representing each character and therefore encodes only 128 specified characters. Later it became common to use an 8-bit byte to store each character in memory, providing an extended ASCII.
      - As computers became more powerful and the need for including more characters from other languages like Chinese, Greek and Arabic became more pressing, UTF-8 became the most common encoding. UTF-8 uses a minimum of one byte and up to four bytes per character.

Data and storage format
.......................

In real scientific applications, data is complex and structured and usually contains both numerical and text data. Here we list a few of the data and file storage formats commonly used.

.. admonition:: Tabular data
   :class: dropdown

   - A very common type of data is “tabular data”.
   - Tabular data is structured into **rows and columns**.
   - Each column usually has a name and a specific data type while each row is a distinct sample which provides data according to each column (including missing values).
   - The simplest and most common way to save tabular data is via the so-called CSV (comma-separated values) file.

.. admonition:: Gridded data
   :class: dropdown

   - Gridded data is another very common data type in which numerical data is normally saved in a multi-dimensional rectangular grid. Most probably it is saved in one of the following formats:

   .. admonition:: more
      :class: dropdown

      - Hierarchical Data Format (HDF5) - Container for many arrays

      - Network Common Data Form (NetCDF) - Container for many arrays which conform to the NetCDF data model

      - Zarr - New cloud-optimized format for array storage


.. admonition:: Meta data
   :class: dropdown

   Metadata consists of various information about the data. Different types of data may have different metadata conventions.

   .. admonition:: more
      :class: dropdown

      - In Earth and Environmental science, there are widespread robust practices around metadata. For NetCDF files, metadata can be embedded directly into the data files. The most common metadata convention is Climate and Forecast (CF) Conventions, commonly used with NetCDF data.

      - When it comes to data storage, there are many types of storage formats used in scientific computing and data analysis. There isn’t one data storage format that works in all cases, so choose a file format that best suits your data.

.. admonition:: CSV (comma-separated values)
   :class: dropdown

   **Best use cases**: Sharing data. Small data. Data that needs to be human-readable.

   - Key features

       Type: Text format
       Packages needed: NumPy, Pandas
       Space efficiency: Bad
       Good for sharing/archival: Yes

   .. admonition:: more
      :class: dropdown

       Tidy data:
               Speed: Bad
               Ease of use: Great

       Array data:
               Speed: Bad
               Ease of use: OK for one or two dimensional data. Bad for anything higher.

.. admonition:: HDF5 (Hierarchical Data Format version 5)
   :class: dropdown

   - HDF5 is a high performance storage format for storing large amounts of data in multiple datasets in a single file. 
   - It is especially popular in fields where you need to store big multidimensional arrays such as physical sciences.
   - **Best use cases**: Working with big datasets in array data format.

   - Key features

       - Type: Binary format
       - Packages needed: Pandas, PyTables, h5py
       - Space efficiency: Good for numeric data.
       - Good for sharing/archival: Yes, if datasets are named well.

   .. admonition:: more
      :class: dropdown

      - Tidy data:
          - Speed: Ok
          - Ease of use: Good

      - Array data:
          - Speed: Great
          - Ease of use: Good

.. admonition:: NETCDF4 (Network Common Data Form version 4)
   :class: dropdown

   - NetCDF4 is a data format that uses HDF5 as its file format, but it has standardized structure of datasets and metadata related to these datasets. 
   - This makes it possible to be read from various different programs.

     **Best use cases**: Working with big datasets in array data format. Especially useful if the dataset contains spatial or temporal dimensions. Archiving or sharing those datasets.
   
   - Key features

       - Type: Binary format
       - Packages needed: Pandas, netCDF4/h5netcdf, xarray
       - Space efficiency: Good for numeric data.
       - Good for sharing/archival: Yes.

   .. admonition:: more
      :class: dropdown

      - Tidy data:
          - Speed: Ok
          - Ease of use: Good

      - Array data:
          - Speed: Good
          - Ease of use: Great

      - NetCDF4 is by far the most common format for storing large data from big simulations in physical sciences.
      - The advantage of NetCDF4 compared to HDF5 is that one can easily add additional metadata, e.g. spatial dimensions (x, y, z) or timestamps (t) that tell where the grid-points are situated. As the format is standardized, many programs can use this metadata for visualization and further analysis.

An overview of common data formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - | Name:
     - | Human
       | readable:
     - | Space
       | efficiency:
     - | Arbitrary
       | data:
     - | Tidy
       | data:
     - | Array
       | data:
     - | Long term
       | storage/sharing:

   * - :ref:`Pickle <pickle>`
     - ❌
     - 🟨
     - ✅
     - 🟨
     - 🟨
     - ❌

   * - :ref:`CSV <csv>`
     - ✅
     - ❌
     - ❌
     - ✅
     - 🟨
     - ✅

   * - :ref:`Feather <feather>`
     - ❌
     - ✅
     - ❌
     - ✅
     - ❌
     - ❌

   * - :ref:`Parquet <parquet>`
     - ❌
     - ✅
     - 🟨
     - ✅
     - 🟨
     - ✅

   * - :ref:`npy <npy>`
     - ❌
     - 🟨
     - ❌
     - ❌
     - ✅
     - ❌

   * - :ref:`HDF5 <hdf5>`
     - ❌
     - ✅
     - ❌
     - ❌
     - ✅
     - ✅

   * - :ref:`NetCDF4 <netcdf4>`
     - ❌
     - ✅
     - ❌
     - ❌
     - ✅
     - ✅

   * - :ref:`JSON <json>`
     - ✅
     - ❌
     - 🟨
     - ❌
     - ❌
     - ✅

   * - :ref:`Excel <excel>`
     - ✅
     - ❌
     - ❌
     - 🟨
     - ❌
     - 🟨

   * - :ref:`Graph formats <https://gephi.org/users/supported-graph-formats/>`
     - 🟨
     - 🟨
     - ❌
     - ❌
     - ❌
     - ✅

.. important:: Legend

    - ✅ : Good
    - 🟨 : Ok / depends on a case
    - ❌ : Bad

    Adapted from Aalto university's `Python for scientific computing <https://aaltoscicomp.github.io/python-for-scicomp/work-with-data/#what-is-a-data-format>`__... seealso::

   - ENCCS course "HPDA-Python": `Scientific data <https://enccs.github.io/hpda-python/scientific-data/>`_
   - Aalto Scientific Computing course "Python for Scientific Computing": `Xarray <https://aaltoscicomp.github.io/python-for-scicomp/xarray/>`_



Computing efficiency with Python
--------------------------------

Python is an interpreted language, and many features that make development rapid with Python are a result of that, with the price of reduced performance in many cases.

- Dynamic typing
- Flexible data structures

- There are some packages that are more efficient than Numpy and Pandas.

    - `SciPy <https://docs.scipy.org/doc/scipy/reference/>`_ is a library that builds on top of NumPy. 
   
        - It contains a lot of interfaces to battle-tested numerical routines written in Fortran or C, as well as Python implementations of many common algorithms.
   
    - `ENCCS course material <https://enccs.github.io/hpda-python/stack/#scipy>`_

XARRAY Package
..............

- ``xarray`` is a Python package that builds on NumPy but adds labels to **multi-dimensional arrays**. 
- It also borrows heavily from the Pandas package for labelled tabular data and integrates tightly with dask for parallel computing. 
- Xarray is particularly tailored to working with NetCDF files. 
- It reads and writes to NetCDF file using

    - ``open_dataset()`` function
    - ``open_dataarray()`` function
    - ``to_netcdf()`` method. 

- Explore these in the exercise below!

Exercise file formats
---------------------

Go over file formats and see if some are more relevant for your work.

.. discussion::

   - Would you look at other file formats and why?

Dask
----

How to use more resources than available?

.. image:: ../img/when-to-use-pandas.png
   :width: 600 px

Dask is very popular for data analysis and is used by a number of high-level
Python libraries:

- Dask is composed of two parts:

    - **Dask Clusters**
        - Dynamic task scheduling optimized for computation. Similar to other workflow management systems, but optimized for interactive computational workloads.
        - `ENCCS course <https://enccs.github.io/hpda-python/dask/#dask-clusters>`_
    - **“Big Data” Collections**
        - Like parallel arrays, dataframes, and lists that extend common interfaces like NumPy, Pandas, or Python iterators to **larger-than-memory** or distributed environments. These parallel collections run on top of dynamic task schedulers.
        -`ENCCS course <https://enccs.github.io/hpda-python/dask/#dask-collections>`_

Dask Collections
................

- Dask provides dynamic parallel task scheduling and three main high-level collections:
  
    - ``dask.array``: Parallel NumPy arrays
        - scales NumPy (see also xarray)
    - ``dask.dataframe``: Parallel Pandas DataFrames
        - scales Pandas workflows
    - ``dask.bag``: Parallel Python Lists 
        - https://enccs.github.io/hpda-python/dask/#dask-bag

Dask Arrays
^^^^^^^^^^^

- A Dask array looks and feels a lot like a NumPy array. 
- However, a Dask array uses the so-called "lazy" execution mode, which allows one to 
    - build up complex, large calculations symbolically 
    - before turning them over the scheduler for execution. 

- Dask divides arrays into many small pieces (chunks), as small as necessary to 
  fit it into memory. 
- Operations are delayed (lazy computing) e.g. tasks are queue and no computation 
  is performed until you actually ask values to be computed (for instance print mean values). 
- Then data is loaded into memory and computation proceeds in a streaming fashion, block-by-block.

.. discussion:: Example from dask.org

   .. code-block::

      # Arrays implement the Numpy API
      import dask.array as da
      x = da.random.random(size=(10000, 10000),
                           chunks=(1000, 1000))
      x + x.T - x.mean(axis=0)
      # It runs using multiple threads on your machine.
      # It could also be distributed to multiple machines

.. seealso::

   - `dask_ml package <https://ml.dask.org/>`_: Dask-ML provides scalable machine learning in Python using Dask alongside popular machine learning libraries like Scikit-Learn, XGBoost, and others.
   - `Dask.distributed <https://distributed.dask.org/en/stable/>`_: Dask.distributed is a lightweight library for distributed computing in Python. It extends both the concurrent.futures and dask APIs to moderate sized clusters.

Chunking
::::::::

Tools like Dask and xarray handle chunking 
automatically. Add one short diagram showing:
Big file → split into chunks → parallel workers → results combined.

.. admonition:: Keywords
 
   - chunk size
   - lazy execution
   - meta-data-rich arrays

.. admonition:: Sum up

   - Load Python modules and activate virtual environments.
   - Request appropriate memory and runtime in SLURM.
   - Store temporary data in local scratch ($SNIC_TMP).
   - Check job memory usage with sacct or sstat.

Exercise DASK
-------------



Allocating RAM
--------------

- Mention memory per core considerations.
- Show SLURM options for memory and time.
- Briefly explain what happens when a Dask job runs on multiple cores.



.. admonition:: Keywords

   OOM



Workflow
--------

Data source → Format choice → Load/Chunk → Process → Write

Exercises
---------

- Pandas 
- xarray
- dask

Summary

.. discussion:: Follow-up discussion

   - New learnings?
   - Useful file formats
   - Resources sufficient to data size
   - Data-chunking as technique if not enough RAM
   - Is xarray useful for you?

.. keypoints::

   - File formats
       - No format fits all requirements
       - HDF5 and NetCDF good for Big data
   - Packages
       - xarray
          - can deal with 3D-data and higher dimensions
       - Dask 
           - uses lazy execution
           - Only use for processing very large amount of data
   - Allocate more RAM by asking for
       - Several cores
       - Nodes will more RAM
   

.. seealso::

   - `Dask documentation <https://docs.dask.org/en/stable/>`_

   Working with data

   - https://aaltoscicomp.github.io/python-for-scicomp/work-with-data/
   
   Tidy data

   - https://coderefinery.github.io/data-visualization-python/tidy-data/
   
   ENCCS
 
   - Dask for scalable analysis
   - https://enccs.github.io/hpda-python/stack/
   - https://enccs.github.io/hpda-python/dask/ 

   - Too be included in the future?

      - `Dask jobqueue <https://jobqueue.dask.org/en/latest/>`_
      - `Dask-MPI <http://mpi.dask.org/en/latest/index.html>`_


