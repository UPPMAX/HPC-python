Big data with Python
====================

.. admonition:: "Learning outcomes"

   Learners 

   - can decide on useful file formats
   - can allocate resources sufficient to data size
   - can use data-chunking as technique
   - have grasped the surface of more efficient packages
    

High-Performance Data Analytics (HPDA)
--------------------------------------

.. admonition:: What is it?
   :class: dropdown

   - **High-performace data analytics (HPDA)**, a subset of high-performance computing which focuses on working with large data.

         - The data can come from either computer models and simulations or from experiments and observations, and the goal is to preprocess, analyse and visualise it to generate scientific results.

   - “**Big data** refers to data sets that are too large or complex to be dealt with by traditional data-processing application software. […] 

         - Big data analysis challenges include capturing data, data storage, data analysis, search, sharing, transfer, visualization, querying, updating, information privacy, and data source.” (from Wikipedia)

.. admonition:: What do we need to cover??
   :class: dropdown

   - File formats
 
   - Methods

      - RAM allocation 
      - chunking 

Types of scientific data
------------------------

.. admonition:: What do we need?
   :class: dropdown

   - Human readable?
   - Space efficiency?
   - Tidy data?
   - Arbitrary data?
   - Array data?
   - Long term storage/sharing?

Bit and Byte
............

- The smallest building block of storage in the computer is a bit, which stores either a 0 or 1. 
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
     - ❌
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

Allocating RAM
--------------

- Storing the data in an efficient way is one thing!

- Using the data in a program is another. 
- How much is actually loaded into the working memory (RAM)
- Is more data in variables created during the run or work?

.. important::

   - Allocate many cores or a full node!
   - You do not have to explicitely run threads or other parallelism.

- Note that shared memory among the cores works within node only.


.. discussion::

   - Take some time to find out the answers on the questions below, using the table of hardware
   - I'll ask around in a few minutes

.. admonition:: Table of hardware
   :class: dropdown

   .. list-table:: Hardware
      :widths: 25 25 25 25 25 25 25 25
      :header-rows: 1

      * - Technology
        - Kebnekaise
        - Rackham
        - Snowy
        - Bianca
        - Cosmos  
        - Tetralith   
        - Dardel
      * - Cores/compute node
        - 28 (72 for largemem, 128/256 for AMD Zen3/Zen4)
        - 20
        - 16
        - 16
        - 48  
        - 32  
        - 128
      * - Memory/compute node
        - 128-3072 GB 
        - 128-1024 GB
        - 128-4096 GB
        - 128-512 GB
        - 256-512 GB  
        - 96-384 GB   
        - 256-2048 GB
      * - GPU
        - NVidia V100, A100, A6000, L40s, H100, A40, AMD MI100 
        - None
        - NVidia T4 
        - NVidia A100
        - NVidia A100 
        - NVidia T4   
        - 4 AMD Instinct™ MI250X á 2 GCDs

.. admonition:: How much memory do I get per core?
   :class: dropdown

   - Divide GB RAM of the booked node with number of cores.

   - Example: 128 GB node with 20 cores
       - ~6.4 GB per core

.. admonition:: How much memory do I get with 5 cores?
   :class: dropdown

   - Multiply the RAM per core with number of allocated cores..

   - Example: 6.4 GB per core 
       - ~32 GB 

.. admonition:: Do you remember how to allocate several cores?
   :class: dropdown

   - Slurm flag ``-n <number of cores>``

- Choose, if necessary a node with more RAM
   - See local HPC center documentation in how to do so!

Dask
----

How to use more resources than available?

.. image:: ../img/when-to-use-pandas.png
   :width: 600 px

Dask is very popular for data analysis and is used by a number of high-level
Python libraries:

- Dask is composed of two parts:

    - **Dask Clusters**: Dynamic task scheduling optimized for computation. Similar to other workflow management systems, but optimized for interactive computational workloads.
        - `ENCCS course <https://enccs.github.io/hpda-python/dask/#dask-clusters>`_
    - **“Big Data” Collections** like parallel arrays, dataframes, and lists that extend common interfaces like NumPy, Pandas, or Python iterators to **larger-than-memory** or distributed environments. These parallel collections run on top of dynamic task schedulers.

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

Exercises
---------

.. note::
   
   - You can do the python exercises in the Python **command line** or Spyder, but better in Jupyter.
   - You may want to use an `interactive session <https://uppmax.github.io/HPC-python/common/interactive_ondemand.html>`_ (on-demand or interactive/salloc)

.. admonition:: Compute allocations in this workshop 

   - Rackham: ``uppmax2025-2-296``
   - Kebnekaise: ``hpc2n2025-076``
   - Cosmos: ``lu2025-7-34``
   - Tetralith: ``naiss2025-22-403``  
   - Dardel: ``naiss2025-22-403``

.. admonition:: Storage space for this workshop 

   - Rackham: ``/proj/hpc-python-uppmax``
   - Kebnekaise: ``/proj/nobackup/hpc-python-spring``
   - Cosmos: ``/lunarc/nobackup/projects/lu2024-17-44``
   - Tetralith: ``/proj/hpc-python-spring-naiss``
   - Dardel: ``/cfs/klemming/projects/snic/hpc-python-spring-naiss``

**Load and run**

.. tabs::

   .. tab:: HPC2N

      .. important::

         You should for this session load

         .. code-block:: console

            ml GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 Tkinter/3.11.3

         - And install ``dask`` & ``xarray`` to ``~/.local/`` if you don't already have it

         .. code-block:: console

            pip install xarray dask

   .. tab:: LUNARC

      .. important::

         You should for this session load

         .. code-block:: console

            ml GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 matplotlib/3.8.2

         - And install ``dask`` & ``xarray`` to ``~/.local/`` if you don't already have it

         .. code-block:: console

            pip install xarray dask

   .. tab:: UPPMAX

      .. important::

         You should for this session load

         .. code-block:: console

            module load python_ML_packages/3.11.8-cpu

         - And install ``xarray`` to ``~/.local/`` if you don't already have it

         .. code-block:: console

            pip install --user xarray 

   .. tab:: NSC

      .. important::

         You should for this session load

         .. code-block:: console

            module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 JupyterLab/4.2.0

         - And install ``dask`` & ``xarray`` to ``~/.local/`` if you don't already have it

         .. code-block:: console

            pip install xarray dask

   .. tab:: Dardel (PDC)

      - Jupyter Lab and Spyder are only available on Dardel via ThinLinc. 
      - As there are only 30 ThinLinc licenses available at this time, we recommend that you work on the exercises with a local installation on a personal computer. 
      - Do not trust that a ThinLinc session will be available or that On-Demand applications run therein will start in time for you to keep up (it is not unusual for wait times to be longer than the requested walltime). 
      - The exercises were written to work on a regular laptop. If you must work on Dardel, follow the steps below.

      .. important::

         For this session, you could load

         .. code-block:: console
        
            ml cray-python/3.11.5
         
         - And install ``dask`` & ``xarray`` to ``~/.local/`` if you don't already have it

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
             conda install xarray dask
             spyder &


.. exercise:: While installing or asking for compute nodes, read a bit more about the different data storage formats

   - Give it 5 minutes
   - Discuss a bit later in the group the different formats
   - Do you have a new favorite?


.. challenge:: Chunk sizes in Dask

   - The following example calculate the mean value of a random generated array. 
   - Run the 2 examples and see the performance improvement by using dask.

   .. tabs::

      .. tab:: NumPy

         .. code-block:: python
           
            import numpy as np

         .. code-block:: python
           
            %%time
            x = np.random.random((20000, 20000))
            y = x.mean(axis=0)

      .. tab:: Dask

         .. code-block:: python
           
            import dask
            import dask.array as da

         .. code-block:: python
           
            %%time
            x = da.random.random((20000, 20000), chunks=(1000, 1000))
            y = x.mean(axis=0)
            y.compute() 

   But what happens if we use different chunk sizes?
   Try out with different chunk sizes:
   
   - What happens if the dask chunks=(20000,20000)
   
   - What happens if the dask chunks=(250,250)


   .. solution:: Choice of chunk size

      The choice is problem dependent, but here are a few things to consider:

      Each chunk of data should be small enough so that it fits comforably in each worker's available memory. 
      Chunk sizes between 10MB-1GB are common, depending on the availability of RAM. Dask will likely 
      manipulate as many chunks in parallel on one machine as you have cores on that machine. 
      So if you have a machine with 10 cores and you choose chunks in the 1GB range, Dask is likely to use at least 
      10 GB of memory. Additionally, there should be enough chunks available so that each worker always has something to work on.

      On the otherhand, you also want to avoid chunk sizes that are too small as we see in the exercise.
      Every task comes with some overhead which is somewhere between 200us and 1ms. Very large graphs 
      with millions of tasks will lead to overhead being in the range from minutes to hours which is not recommended.

.. exercise:: Use Xarray to work with NetCDF files

   This exercise is derived from `Xarray Tutorials <https://tutorial.xarray.dev/intro.html>`__,
   which is distributed under an Apache-2.0 License.

   First create an Xarray dataset: 

   .. code-block:: python

      import numpy as np
      import xarray as xr

      ds1 = xr.Dataset(
          data_vars={
              "a": (("x", "y"), np.random.randn(4, 2)),
              "b": (("z", "x"), np.random.randn(6, 4)),
          },
          coords={
              "x": np.arange(4),
              "y": np.arange(-2, 0),
              "z": np.arange(-3, 3),
          },
      )
      ds2 = xr.Dataset(
          data_vars={
              "a": (("x", "y"), np.random.randn(7, 3)),
              "b": (("z", "x"), np.random.randn(2, 7)),
          },
          coords={
              "x": np.arange(6, 13),
              "y": np.arange(3),
              "z": np.arange(3, 5),
          },
      )

   Then write the datasets to disk using :meth:`to_netcdf` method:

   .. code-block:: python

      ds1.to_netcdf("ds1.nc")
      ds2.to_netcdf("ds2.nc")

   You can read an individual file from disk by using :meth:`open_dataset` method:

   .. code-block:: python

      ds3 = xr.open_dataset("ds1.nc")

   or using the :meth:`load_dataset` method:

   .. code-block:: python

      ds4 = xr.load_dataset('ds1.nc')

   Tasks:

   - Explore the hierarchical structure of the ``ds1`` and ``ds2`` datasets in a Jupyter notebook by typing the 
     variable names in a code cell and execute. Click the disk-looking objects on the right to expand the fields.
   - Explore ``ds3`` and ``ds4`` datasets, and compare them with ``ds1``. What are the differences?

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
