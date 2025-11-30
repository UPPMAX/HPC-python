.. _big-data:

Big data with Python
====================

.. admonition:: "Learning outcomes"

   Learners 

   - can allocate resources sufficient to data size
   - can decide on useful file formats
   - can use data-chunking as technique
   - know where to learn more
    
.. admonition:: "For teacher"

   Preliminary timings. Starting at 13.00

   - Intro 10 min
   - Memory 5
   - Exercise Allocation 10
   - Files 5
   - Exercise files 10
   - Packages 5
   - Dask 5
   - BREAK 15min 13.50-14.05
   - Exercise Dask 30

High-Performance Data Analytics (HPDA)
--------------------------------------

.. admonition:: What is it?
   :class: dropdown

   - **High-performace data analytics (HPDA)**, a subset of high-performance computing which focuses on working with **large data**.

         - The data can come from either computer models and simulations or from experiments and observations, and the goal is to preprocess, analyse and visualise it to generate scientific results.

   - “**Big data** refers to data sets that are too large or complex to be dealt with by traditional data-processing application software. […] 

         - Big data analysis challenges include capturing data, data storage, data analysis, search, sharing, transfer, visualization, querying, updating, information privacy, and data source.” (from Wikipedia)

.. discussion:: 

   Do you already work with large data sets?

Why we need to take special actions
-----------------------------------

Remember this one?

.. image:: ../img/when-to-use-pandas.png
      :width: 600 px

.. discussion:: 

   - What can limit us?

What the constraints are
------------------------

- storage
- memory

.. admonition:: What do we need to cover??
   :class: dropdown

   - storage --> make more effective files
   - reading into memory
       - --> read just parts of files into memory
       - --> chunking
   - allocate more memory

Solutions and tools
-------------------

- Allocate enough RAM
    - If you are running ready tools
    - or cannot update code or use other packages
- Choose file format for reading and writing
- Choose the right Python package
- Is chunking suitable?

Allocating RAM
--------------

- How much is actually loaded into the working memory (RAM)
- Is more data in variables created during the run or work?

.. discusioon::

   Have you seen the Out-of-memory (OOM) error? 

.. admonition:: What to do

   - By allocating **many cores** on a node will give you more available memory
   - If the order 128 GB is not enough there are so-called **fat nodes** with at least 512 GB and up to 3 TB.
   - On some clusters you do not have to request additional CPUs to get additional memory. Use the ``slurm`` options
       - ``--mem`` or
       - ``--mem-per-cpu``

.. important::

   - You do not have to explicitely run threads or other parallelism.
   - Allocating several nodes for one one big problem is not useful.
      - Note that shared memory among the cores works within node only.

Principles
..........

Use the Slurm options for either "BATCH", "INTERACTIVE" from command line or from OnDemand GUIs.

- Allocate RAM using the **full node RAM divided by number of course principle**
   - Ex: 128 GB with 20 cores --> 6.4 GB per core
   - Allocate number of cores to cover your needs.
   - ``-n <number>``
- Request the memory needed and choose number of cores 
   - ``--mem=<size>[K|M|G|T]``
   - Example: ``--mem=1T|``
- Request the memory-per-core needed and choose number of cores 
   - ``--mem-core=<size>[K|M|G]
   - Example: ``--mem-per-cpu=16G``
- Request a "FAT" node.
   - Typically you can only allocate a full node here, no core parts.
   - You ask here for a non-default partition.
   - How to do this, search your cluster documentation, see exercise below.

.. note::

   - "core-hours" drawn from your project may be set to the maximum of "number of cores" and "memory part of node" requested. 
   - So there is no win to ask for one core but much memory! 

Exercise: Memory allocation (10 min)
------------------------------------

.. info:: Break-out rooms per Cluster or Cluster Type (OnDemand vs terminal)

1. Log in to a Desktop (ThinLinc or OnDemand) (see :ref:`common-login`)

- Tetralith (ThinLinc client: ``tetralith.nsc.liu.se``)
- Dardel (ThinLinc client: ``dardel-vnc.pdc.kth.se``)
- Alvis (https://alvis.c3se.chalmers.se/)
- Bianca (https://bianca.uppmax.uu.se/)
- Pelle (https://pelle-gui.uppmax.uu.se/)
- Cosmos (ThinLinc client: ``cosmos-dt.lunarc.lu.se``)
- Kebnekaise(https://portal.hpc2n.umu.se/public/landing_page.html)

.. discussion::

   - Take some time to find out the answers for your specific cluster for the questions below, using the table of hardware below.

.. admonition:: Table of hardware
   :class: dropdown

   .. list-table:: Hardware
      :widths: 25 25 25 25 25 25 25 25
      :header-rows: 1

      * - Technology
        - Kebnekaise
        - Pelle
        - Bianca
        - Cosmos  
        - Tetralith   
        - Dardel
        - Alvis
      * - Cores/compute node
        - 28 (72 for largemem, 128/256 for AMD Zen3/Zen4)
        - 48 (96 with hyperthreading/SMT)
        - 16
        - 16
        - 48  
        - 32  
        - 128
        - many different (updated soon)
      * - Memory/compute node
        - 128-3072 GB 
        - 768-3072 GB
        - 128-512 GB
        - 256-512 GB  
        - 96-384 GB   
        - 256-1760 GB
        - many different
      * - GPU
        - NVidia V100, A100, A6000, L40s, H100, A40, AMD MI100 
        - NVidia L40s, H100, T4, A2)
        - NVidia A100
        - NVidia A100 
        - NVidia T4   
        - 4 AMD Instinct™ MI250X á 2 GCDs
        - many different

.. admonition:: How much memory do I get per core?
   :class: dropdown

   - Divide GB RAM of the booked node with number of cores.

   - Example: 128 GB node with 16 cores
       - ~8 GB per core
   - NOTE: You may get less due to background system threads running in the background.
       - Example: On Bianca you may get 7 GB instead of 8 GB.


.. admonition:: How much memory do I get with 5 cores?
   :class: dropdown

   - Multiply the RAM per core with number of allocated cores..

   - Example: 8 GB per core 
       - ~40 GB 

.. admonition:: Do you remember how to allocate several cores?
   :class: dropdown

   - Slurm flag ``-n <number of cores>``

.. admonition:: Actually start an interactive sesion with 4 cores for 3 hours. 

   - We will use it for the exercises later.
   - Since it may take some time to get the allocation we do it now already!
   - Follow the best procedure for your cluster, e.g. from **command-line** or **OnDemand**.

.. admonition:: How?
   :class: drop-down

   The following Slurm options needs to be set

   - ``-t 3:0:0``
   - ``-n 4``
   - ``-A <proj>``
   - ``-p <partition>`` may be needed in some clusters

.. admonition:: Compute allocations in this workshop 
   :class: dropdown   

   - Pelle: ``uppmax2025-2-393``
   - Kebnekaise: ``hpc2n2025-151``
   - Cosmos: ``lu2025-7-106``
   - Alvis: ``naiss2025-22-934``
   - Tetralith: ``naiss2025-22-934``  
   - Dardel: ``naiss2025-22-934``

.. admonition:: How to get a node with more RAM

   - See local HPC center documentation in how to do so!

.. solution::

   .. tabs::

      .. tab:: Tetralith

         Scroll down a bit at https://www.nsc.liu.se/systems/tetralith/

      .. tab:: Dardel

         https://support.pdc.kth.se/doc/run_jobs/job_scheduling/#dardel-compute-nodes

      .. tab:: Alvis

         https://www.c3se.chalmers.se/documentation/submitting_jobs/running_jobs/#memory-and-other-node-features

      .. tab:: Bianca

         - https://docs.uppmax.uu.se/cluster_guides/slurm/#need-more-resources-or-gpu

      .. tab:: Pelle

         - https://docs.uppmax.uu.se/cluster_guides/slurm_on_pelle/#the-fat-partition

      .. tab:: Cosmos

         - https://lunarc-documentation.readthedocs.io/en/latest/manual/submitting_jobs/manual_specifying_requirements/#specifying-a-project-allocation-and-partition
         - https://www.lunarc.lu.se/systems/cosmos 

      .. tab:: Kebnekaise

         - https://docs.hpc2n.umu.se/documentation/batchsystem/resources/
         - https://docs.hpc2n.umu.se/documentation/batchsystem/resources/#requesting__specific__features__ie__setting__contraints__on__the__job
         - https://docs.hpc2n.umu.se/documentation/batchsystem/resources/#for__selecting__large__memory__nodes

.. solution::

   .. tabs::

      .. tab:: Tetralith

         ``-C fat --exclusive`` (384 GiB)

      .. tab:: Dardel

         - ``-p memory --mem=440GB``
         - ``-p memory --mem=880GB``
         - ``-p memory --mem=1760GB``

      .. tab:: Alvis

         - ``-C MEM512``
         - ``-C MEM1536``

      .. tab:: Bianca

         - ``-C mem256GB``
         - ``-C mem512GB``

      .. tab:: Pelle

         - ``-p fat -C 2TB``
         - ``-p fat -C 3TB``

      .. tab:: Cosmos

         - Part of GPU partitions
         - INTEL CPUs+A100 GPUs (384 GB): ``-p gpua100``
         - AMD CPUs+A100 GPUs (512 GB): ``-p gpua100``

      .. tab:: Kebnekaise

         ``-C largemem``


.. admonition:: 

   - We recommend a desktop environment for speed of the graphics.
   - connecting from local terminal with "ssh -X" (X11 forwarding) can be be used but is slower.

File formats
------------

.. admonition:: Bits and Bytes

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

    Adapted from Aalto university's `Python for scientific computing <https://aaltoscicomp.github.io/python-for-scicomp/work-with-data/#what-is-a-data-format>`__

... seealso::

   - ENCCS course "HPDA-Python": `Scientific data <https://enccs.github.io/hpda-python/scientific-data/>`_
   - Aalto Scientific Computing course "Python for Scientific Computing": `Xarray <https://aaltoscicomp.github.io/python-for-scicomp/xarray/>`_

Exercise file formats (10 minutes)
---------------------------------

.. challenge:: Reading NetCDF files

   - Read: https://stackoverflow.com/questions/49854065/python-netcdf4-library-ram-usage
   - What about using NETCDF files and memory?

.. challenge::

   - Start Jupyter or just a Python shell and
   - Go though and test the lines at the page at https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.io.netcdf_file.html

.. challenge:: 

   - Go over file formats and see if some are more relevant for your work.
   - Would you look at other file formats and why?


Computing efficiency with Python
--------------------------------

Python is an interpreted language, and many features that make development rapid with Python are a result of that, with the price of reduced performance in many cases.

- Dynamic typing
- Flexible data structures

- There are some packages that are more efficient than Numpy and Pandas.

    - `SciPy <https://docs.scipy.org/doc/scipy/reference/>`_ is a library that builds on top of NumPy. 
   
        - It contains a lot of interfaces to battle-tested numerical routines written in Fortran or C, as well as Python implementations of many common algorithms.
        - Reads NETCDF!
   
    - `ENCCS course material <https://enccs.github.io/hpda-python/stack/#scipy>`_

Xarray package
..............

- ``xarray`` is a Python package that builds on NumPy but adds labels to **multi-dimensional arrays**. 
    -  introduces labels in the form of dimensions, coordinates and attributes on top of raw NumPy-like multidimensional arrays, which allows for a more intuitive, more concise, and less error-prone developer experience.

    - It also borrows heavily from the Pandas package for labelled tabular data and integrates tightly with dask for parallel computing. 

- Xarray is particularly tailored to working with NetCDF files. 
- It reads and writes to NetCDF file using
 
- Explore it a bit in the (optional) exercise below!

Polars package
..............

**Blazingly Fast DataFrame Library**

.. admonition:: Goals 

   The goal of Polars is to provide a lightning fast DataFrame library that:

   - Utilizes all available cores on your machine.
   - Optimizes queries to reduce unneeded work/memory allocations.
   - Handles datasets much larger than your available RAM.
   - A consistent and predictable API.
   - Adheres to a strict schema (data-types should be known before running the query).

.. admonition:: Key features
   :class: drop-down

   - Fast: Written from scratch in Rust
   - I/O: First class support for all common data storage layers: 
   - Intuitive API: Write your queries the way they were intended. Internally, there is a query optimizer.
   - Out of Core: streaming without requiring all your data to be in memory at the same time.
   - Parallel: dividing the workload among the available CPU cores without any additional configuration.
   - GPU Support: Optionally run queries on NVIDIA GPUs
   - Apache Arrow support

   https://pola.rs/

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

   - Briefly explain what happens when a Dask job runs on multiple cores.



Exercise DASK
-------------







Workflow
--------

Data source → Format choice → Load/Chunk → Process → Write

Exercises
---------


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

.. challenge:: (Optional) Xarray

   - https://stackoverflow.com/questions/72155514/when-to-use-xarray-over-numpy-for-medium-rank-multidimensional-data

   - Browse: https://docs.xarray.dev/en/v2024.11.0/getting-started-guide/why-xarray.html# or change to more applicabe version in drop-down menu to lower right.
       - find something interesting for you! Test some lines if you want to!
       - tips: 
           - Pandas: https://docs.xarray.dev/en/v2024.11.0/getting-started-guide/faq.html#why-is-pandas-not-enough
           - gallery: https://docs.xarray.dev/en/v2024.11.0/gallery.html
           - ecosystems: https://docs.xarray.dev/en/v2024.11.0/ecosystem.html
           - Quick overview: https://docs.xarray.dev/en/v2024.11.0/getting-started-guide/quick-overview.html



.. challenge:: (Optional) Polars

   - Browse: https://docs.pola.rs/.
       - find something interesting for you! Test some lines if you want to!
       - tips: 

   - Check if your cluster has Polars!

   .. solution::

      - Check with ``ml spider polars``
      - If it is installed it will show up as 

      .. code-block:: console

         --------------------------------------------------------------------
           polars:
         --------------------------------------------------------------------
             Description:
               Polars is a blazingly fast DataFrame library for manipulating
               structured data. The core is written in Rust and this module
               provides its interface for Python.
         
              Versions:
                 polars/1.28.1-gfbf-2024a
                 polars/1.29.0-gfbf-2024a
         
         --------------------------------------------------------------------
           For detailed information about a specific "polars" package (including
         how to load the modules) use the module's full name.
           Note that names that have a trailing (E) are extensions provided by ot
         her modules.
           For example:
         
              $ module spider polars/1.29.0-gfbf-2024a
         --------------------------------------------------------------------

      - Load the module or install it in your present ``conda`` or ``venv`` environment

      - Try the most interesting examples: https://docs.pola.rs/user-guide/getting-started/#reading-writing


Summary
-------

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


