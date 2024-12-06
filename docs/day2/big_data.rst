Big data with Python
====================

.. admonition:: "Learning outcomes"

   Learners can

   - allocate resources sufficient to data size
   - decide on useful file formats
   - use data-chunking as technique

To cover
--------

- File format
 
- Methods
   - RAM allocation 
   - chunking 


Files formats
-------------

Types of scientific data
........................

Bit and Byte
^^^^^^^^^^^^


The smallest building block of storage in the computer is a bit, which stores either a 0 or 1. Normally a number of 8 bits are combined in a group to make a byte. One byte (8 bits) can represent/hold at most 2^8 distinct values. Organising bytes in different ways can represent different types of information, i.e. data.

.. admonition:: Numerical data
   :class: dropdown

   Different numerical data types (e.g. integer and floating-point numbers) can be represented by bytes. The more bytes we use for each value, the larger is the range or precision we get, but more bytes require more memory.

   - DataTypes
       - 16-bit: short (integer
       - 32-bit: Int (integer)
       - 32-bit: Single (floating point
       - 64-bit: Double (floating point

.. admonition:: Text data
   :class: dropdown

   - DataTypes
       - 8-bit: char

   When it comes to text data, the simplest character encoding is ASCII (American Standard Code for Information Interchange) and was the most common character encodings until 2008 when UTF-8 took over. The original ASCII uses only 7 bits for representing each character and therefore encodes only 128 specified characters. Later it became common to use an 8-bit byte to store each character in memory, providing an extended ASCII.

   As computers became more powerful and the need for including more characters from other languages like Chinese, Greek and Arabic became more pressing, UTF-8 became the most common encoding. UTF-8 uses a minimum of one byte and up to four bytes per character.

Data and storage format
.......................

In real scientific applications, data is complex and structured and usually contains both numerical and text data. Here we list a few of the data and file storage formats commonly used.

.. admonition:: Tabular data
   :class: dropdown

   A very common type of data is “tabular data”. Tabular data is structured into rows and columns. Each column usually has a name and a specific data type while each row is a distinct sample which provides data according to each column (including missing values). The simplest and most common way to save tabular data is via the so-called CSV (comma-separated values) file.

.. admonition:: Gridded data
   :class: dropdown

   Gridded data is another very common data type in which numerical data is normally saved in a multi-dimensional rectangular grid. Most probably it is saved in one of the following formats:

   - Hierarchical Data Format (HDF5) - Container for many arrays

   - Network Common Data Form (NetCDF) - Container for many arrays which conform to the NetCDF data model

   - Zarr - New cloud-optimized format for array storage


.. admonition:: Meta data
   :class: dropdown

   Metadata consists of various information about the data. Different types of data may have different metadata conventions.

   - In Earth and Environmental science, there are widespread robust practices around metadata. For NetCDF files, metadata can be embedded directly into the data files. The most common metadata convention is Climate and Forecast (CF) Conventions, commonly used with NetCDF data.

   - When it comes to data storage, there are many types of storage formats used in scientific computing and data analysis. There isn’t one data storage format that works in all cases, so choose a file format that best suits your data.


.. admonition:: CSV (comma-separated values)

   
   Key features

       Type: Text format
       Packages needed: NumPy, Pandas
       Space efficiency: Bad
       Good for sharing/archival: Yes

       Tidy data:
               Speed: Bad
               Ease of use: Great

       Array data:
               Speed: Bad
               Ease of use: Ok for one or two dimensional data. Bad for anything higher.

       Best use cases: Sharing data. Small data. Data that needs to be human-readable.


.. admonition:: HDF5 (Hierarchical Data Format version 5)
   :class: dropdown

   Key features

    Type: Binary format
    Packages needed: Pandas, PyTables, h5py
    Space efficiency: Good for numeric data.
    Good for sharing/archival: Yes, if datasets are named well.

    Tidy data:
            Speed: Ok
            Ease of use: Good

    Array data:
            Speed: Great
            Ease of use: Good

    Best use cases: Working with big datasets in array data format.

HDF5 is a high performance storage format for storing large amounts of data in multiple datasets in a single file. It is especially popular in fields where you need to store big multidimensional arrays such as physical sciences.

.. admonition:: (Network Common Data Form version 4)
   :class: dropdown

   

Key features

    Type: Binary format
    Packages needed: Pandas, netCDF4/h5netcdf, xarray
    Space efficiency: Good for numeric data.
    Good for sharing/archival: Yes.

    Tidy data:
            Speed: Ok
            Ease of use: Good

    Array data:
            Speed: Good
            Ease of use: Great

    Best use cases: Working with big datasets in array data format. Especially useful if the dataset contains spatial or temporal dimensions. Archiving or sharing those datasets.

NetCDF4 is a data format that uses HDF5 as its file format, but it has standardized structure of datasets and metadata related to these datasets. This makes it possible to be read from various different programs.

NetCDF4 is by far the most common format for storing large data from big simulations in physical sciences.

The advantage of NetCDF4 compared to HDF5 is that one can easily add additional metadata, e.g. spatial dimensions (x, y, z) or timestamps (t) that tell where the grid-points are situated. As the format is standardized, many programs can use this metadata for visualization and further analysis.

XARRAY
......
Xarray is a Python package that builds on NumPy but adds labels to multi-dimensional arrays. It also borrows heavily from the Pandas package for labelled tabular data and integrates tightly with dask for parallel computing. NumPy, Pandas and Dask will be covered in later episodes.
Xarray is particularly tailored to working with NetCDF files. It reads and writes to NetCDF files using the open_dataset() / open_dataarray() functions and the to_netcdf() method. Explore these in the exercise below!

.. seealso::

   - ENCCS course HPDA-Python: `Scientific data <https://enccs.github.io/hpda-python/scientific-data/>`_

Allocating RAM
--------------

- allocate many cores
    - within node only
    - shared memory
    - divide GB RAM  of the booked node with number of cores

.. admonition:: Do you remeber how to allocate several cores?
   :class: dropdown

   - Slurm flag ``-n <number of cores>``

.. admonition:: Do you remeber how to allocate several cores?
   :class: dropdown


   .. list-table:: Hardware
      :widths: 25 25 25 25 25 25 25
      :header-rows: 1

      * - Technology
        - Kebnekaise
        - Rackham
        - Snowy
        - Bianca
        - Cosmos  
        - Tetralith   
      * - Cores/compute node
        - 28 (72 for largemem, 128/256 for AMD Zen3/Zen4)
        - 20
        - 16
        - 16
        - 48  
        - 32  
      * - Memory/compute node
        - 128-3072 GB 
        - 128-1024 GB
        - 128-4096 GB
        - 128-512 GB
        - 256-512 GB  
        - 96-384 GB   
      * - GPU
        - NVidia V100, A100, A6000, L40s, H100, A40, AMD MI100 
        - None
        - NVidia T4 
        - NVidia A100
        - NVidia A100 
        - NVidia T4   

Exercise
--------

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

         - And install ``xarray`` to ``~/.local/`` if you don't already have it.

         .. code-block:: console
        
            pip install xarray

   .. tab:: NSC

      .. important::

         You should for this session load

         .. code-block:: console
        
            module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 JupyterLab/4.2.0

         - And install ``dask`` & ``xarray`` to ``~/.local/`` if you don't already have it

         .. code-block:: console
        
            pip install xarray dask

.. challenge:: Start interactive session with Jupyter
 
   .. tabs::

      .. tab:: HPC2N

         Jupyter notebooks for other purposes than just reading it, must be
         run in batch mode. First, create a batch script using the following one
         as a template: 

         .. code-block:: sh

            #!/bin/bash
            #SBATCH -A hpc2n20XX-XYZ
            #SBATCH -t 00:05:00
            #SBATCH -n 4
            #SBATCH -o output_%j.out   # output file
            #SBATCH -e error_%j.err    # error messages

            ml purge > /dev/null 2>&1
            ml GCC/12.3.0 OpenMPI/4.1.5 JupyterLab/4.0.5 dask/2023.9.2

            # Start JupyterLab
            jupyter lab --no-browser --ip $(hostname)

         Then, copy and paste the notebook located here ``Exercises/examples/Dask-Ini.ipynb`` to your
         current folder. Send the job to the queue (*sbatch job.sh*) and once the job starts copy the line 
         containing the string **http://b-cnyyyy.hpc2n.umu.se:8888/lab?token=** and paste it 
         in a local browser on Kebnekaise. Now you can select the notebook. 

      .. tab:: UPPMAX

         - To test this on UPPMAX it is easiest run in an **interactive session** started in a **ThinLinc session**
         - Also since Dask is installed already in ``Python/3.11.4``, we choose that version instead and run **jupyter-lab**.
         - The we can start a web browser from the login node on Thinlinc, either from the menu to the upper left or from a new terminal 

         - So, in Thinlinc, in a new terminal:

         .. code-block:: console

            $ interactive -A naiss2024-22-1442 -p devcore -n 4 -t 1:0:0
            $ deactivate # Be sure to deactivate you virtual environment
            $ cd <git-folder-for-course>
            $ ml python_ML_packages/3.11.8-cpu
            $ jupyter-lab --ip 0.0.0.0 --no-browser &

         - Copy the url in the output, containing the ``r<xxx>.uppmax.uu.se:8888/lab?token=<token-number>``, like for example:

            - Example: ``http://r484.uppmax.uu.se:8888/lab?token=5b72a4bbad15a617c8e75acf0528c70d12bb879807752893``
            - This address will certainly not work!

         - In ThinLinc, either start **Firefox** from the menu to the upper left 

            - or start a new terminal and type: ``firefox &``

         - Paste the url into the address field and press enter.
         - jupyter-lab starts
         - Double-click ``Dask-Ini.ipynb`` 
         - Restart kernel and run all cells!

      .. tab:: LUNARC

      .. tab:: NSC




Dask
----

.. image:: ../img/when-to-use-pandas.png
   :width: 600 px


- Dask is a array model extension and task scheduler. 
- By using the new array classes, you can automatically distribute operations across multiple CPUs.
- Dask is a library in Python for flexible parallel computing. 
- Among the features are the ability to deal with arrays and data frames, and the possibility of performing asynchronous computations, where first a computation graph is generated and the actual computations are activated later on demand.

Dask is very popular for data analysis and is used by a number of high-level
Python libraries:

   - Dask arrays scale NumPy (see also xarray)
   - Dask dataframes scale Pandas workflows
   - Dask-ML scales Scikit-Learn

- Dask divides arrays into many small pieces (chunks), as small as necessary to 
  fit it into memory. 
- Operations are delayed (lazy computing) e.g. tasks are queue and no computation 
  is performed until you actually ask values to be computed (for instance print mean values). 
- Then data is loaded into memory and computation proceeds in a streaming fashion, block-by-block.




Exercises
---------

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

.. challenge:: Chunk size

   The following example calculate the mean value of a random generated array. 
   Run the example and see the performance improvement by using dask.

   .. tabs::

      .. tab:: NumPy

         .. literalinclude:: example/chunk_np.py
            :language: python

      .. tab:: Dask

         .. literalinclude:: example/chunk_dask.py
            :language: python


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

.. seealso

   Working with data

   - https://aaltoscicomp.github.io/python-for-scicomp/work-with-data/
   
   Tidy data

   - https://coderefinery.github.io/data-visualization-python/tidy-data/
   
   ENCCS
   - Dask for scalable analysis
   - https://enccs.github.io/hpda-python/stack/
   - https://enccs.github.io/hpda-python/dask/ 

.. seealso:: 

   - `Dask documentation <https://docs.dask.org/en/stable/>`_
   - `Introduction to Dask by Aalto Scientific Computing and CodeRefinery <https://aaltoscicomp.github.io/python-for-scicomp/parallel/#dask-and-task-queues>`_
   - `Intermediate level Dask by ENCCS <https://enccs.github.io/hpda-python/dask/>`_.
   - Not tested yet at UPPMAX/HPC2N (?):

      - `Dask jobqueue <https://jobqueue.dask.org/en/latest/>`_
      - `Dask-MPI <http://mpi.dask.org/en/latest/index.html>`_



.. keypoints

   - Dask uses lazy execution
   - Only use Dask for processing very large amount of data

