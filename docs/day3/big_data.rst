Big data with Python
====================

.. admonition:: "Learning outcomes"

   Learners 

   - can decide on useful file formats
   - can allocate resources sufficient to data size
   - can use data-chunking as technique
   - know where to learn more
    
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

   Any changes and why?

Allocating RAM
--------------

- Mention memory per core considerations.
- Show SLURM options for memory and time.
- Briefly explain what happens when a Dask job runs on multiple cores.

.. admonition:: Keywords

   OOM

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


