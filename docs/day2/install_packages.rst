.. _install-packages:
Install packages
================

.. objectives::

    - Learn how to install a (general-purpose) Python package with `pip`
    - Understand limitations of this way, e.g. use cases/best practices

Introduction
------------

There are 2 ways to install missing python packages at a HPC cluster.

- Local installation, always available for the version of Python you had active when doing the installation
    - ``pip install --user [package name]``
- Isolated environment. See next session
    - virtual environents provided by python
    - conda

Normally you want reproducibility and the safe way to go is with isolated environments specific to your different projects.

.. admonition: Use cases of local general packages

   - Packages, missing in the loaded Python module
       - Ex: You usually use 3D data and ``xarray`` is not installed
 
Typical workflow
................

1. Load the Python module with correct version.
    - Differs among the clusters

2. Check that the right python is used with ``which python3`` or ``which python``
    - Double check the version ``python3 -V`` or ``python -V``

3. Install with:  ``pip install --user [package name]`` 

Versions
........

- Package name can be pinned, 
   - like ``numpy==1.26.4`` (Note the double ``==``)
   - like ``numpy>1.22``
   - read `more <https://peps.python.org/pep-0440/#version-specifiers>`_ 

- If not pinned you will get the latest version compatible with the python version you are using.

Installation directory
......................

- The package most often ends up in ``~/.local/lib/python3.X``
- Target directory can be changed by adding ``--prefix=[root_folder of installation]``

.. note::

   - Note that if you install for 3.11.X the package will not be seen by another minor version, like 3.12.X (or may not even be compatible with)
   - Note that installing with python 3.11.7 will end up in same folder as 3.11.5 and can be used by both bugfix versions.
   - Naming convention: python/major.minor.bugfix

Exercise
--------

.. challenge:: (optional) Exercise 1: Install a python package you know of for an old version of Python

   - Load an older Python module (perhaps one you won't use anymore)
   - install the python package (it may already be there but with an older version)
       - (you can always remove your local installation later if you regret it)

   - We may add a solution in a coming instance of the course



.. admonition:: Already installed Python packages in HPC and ML
   :class: dropdown

   It is difficult to give an exhaustive list of useful packages for Python in HPC, but this list contains some of the more popular ones: 

   .. list-table:: Popular packages
      :widths: 8 10 10 20 
      :header-rows: 1

      * - Package
        - Module to load, UPPMAX
        - Module to load, HPC2N
        - Brief description 
      * - Dask
        - ``python``
        - ``dask``
        - An open-source Python library for parallel computing.
      * - Keras
        - ``python_ML_packages``
        - ``Keras``
        - An open-source library that provides a Python interface for artificial neural networks. Keras acts as an interface for both the TensorFlow and the Theano libraries. 
      * - Matplotlib
        - ``python`` or ``matplotlib``
        - ``matplotlib``
        - A plotting library for the Python programming language and its numerical mathematics extension NumPy.
      * - Mpi4Py
        - Not installed
        - ``SciPy-bundle``
        - MPI for Python package. The library provides Python bindings for the Message Passing Interface (MPI) standard.
      * - Numba 
        - ``python``
        - ``numba``
        - An Open Source NumPy-aware JIT optimizing compiler for Python. It translates a subset of Python and NumPy into fast machine code using LLVM. It offers a range of options for parallelising Python code for CPUs and GPUs. 
      * - NumPy
        - ``python``
        - ``SciPy-bundle``
        - A library that adds support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. 
      * - Pandas
        - ``python`` 
        - ``SciPy-bundle``
        - Built on top of NumPy. Responsible for preparing high-level data sets for machine learning and training. 
      * - PyTorch/Torch
        - ``PyTorch`` or ``python_ML_packages``
        - ``PyTorch``
        - PyTorch is an ML library based on the C programming language framework, Torch. Mainly used for natural language processing or computer vision.  
      * - SciPy
        - ``python``
        - ``SciPy-bundle``
        - Open-source library for data science. Extensively used for scientific and technical computations, because it extends NumPy (data manipulation, visualization, image processing, differential equations solver).  
      * - Seaborn 
        - ``python``
        - Not installed
        - Based on Matplotlib, but features Pandas’ data structures. Often used in ML because it can generate plots of learning data. 
      * - Sklearn/SciKit-Learn
        - ``scikit-learn``
        - ``scikit-learn``
        - Built on NumPy and SciPy. Supports most of the classic supervised and unsupervised learning algorithms, and it can also be used for data mining, modeling, and analysis. 
      * - StarPU
        - Not installed 
        - ``StarPU``
        - A task programming library for hybrid architectures. C/C++/Fortran/Python API, or OpenMP pragmas. 
      * - TensorFlow
        - ``TensorFlow``
        - ``TensorFlow``
        - Used in both DL and ML. Specializes in differentiable programming, meaning it can automatically compute a function’s derivatives within high-level language. 
      * - Theano 
        - Not installed 
        - ``Theano``
        - For numerical computation designed for DL and ML applications. It allows users to define, optimise, and gauge mathematical expressions, which includes multi-dimensional arrays.  

   Remember, in order to find out how to load one of the modules, which prerequisites needs to be loaded, as well as which versions are available, use ``module spider <module>`` and ``module spider <module>/<version>``. 

   Often, you also need to load a python module, except in the cases where it is included in ``python`` or ``python_ML_packages`` at UPPMAX or with ``SciPy-bundle`` at HPC2N. 

   NOTE that not all versions of Python will have all the above packages installed! 

