.. meta::
   :description: Using packages
   :keywords: packages, modules, package modules

.. _use-packages:

Using packages
==============

.. admonition:: Learning outcomes

    - Practice using the documentation of your HPC cluster
    - Can find and load a Python package module
    - Can determine if a Python package is installed

Why Python packages are important
---------------------------------

Python packages are pieces of tested Python code.
Prefer using a Python package over writing your own code.

.. admonition:: Some definitions

   - Library: A collection of code used by a program.
   - Package: A library that has been made easily installable and reusable. Often published on public repositories such as the Python Package Index
   - Dependency: A requirement of another program, not included in that program.

What packages are out there
---------------------------

- Core numerics libraries: Ex ``numpy``
- Plotting: Ex ``matplotlib`` and ``seaborn``
- Data analysis and other important core packages: Ex ``pandas``, ``dask``, ``xarray``
- Interactive computing and human interface: Ex ``Jupyter``, ``spyder``
- Data format support and data ingestion: Ex ``h5py``
- Speeding up code and parallelism: Ex ``mpi4py``, ``numba``, ``dask``
- Machine learning: Ex ``scikit-learn``
- Deep learning: Ex ``pytorch``, ``tensorflow``, ``keras``

Plan of the week:

- Cover the use of the above packages in more or less detail

Why software modules are important on a HPC cluster
---------------------------------------------------

Software modules allows users of any HPC cluster
to activate their favorite software and/or packages of any version.
This helps to assure reproducible research.

How to see which Python packages are installed
----------------------------------------------

There are two ways to determine which Python packages are installed:

+-------------------------+------------------------------------------------+--------------------------------+
|Where                    |Command to run                                  |The package is present when ... |
+=========================+================================================+================================+
|On the command-line      |``pip list``                                    |It shows up in the list         |
+-------------------------+------------------------------------------------+--------------------------------+
|In the Python interpreter|``import [package_name]``, e.g. ``import scipy``|There is no error               |
+-------------------------+------------------------------------------------+--------------------------------+

Exercises
---------

.. admonition:: Want to see the answers as a video?
    :class: dropdown

    Some HPC clusters have multiple remote desktops. We recommend:

    +-----------+---------------------------------------+
    |HPC cluster|YouTube video                          |
    +===========+=======================================+
    |Alvis      |`Here <https://youtu.be/4ni7Z5NGRqQ>`__|
    +-----------+---------------------------------------+
    |Bianca     |`Here <https://youtu.be/-wOsA4yolNo>`__|
    +-----------+---------------------------------------+
    |COSMOS     |`Here <https://youtu.be/h-6qTaJ62vs>`__|
    +-----------+---------------------------------------+
    |Dardel     |`Here <https://youtu.be/GFsH3cx2rnY>`__|
    +-----------+---------------------------------------+
    |Kebnekaise |`Here <https://youtu.be/gziyoBeMLYo>`__|
    +-----------+---------------------------------------+
    |LUMI       |`Here <https://youtu.be/a7MEhsfMEIY>`__|
    +-----------+---------------------------------------+
    |Pelle      |`Here <https://youtu.be/eyee2tZgL8k>`__|
    +-----------+---------------------------------------+
    |Rackham    |`Here <https://youtu.be/NzjjNxsek54>`__|
    +-----------+---------------------------------------+
    |Tetralith  |`Here <https://youtu.be/TRnt07p1J94>`__|
    +-----------+---------------------------------------+


Exercise 1: using Python packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- login to your HPC cluster

.. admonition:: Forgot how to do this?
    :class: dropdown

    Answer can be found at
    `day 1 <https://uppmax.github.io/naiss_intro_python/sessions/using_the_python_interpreter/#exercise-1-login-to-your-hpc-cluster>`__

- load the Python module of the version below

+------------+-----------------+
| HPC cluster|Python version   |
+============+=================+
| Alvis      |``3.12.3``       |
+------------+-----------------+
| Bianca     |``3.12.7``       |
+------------+-----------------+
| COSMOS     |``3.11.5``       |
+------------+-----------------+
| Dardel     |``3.11.4``       |
+------------+-----------------+
| Kebnekaise |``3.11.3``       |
+------------+-----------------+
| LUMI       |``3.11.7``       |
+------------+-----------------+
| Pelle      |``3.12.3``       |
+------------+-----------------+
| Tetralith  |``3.11.5`` (bare)|
+------------+-----------------+

.. admonition:: Forgot how to do this?
    :class: dropdown

    +------------+----------------------------------------------------+
    | HPC cluster|Python version                                      |
    +============+====================================================+
    | Alvis      |``module load Python/3.12.3-GCCcore-13.3.0``        |
    +------------+----------------------------------------------------+
    | Bianca     |``module load python/3.12.7``                       |
    +------------+----------------------------------------------------+
    | COSMOS     |``module load GCCcore/13.2.0 Python/3.11.5``        |
    +------------+----------------------------------------------------+
    | Dardel     |``module load bioinfo-tools python/3.11.4``         |
    +------------+----------------------------------------------------+
    | Kebnekaise |``module load GCC/12.3.0 Python/3.11.3``            |
    +------------+----------------------------------------------------+
    | LUMI       |``module load cray-python/3.11.7``                  |
    +------------+----------------------------------------------------+
    | Pelle      |``module load Python/3.12.3-GCCcore-13.3.0``        |
    +------------+----------------------------------------------------+
    | Tetralith  |``module load Python/3.11.5-bare-hpc1-gcc-2023b-eb``|
    +------------+----------------------------------------------------+


- Confirm that the Python package, indicated in the table below, is absent.
  You can use any way to do so.

+------------+------------------------------+
| HPC cluster| Python package               |
+============+==============================+
| Alvis      | ``scipy``                    |
+------------+------------------------------+
| Bianca     | ``tensorflow`` (CPU version) |
+------------+------------------------------+
| COSMOS     | ``scipy``                    |
+------------+------------------------------+
| Dardel     | ``tensorflow``               |
+------------+------------------------------+
| Kebnekaise | ``scipy``                    |
+------------+------------------------------+
| LUMI       | ``matplotlib``               |
+------------+------------------------------+
| Pelle      | ``torch``                    |
+------------+------------------------------+
| Tetralith  | ``scipy``                    |
+------------+------------------------------+

.. admonition:: Answer
    :class: dropdown

    From the terminal, use the command below
    to confirm that the package is not available yet:

    +------------+-------------------------+
    | HPC cluster| Command                 |
    +============+=========================+
    | Alvis      |``pip list``             |
    +------------+-------------------------+
    | Bianca     |``pip list``             |
    +------------+-------------------------+
    | COSMOS     |``pip list``             |
    +------------+-------------------------+
    | Dardel     |``pip list``             |
    +------------+-------------------------+
    | Kebnekaise |``pip list``             |
    +------------+-------------------------+
    | LUMI       |``pip list``             |
    +------------+-------------------------+
    | Pelle      |``pip list``             |
    +------------+-------------------------+
    | Tetralith  |``pip list``             |
    +------------+-------------------------+

    In all cases, the package is not yet installed,
    as that is what we'll be doing next :-)

- Find the software module to load the package. Use either
  the documentation of the HPC center, or use the module system

.. admonition:: Answer: where is this documented?
    :class: dropdown

    +------------+----------------------------------------------------------------------------------------------------------------------------------------------+
    | HPC cluster|URL to documentation                                                                                                                          |
    +============+==============================================================================================================================================+
    | Alvis      |`Here <https://www.c3se.chalmers.se/documentation/module_system/python/#numpy-and-scipy>`__                                                   |
    +------------+----------------------------------------------------------------------------------------------------------------------------------------------+
    | Bianca     |`Here <https://docs.uppmax.uu.se/software/tensorflow/#tensorflow-as-a-python-package-for-cpu>`__                                              |
    +------------+----------------------------------------------------------------------------------------------------------------------------------------------+
    | COSMOS     |`Here <https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/>`__                                                  |
    +------------+----------------------------------------------------------------------------------------------------------------------------------------------+
    | Dardel     |`Here <https://support.pdc.kth.se/doc/applications/tensorflow/>`__, but it is irrelevant                                                      |
    +------------+----------------------------------------------------------------------------------------------------------------------------------------------+
    | Kebnekaise |`Here <https://docs.hpc2n.umu.se/software/apps/#scipy>`__                                                                                     |
    +------------+----------------------------------------------------------------------------------------------------------------------------------------------+
    | LUMI       |`Has no software modules <https://docs.lumi-supercomputer.eu/software/installing/python/#use-an-existing-container>`__                        |
    +            +----------------------------------------------------------------------------------------------------------------------------------------------+
    |            |`Use the thanard/matplotlib container <https://hub.docker.com/r/thanard/matplotlib>`__                                                        |
    +------------+----------------------------------------------------------------------------------------------------------------------------------------------+
    | Pelle      |`Python bundles <https://docs.uppmax.uu.se/software/python_bundles/#pytorch>`__                                                               |
    +------------+----------------------------------------------------------------------------------------------------------------------------------------------+
    | Tetralith  |`Here <https://www.nsc.liu.se/software/python/>`__                                                                                            |
    +------------+----------------------------------------------------------------------------------------------------------------------------------------------+

.. admonition:: Answer: how to use the module system?
    :class: dropdown

    In the terminal, type the command as shown below to get a decent hint.
    There are many possible terms to use with ``module spider``: whatever
    works for you is good too :-)

    +------------+---------------------------------------------------+
    | HPC cluster| Command                                           |
    +============+===================================================+
    | Alvis      |``module spider SciPy``                            |
    +------------+---------------------------------------------------+
    | Bianca     |``module spider <package>``                        |
    +------------+---------------------------------------------------+
    | COSMOS     |``module spider SciPy``                            |
    +------------+---------------------------------------------------+
    | Dardel     |``module spider <package>``                        |
    +------------+---------------------------------------------------+
    | Kebnekaise |``module spider SciPy``                            |
    +------------+---------------------------------------------------+
    | LUMI       |Has no module system, use a container instead.     |
    +            +---------------------------------------------------+
    |            |``singularity pull docker://thanard/matplotlib``   |
    +------------+---------------------------------------------------+
    | Pelle      |``module spider torch``                            |
    +------------+---------------------------------------------------+
    | Tetralith  |``module spider Python``                           |
    +------------+---------------------------------------------------+


- Load the software module

.. admonition:: Answer
    :class: dropdown

    In the terminal, type the following command:

    +------------+--------------------------------------------------------------------------------------------------------------------+
    | HPC cluster| Command                                                                                                            |
    +============+====================================================================================================================+
    | Alvis      | ``module load SciPy-bundle/2024.05-gfbf-2024a``                                                                    |
    +------------+--------------------------------------------------------------------------------------------------------------------+
    | Bianca     | ``module load python_ML_packages/3.9.5-cpu``. You will be asked to do a ``module unload python`` first. Do so :-)  |
    +------------+--------------------------------------------------------------------------------------------------------------------+
    | COSMOS     | ``module load module load GCC/13.3.0 SciPy-bundle/2024.05``                                                        |
    +------------+--------------------------------------------------------------------------------------------------------------------+
    | Dardel     | ``module load python_ML_packages/3.11.8-cpu``. You will be asked to do a ``module unload python`` first. Do so :-) |
    +------------+--------------------------------------------------------------------------------------------------------------------+
    | Kebnekaise | ``module load GCC/13.3.0 SciPy-bundle/2024.05``                                                                    |
    +------------+--------------------------------------------------------------------------------------------------------------------+
    | LUMI       | Not applicable: we are using a container                                                                           |
    +------------+--------------------------------------------------------------------------------------------------------------------+
    | Pelle      | ``module load PyTorch/2.6.0-foss-2024a``                                                                           |
    +------------+--------------------------------------------------------------------------------------------------------------------+
    | Tetralith  | ``module load buildtool-easybuild/4.9.4-hpc71cbb0050 GCC/13.2.0 SciPy-bundle/2023.11``                             |
    +            +--------------------------------------------------------------------------------------------------------------------+
    |            | Alternatively: ``module load Python/3.11.5`` (which happens to be a Python version with ``scipy`` installed)       |
    +------------+--------------------------------------------------------------------------------------------------------------------+

- See the package is now present

.. admonition:: Answer
    :class: dropdown

    From the terminal, use the command below
    to confirm that the package is now available:

    +------------+-------------------------+
    | HPC cluster| Command                 |
    +============+=========================+
    | Alvis      |``pip list``             |
    +------------+-------------------------+
    | Bianca     |``pip list``             |
    +------------+-------------------------+
    | COSMOS     |``pip list``             |
    +------------+-------------------------+
    | Dardel     |``pip list``             |
    +------------+-------------------------+
    | Kebnekaise |``pip list``             |
    +------------+-------------------------+
    | LUMI       |``./matplotlib pip list``|
    +------------+-------------------------+
    | Pelle      |``pip list``             |
    +------------+-------------------------+
    | Tetralith  |``pip list``             |
    +------------+-------------------------+

In all cases, the package is now installed.
Well done!



.. admonition:: **Done?**

    When done, and if you haven't done so yet,
    do :ref:`common-use-tarball`.

    After that, read what the next session is about.

    You can easily navigate there by pressing the 'Next' arrow
    at the bottom of this page, at the right-hand side

.. admonition:: Core numerics libraries

   - numpy - Arrays and array math.
   - scipy - Software for math, science, and engineering.

.. admonition:: Plotting

   - matplotlib - Base plotting package, somewhat low level but almost everything builds on it.
   - seaborn - Higher level plotting interface; statistical graphics.
   - Vega-Altair - Declarative Python plotting.
   - mayavi - 3D plotting
   - Plotly - Big graphing library.

.. admonition:: Data analysis and other important core packages

   - pandas - Columnar data analysi.
   - polars <https://pola.rs/> - Alternative to pandas that uses similar API, but is re-imagined for more speed.
   - Vaex - Alternative for pandas that uses similar API for lazy-loading and processing huge DataFrames.
   - Dask - Alternative to Pandas that uses similar API and can do analysis in parallel.
   - xarrray - Framework for working with mutli-dimensional arrays.
   - statsmodels - Statistical models and tests.
   - SymPy - Symbolic math.
   - networkx - Graph and network analysis.
   - graph-tool - Graph and network analysis toolkit implemented in C++.

.. admonition:: Interactive computing and human interface

   - Interactive computing
      - IPython - Nicer interactive interpreter
      - Jupyter - Web-based interface to IPython and other languages (includes projects such as jupyter notebook, lab, hub, …)
   - Testing
      - pytest - Automated testing interface
   - Documentation
      - Sphinx - Documentation generator (also used for this lesson…)
   - Development environments
      - Spyder - Interactive Python development environment.
      - Visual Studio Code - Microsoft’s flagship code editor.
      - PyCharm - JetBrains’s Python IDE.
   - Binder - load any git repository in Jupyter automatically, good for reproducible research

.. admonition:: Data format support and data ingestion

   - pillow - Image manipulation. The original PIL is no longer maintained, the new “Pillow” is a drop-in replacement.
   - h5py and PyTables - Interfaces to the HDF5 file format.

.. admonition:: Speeding up code and parallelism

   - MPI for Python (mpi4py) - Message Passing Interface (MPI) in Python for parallelizing jobs.
   - cython - easily make C extensions for Python, also interface to C libraries
   - numba - just in time compiling of functions for speed-up
   - PyPy - Python written in Python so that it can internally optimize more.
   - Dask - Distributed array data structure for distributed computation
   - Joblib - Easy embarrassingly parallel computing
   - IPyParallel - Easy parallel task engine.
   - numexpr - Fast evaluation of array expressions by automatically compiling the arithmetic.

.. admonition:: Machine learning

   - nltk - Natural language processing toolkit.
   - scikit-learn - Traditional machine learning toolkit.
   - xgboost - Toolkit for gradient boosting algorithms.

.. admonition:: Deep learning

   - tensorflow - Deep learning library by Google.
   - pytorch - Currently the most popular deep learning library.
   - keras - Simple libary for doing deep learning.
   - huggingface - Ecosystem for sharing and running deep learning models and datasets. Incluses packages like transformers, datasets, accelerate, etc.
   - jax - Google’s Python library for running NumPy and automatic differentiation on GPUs.
   - flax - Neural network framework built on Jax.
   - equinox - Another neural network framework built on Jax.
   - DeepSpeed - Algorithms for running massive scale trainings. Included in many of the frameworks.
   - PyTorch Lightning - Framework for creating and training PyTorch models.
   - Tensorboard <https://www.tensorflow.org/tensorboard/> - Tool for visualizing model training on a web page.

.. admonition:: Other packages for special cases

   - dateutil and pytz - Date arithmetic and handling, timezone database and conversion.




