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

Why software modules are important
----------------------------------

Software modules allows users of any HPC cluster
to activate their favorite software of any version.
This helps to assure reproducible research.

Exercises
---------

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
| Rackham    |``3.12.7``       |
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
    | Rackham    |``module load python/3.12.7``                       |
    +------------+----------------------------------------------------+
    | Tetralith  |``module load Python/3.11.5-bare-hpc1-gcc-2023b-eb``|
    +------------+----------------------------------------------------+


- Confirm that the Python package, indicated in the table below, is absent.
  You can use any way to do so.

+------------+----------------+
| HPC cluster| Python package |
+============+================+
| Alvis      | ``scipy``      |
+------------+----------------+
| COSMOS     | ``scipy``      |
+------------+----------------+
| Dardel     | ``tensorflow`` |
+------------+----------------+
| Kebnekaise | ``scipy``      |
+------------+----------------+
| Rackham    | ``tensorflow`` |
+------------+----------------+
| Tetralith  | ``scipy``      |
+------------+----------------+

.. admonition:: Answer
    :class: dropdown

    On the terminal, type ``pip list`` to see all the
    packages that are installed.

    In all cases, the package is not yet installed,
    as that is what we'll be doing next :-)

- Find the software module to load the package. Use either
  the documentation of the HPC center, or use the module system

.. admonition:: Answer: where is this documented?
    :class: dropdown

    +------------+------------------------------------------------------------------------------------------------+
    | HPC cluster|URL to documentation                                                                            |
    +============+================================================================================================+
    | Alvis      |`Here <https://www.c3se.chalmers.se/documentation/module_system/python/#numpy-and-scipy>`__     |
    +------------+------------------------------------------------------------------------------------------------+
    | COSMOS     |`Here <https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/>`__    |
    +------------+------------------------------------------------------------------------------------------------+
    | Dardel     |`Here <https://support.pdc.kth.se/doc/applications/tensorflow/>`__, but it is irrelevant        |
    +------------+------------------------------------------------------------------------------------------------+
    | Kebnekaise |`Here <https://docs.hpc2n.umu.se/software/apps/#scipy>`__                                       |
    +------------+------------------------------------------------------------------------------------------------+
    | Rackham    |`Here <https://docs.uppmax.uu.se/software/tensorflow/#tensorflow-as-a-python-package-for-cpu>`__|
    +------------+------------------------------------------------------------------------------------------------+
    | Tetralith  |`Here <https://www.nsc.liu.se/software/python/>`__                                              |
    +------------+------------------------------------------------------------------------------------------------+

.. admonition:: Answer: how to use the module system?
    :class: dropdown

    In the terminal, type the command as shown below to get a decent hint.
    There are many possible terms to use with ``module spider``: whatever
    works for you is good too :-)

    +------------+--------------------------+
    | HPC cluster| Command                  |
    +============+==========================+
    | Alvis      |``module spider SciPy``   |
    +------------+--------------------------+
    | COSMOS     |``module spider SciPy``   |
    +------------+--------------------------+
    | Dardel     |``module spider package`` |
    +------------+--------------------------+
    | Kebnekaise |``module spider SciPy``   |
    +------------+--------------------------+
    | Rackham    |``module spider packages``|
    +------------+--------------------------+
    | Tetralith  |``module spider Python``  |
    +------------+--------------------------+


- Load the software module

.. admonition:: Answer
    :class: dropdown

    In the terminal, type the following command:

    +------------+--------------------------------------------------------------------------------------------------------------------+
    | HPC cluster| Command                                                                                                            |
    +============+====================================================================================================================+
    | Alvis      | ``module load SciPy-bundle/2024.05-gfbf-2024a``                                                                    |
    +------------+--------------------------------------------------------------------------------------------------------------------+
    | COSMOS     | ``module load module load GCC/13.3.0 SciPy-bundle/2024.05``                                                        |
    +------------+--------------------------------------------------------------------------------------------------------------------+
    | Dardel     | ``module load python_ML_packages/3.11.8-cpu``. You will be asked to do a ``module unload python`` first. Do so :-) |
    +------------+--------------------------------------------------------------------------------------------------------------------+
    | Kebnekaise | ``module load GCC/13.3.0 SciPy-bundle/2024.05``                                                                    |
    +------------+--------------------------------------------------------------------------------------------------------------------+
    | Rackham    | ``module load python_ML_packages/3.11.8-cpu``. You will be asked to do a ``module unload python`` first. Do so :-) |
    +------------+--------------------------------------------------------------------------------------------------------------------+
    | Tetralith  | ``module load Python/3.11.5``                                                                                      |
    +------------+--------------------------------------------------------------------------------------------------------------------+

- See the package is now present

.. admonition:: Answer
    :class: dropdown

    From the terminal, type ``pip list`` to see all the
    packages that are installed.

    In all cases, the package is now installed.
    Well done!
