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
    |COSMOS     |``TODO``                               |
    +-----------+---------------------------------------+
    |Dardel     |``TODO``                               |
    +-----------+---------------------------------------+
    |Kebnekaise |``TODO``                               |
    +-----------+---------------------------------------+
    |LUMI       |``TODO``                               |
    +-----------+---------------------------------------+
    |Rackham    |``TODO``                               |
    +-----------+---------------------------------------+
    |Tetralith  |``TODO``                               |
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
    | LUMI       |``module load cray-python/3.11.7``                  |
    +------------+----------------------------------------------------+
    | Rackham    |``module load python/3.12.7``                       |
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
| Rackham    | ``tensorflow`` (CPU version) |
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
    | Rackham    |``pip list``             |
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
    | Rackham    |`Here <https://docs.uppmax.uu.se/software/tensorflow/#tensorflow-as-a-python-package-for-cpu>`__                                              |
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
    | Bianca     |``module spider packages``                         |
    +------------+---------------------------------------------------+
    | COSMOS     |``module spider SciPy``                            |
    +------------+---------------------------------------------------+
    | Dardel     |``module spider package``                          |
    +------------+---------------------------------------------------+
    | Kebnekaise |``module spider SciPy``                            |
    +------------+---------------------------------------------------+
    | LUMI       |Has no module system, use a container instead.     |
    +            +---------------------------------------------------+
    |            |``singularity pull docker://thanard/matplotlib``   |
    +------------+---------------------------------------------------+
    | Rackham    |``module spider packages``                         |
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
    | Rackham    | ``module load python_ML_packages/3.11.8-cpu``. You will be asked to do a ``module unload python`` first. Do so :-) |
    +------------+--------------------------------------------------------------------------------------------------------------------+
    | Tetralith  | ``module load Python/3.11.5``                                                                                      |
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
    | Rackham    |``pip list``             |
    +------------+-------------------------+
    | Tetralith  |``pip list``             |
    +------------+-------------------------+

In all cases, the package is now installed.
Well done!
