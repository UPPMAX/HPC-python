.. meta::
   :description: Using packages
   :keywords: packages, modules, day 2

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

+------------+----------------+
| HPC cluster| Python version |
+============+================+
| Alvis      | ``3.12.3``     |
+------------+----------------+
| Bianca     | ``3.11.4``     |
+------------+----------------+
| COSMOS     | ``3.11.5``     |
+------------+----------------+
| Dardel     | ``3.11.4``     |
+------------+----------------+
| Kebnekaise | ``3.11.3``     |
+------------+----------------+
| Rackham    | ``3.12.7``     |
+------------+----------------+
| Tetralith  | ``3.10.4``     |
+------------+----------------+

.. admonition:: Forgot how to do this?
    :class: dropdown

    Answer can be found at
    `day 1 <https://uppmax.github.io/naiss_intro_python/sessions/using_the_python_interpreter/#exercise-2-load-the-python-module>`__

    .. note to self

        HPC Cluster|Link to documentation                                                                              |Solution
        -----------|---------------------------------------------------------------------------------------------------|------------------------------------------------------
        Alvis      |[short](https://www.c3se.chalmers.se/documentation/module_system/python_example/) or [long](https://www.c3se.chalmers.se/documentation/module_system/modules/) |`module load Python/3.12.3-GCCcore-13.3.0`
        Bianca     |[here](https://docs.uppmax.uu.se/software/python/#loading-python)                                  |`module load python/3.11.4`
        COSMOS     |[here](https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/)          |`module load GCCcore/13.2.0 Python/3.11.5`
        Dardel     |:warning: [here](https://support.pdc.kth.se/doc/software/module/) and [here](https://support.pdc.kth.se/doc/applications/python/)    |`module load bioinfo-tools python/3.11.4`
        Kebnekaise |[here](https://docs.hpc2n.umu.se/software/userinstalls/#python__packages)                          |`module load GCC/12.3.0 Python/3.11.3`
        LUMI       |:warning: [here](https://docs.lumi-supercomputer.eu/software/installing/python/)                   |Unknown
        Rackham    |[here](http://docs.uppmax.uu.se/software/python/)                                                  |`module load python`
        Tetralith  |[here](https://www.nsc.liu.se/software/python/)                                                    |`module load Python/3.10.4-env-hpc2-gcc-2022a-eb`



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
| Rackham    | ?              |
+------------+----------------+
| Tetralith  | ?              |
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

    +------------+---------------------------------------------------------------------------------------------+
    | HPC cluster| URL to documentation                                                                        |
    +============+=============================================================================================+
    | Alvis      | `Here <https://www.c3se.chalmers.se/documentation/module_system/python/#numpy-and-scipy>`__ |
    +------------+---------------------------------------------------------------------------------------------+
    | COSMOS     | `Here <https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/>`__|
    +------------+---------------------------------------------------------------------------------------------+
    | Dardel     | `Here <https://support.pdc.kth.se/doc/applications/tensorflow/>`__, but it is irrelevant    |
    +------------+---------------------------------------------------------------------------------------------+
    | Kebnekaise | `here <https://docs.hpc2n.umu.se/software/apps/#scipy>`__                                   |
    +------------+---------------------------------------------------------------------------------------------+
    | Rackham    | ?                                                                                           |
    +------------+---------------------------------------------------------------------------------------------+
    | Tetralith  | ?                                                                                           |
    +------------+---------------------------------------------------------------------------------------------+

.. admonition:: Answer: how to use the module system?
    :class: dropdown

    In the terminal, type the command as shown below to get a decent hint.
    There are many possible terms to use with ``module spider``: whatever
    works for you is good too :-)

    +------------+--------------------------+
    | HPC cluster| Command                  |
    +============+==========================+
    | Alvis      | ``module spider SciPy``  |
    +------------+--------------------------+
    | COSMOS     | ``module spider SciPy``  |
    +------------+--------------------------+
    | Dardel     | ``module spider package``|
    +------------+--------------------------+
    | Kebnekaise | ``module spider SciPy``  |
    +------------+--------------------------+
    | Rackham    | ``module spider ?``      |
    +------------+--------------------------+
    | Tetralith  | ``module spider ?``      |
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
    | Rackham    | ``module load ?``                                                                                                  |
    +------------+--------------------------------------------------------------------------------------------------------------------+
    | Tetralith  | ``module load ?``                                                                                                  |
    +------------+--------------------------------------------------------------------------------------------------------------------+

- See the package is now present

.. admonition:: Answer
    :class: dropdown

    From the terminal, type ``pip list`` to see all the
    packages that are installed.

    In all cases, the package is now installed.
    Well done!
