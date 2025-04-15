.. meta::
   :description: Using packages
   :keywords: packages, modules, day 2

.. _use-packages:
Using packages
==============

!!! info "Learning outcomes"

    - Practice using the documentation of your HPC cluster
    - Find installed Python packages using `pip list`
    - Find a software module with Python packages
    - Load a software module with Python packages

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
    `day 1 <https://uppmax.github.io/naiss_intro_python/sessions/using_the_python_interpreter/#exercise-1-login-to-your-hpc-cluster>`_

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
| LUMI       | ``TBA``        |
+------------+----------------+
| Rackham    | ``3.12.7``     |
+------------+----------------+
| Tetralith  | ``3.10.4``     |
+------------+----------------+

.. admonition:: Forgot how to do this?
    :class: dropdown

    Answer can be found at
    `day 1 <https://uppmax.github.io/naiss_intro_python/sessions/using_the_python_interpreter/#exercise-2-load-the-python-module>`_


- Confirm that the Python package, indicated in the table below, is absent.
  You can use any way to do so.

+------------+----------------+
| HPC cluster| Python package |
+============+================+
| Alvis      | ?              |
+------------+----------------+
| COSMOS     | ?              |
+------------+----------------+
| Dardel     | ?              |
+------------+----------------+
| Kebnekaise | ?              |
+------------+----------------+
| LUMI       | ?              |
+------------+----------------+
| Rackham    | ?              |
+------------+----------------+
| Tetralith  | ?              |
+------------+----------------+

.. admonition:: Answer
    :class: dropdown

    Within the Python interpreter, type ``pip list`` to see all the
    packages that are installed.

    In all cases, the package is not yet installed,
    as that is what we'll be doing next :-)

- Find the software module to load the package. Use either
  the documentation of the HPC center, or use the module system

.. admonition:: Answer: where is this documented?
    :class: dropdown

    +------------+----------------------+
    | HPC cluster| URL to documentation |
    +============+======================+
    | Alvis      | ?                    |
    +------------+----------------------+
    | COSMOS     | ?                    |
    +------------+----------------------+
    | Dardel     | ?                    |
    +------------+----------------------+
    | Kebnekaise | ?                    |
    +------------+----------------------+
    | LUMI       | ?                    |
    +------------+----------------------+
    | Rackham    | ?                    |
    +------------+----------------------+
    | Tetralith  | ?                    |
    +------------+----------------------+

.. admonition:: Answer: how to use the module system?
    :class: dropdown

    In the terminal, type the following command:

    +------------+----------------------+
    | HPC cluster| Command              |
    +============+======================+
    | Alvis      | ``module spider ?``  |
    +------------+----------------------+
    | COSMOS     | ``module spider ?``  |
    +------------+----------------------+
    | Dardel     | ``module spider ?``  |
    +------------+----------------------+
    | Kebnekaise | ``module spider ?``  |
    +------------+----------------------+
    | LUMI       | ``?``                |
    +------------+----------------------+
    | Rackham    | ``module spider ?``  |
    +------------+----------------------+
    | Tetralith  | ``module spider ?``  |
    +------------+----------------------+

- Load the software module

.. admonition:: Answer
    :class: dropdown

    In the terminal, type the following command:

    +------------+----------------------+
    | HPC cluster| Command              |
    +============+======================+
    | Alvis      | ``module load ?``    |
    +------------+----------------------+
    | COSMOS     | ``module load ?``    |
    +------------+----------------------+
    | Dardel     | ``module load ?``    |
    +------------+----------------------+
    | Kebnekaise | ``module load ?``    |
    +------------+----------------------+
    | LUMI       | ``?``                |
    +------------+----------------------+
    | Rackham    | ``module load ?``    |
    +------------+----------------------+
    | Tetralith  | ``module load ?``    |
    +------------+----------------------+

- See the package is now present

.. admonition:: Answer
    :class: dropdown

    Within the Python interpreter, type ``pip list`` to see all the
    packages that are installed.

    In all cases, the package is now installed.
    Well done!
