.. _install-packages:
Install packages
================

.. objectives::

    - Learn how to install a (general-purpose) Python package with `pip`
    - Understand limitations of this way, e.g. use cases/best practices

Introduction
------------

There are 2-3 ways to install missing python packages at a HPC cluster.

- Local installation, always available for the version of Python you had active when doing the installation
    - ``pip install --user [package name]``
- Isolated environment. See next session

Normally you want reproducibility and the safe way to go is with isolated environments specific to your different projects.

.. admonition: Use cases of local general packages

   - General packages, missing in the environment of the loaded Python module
       - If you believe a package is useful for all your work
       - Ex. ``numpy`` is not installed, then make your own!


Typical workflow
................

1. Load the Python module with correct version.
    - Differs among the clusters

2. Check that the right python is used with ``which python3`` or ``which python``
    - Double check the version ``python3 -V`` or ``python -V``

3. Install with:  ``pip install --user [package name]`` 


- package name can be pinned, 
   - like ``numpy==1.26.4`` (Note the double ``==``)
   - like ``numpy>1.22``
   - read `more <https://peps.python.org/pep-0440/#version-specifiers>`_ 

- The package most often ends up in ``~/.local/lib/python3.X``
- Target directory can be changed by adding ``--prefix=[root_folder of installation]``

.. note::

   - Note that if you install for 3.11.X the package will not be seen by another minor version, like 3.12.X (or may not even be compatible with)
   - Note that installing with python 3.11.7 will end up in same folder as 3.11.5 and can be used by both bugfix versions.
   - Naming convention: python/major.minor.bugfix

Exercise
--------

.. challenge:: (optional) Exercise 1: Install a python package you know of for an old version 

   - We may add a solution in a coming instance of the course
