.. _install-packages:
Install packages
================

!!! info "Learning objectives"

    - Learn how to install a (general-purpose) Python package with `pip`
    - Understand limitations of this way, e.g. use cases/best practices

Introduction
------------

There are 2-3 ways to install missing python packages at a HPC cluster.

- Local installation, always available for the version of Python you had active when doing the installation
    - ``pip install --user [package name]``
- Isolated environment. See next session

.. note::

   - The package most often end up in ``~/.local/lib/python3.X``
   - Note that if you install for 3.11.X the package will not be seen by another minor version, like 3.12.X (or may not even be compatible with)
   - Note that installing with python 3.11.7 will end up in same folder as 3.11.5 and can be used by both bugfix versions.
   - Naming convention: python/major.minor.bugfix


Normally you want reproducibility and the safe way to go is with isolated environments specific to your different projects.


.. admonition: Use cases of local general packages

   - General packages, missing in the environent of the loaded Python module
       - If you believe a package is useful for all your work
       - Ex. ``numpy`` is not natively installed, then make your own!
   - Your installed package will only be availbale for the version 


Typical workflow
................

.. note::

   ``pip install --user [package name]`` 

    - The package most often end up in ``~/.local``
    - target directory can be changed by ``--prefix=[root_folder of installation]``

Exercise

We do not provide any here For some clusters you may prepare your course environment in this way.
