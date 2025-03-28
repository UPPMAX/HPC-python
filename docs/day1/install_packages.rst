Install packages
================

!!! info "Learning objectives"

    - Install a Python package with `pip`

<!-- below is still old -->

.. objectives:: 

   Learners can 

   - work (create, activate, work, deactivate) with virtual environments
   - install a python package
   - export and import a virtual environment

Introduction
------------

There are 2-3 ways to install missing python packages at a HPC cluster.

- Local installation, always available for the version of Python you had active when doing the installation
    - ``pip install --user [package name]``
- Isolated environment. Use some packages just needed for a specific use case.
    - ``venv``/``virtualenv`` in combination with ``pip`` 
        - recommended/working in all HPC centers in Sweden
    - ``conda``
        - just recommended in some HPC centers in Sweden


Local (general installation)
............................

.. note::

   ``pip install --user [package name]`` 

    - The package end up in ``~/.local``
    - target directory can be changed by ``--prefix=[root_folder of installation]``
