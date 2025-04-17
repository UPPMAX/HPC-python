.. _install-packages:
Install packages
================

!!! info "Learning objectives"

    - Practice using the documentation of your HPC cluster
    - Install a (general-purpose) Python package with `pip`
    - <!-- RB: this is in the tarball exercise too --> Be able to navigate to the project folder (except for the COSMOS HPC cluster)
    - <!-- RB: this is in the tarball exercise too --> Be able to create a subfolder in the project folder
    - Understand limitations of this way, e.g. use cases/best practices

<!-- exercise:

HPC cluster| Who creates answer
-----------|----------------------------------------------
Alvis      | RB
Bianca     | BC, because use conda to install packages
COSMOS     | RB
Dardel     | BC
Kebnekaise | BC
LUMI       | RB
Rackham    | RB
Tetralith  | BC



    - Confirm a package is not there
    - Install it
    - Confirm the package is there

-->

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
