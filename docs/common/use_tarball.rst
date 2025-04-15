.. _common-use-tarball:

Use the tarball with exercises
==============================

.. admonition:: Goal

    You can run the example files needed for the exercises 

A tarball is a file that contains multiple files,
similar to a zip file.
To use the files it contains, it needs to be untarred/unzipped/uncompressed
first.

Procedure
---------

.. admonition:: Prefer a video?
    :class: dropdown

    See **FIX LINK**...

The procedure has these steps:

- Don't you have a work directory yet?
    - :ref:`work-directory`

- Get the tarball
- Uncompress the tarball

Step 1: navigate to your project folder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- On your favorite HPC cluster, navigate to the project folder
  of this course. See :ref:`work-directory` for where these are.

.. admonition:: Answer
    :class: dropdown

    In the terminal, type the following command:

    +------------+------------------------------------------------------------+
    | HPC cluster| Course project                                             |
    +============+============================================================+
    | COSMOS     | ``cd ~``                                                   |
    +------------+------------------------------------------------------------+
    | Dardel     | ``cd /cfs/klemming/projects/snic/hpc-python-spring-naiss`` |
    +------------+------------------------------------------------------------+
    | Kebnekaise | ``cd /proj/nobackup/hpc-python-spring``                    |
    +------------+------------------------------------------------------------+
    | Rackham    | ``cd /proj/hpc-python-uppmax``                             |
    +------------+------------------------------------------------------------+
    | Tetralith  | ``cd /proj/hpc-python-spring-naiss/users/``                |
    +------------+------------------------------------------------------------+

Step 2: create a folder for yourself
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- In the project folder of this course, create a folder for yourself.
  For example, if your name is Sven, create a folder called ``sven``

.. admonition:: Answer
    :class: dropdown

    In the terminal, type the following command:

    ``mkdir sven``

Step 3: navigate inside that folder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- From the project folder of this course, 
  navigate inside your personal folder

.. admonition:: Answer
    :class: dropdown

    In the terminal, type the following command:

    ``cd sven``

Step 4: get the tarball
^^^^^^^^^^^^^^^^^^^^^^^

In the terminal, type the following command:

``wget https://github.com/UPPMAX/HPC-python/raw/refs/heads/main/exercises.tar.gz``

Step 5: Uncompress the tarball
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the terminal, type the following command:

``tar -xvzf exercises.tar.gz``

Step 6: Navigate in the folder of that day
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            
After decompressing, there is a folder called  ``day2``, or ``day3`` or ``day4``
that contains the exercises. Navigate into that folder.

.. admonition:: Answer
    :class: dropdown

    In the terminal, type the following command for day 2:

    ``cd day2``
