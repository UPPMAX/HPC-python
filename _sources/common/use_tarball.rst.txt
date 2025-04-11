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

Step 1: get the tarball
^^^^^^^^^^^^^^^^^^^^^^^

- In a terminal, ``cd`` to a good directory to keep the exercises (for instance in your just created folder in the project directory)
- You may create a new folder (``mkdir``), called ``exercises`` or similar).
- Use the following command to download the file to your current folder:

.. tabs::

   .. tab:: day2

      **FIX**

      .. code-block::  console

          wget ... 

   .. tab:: day3 (wait until that day)

      .. code-block::  console

          wget ...

   .. tab:: day4 (wait until that day)

      .. code-block::  console

          wget ...

.. admonition:: How does that look like?
   :class: dropdown

   Your output will look somewhat like  this:

   **FIX**

    .. code-block::  console

        [sven@rackham3 ~]$ wget 

Step 2: Uncompress the tarball
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In a terminal, use the following command to uncompress the file:

**FIX**  

.. tabs::

   .. tab:: day 2

      .. code-block::  console

         tar -xvzf exercisesDay2.tar.gz 

   .. tab:: day 3 (wait until that day)

      .. code-block::  console

         tar -xvzf exercisesDay3.tar.gz 

   .. tab:: day 4 (wait until that day)

      .. code-block::  console

         tar -xvzf exercisesDay4.tar.gz 
            
After decompressing, there is a folder called  ``day2``, or ``day3`` or ``day4``
that contains the exercises.

.. warning:: Do you want the whole repo?

   - If you are happy with just the exercises, the tarballs of the language specific ones are enough.
   - By cloning the whole repo, you get all the materials, planning documents, and exercises.
   - If you think this makes sense type this in the command line in the directory you want it.
     - ``git clone https://github.com/UPPMAX/HPC-python.git``
   - Note however, that if you during exercise work modify files, they will be overwritten if you make ``git pull`` (like if the teacher needs to modify something).
      - Then make a copy somewhere else with your answers!



