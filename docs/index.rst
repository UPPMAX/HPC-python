.. Python-at-UPPMAX documentation master file, created by
   sphinx-quickstart on Fri Jan 21 18:24:04 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
Welcome to HPC python course material
=====================================

.. admonition:: This material
   
   Here you will find the content of the workshop Using Python in an HPC environment.
   
   - Documentation at the HPC centres UPPMAX and HPC2N
      - UPPMAX: https://www.uppmax.uu.se/support/user-guides/python-user-guide/
      - HPC2N: https://www.hpc2n.umu.se/resources/software/user_installed/python


.. admonition:: Content

   - This course aims to give a brief, but comprehensive introduction to using Python in an HPC environment.
   - You will learn how to
      - use modules to load Python
      - find site installed Python packages
      - install packages yourself
      - use virtual environments, 
      - write a batch script for running Python
      - use Python in parallel
      - how to use Python for ML and on GPUs. 
   - This course will consist of lectures interspersed with hands-on sessions where you get to try out what you have just learned.    


.. admonition:: Cluster-specific approaches

   - The course is a cooperation between UPPMAX (Rackham, Snowy, Bianca) and HPC2N (Kebnekaise). The main focus will be on UPPMAX's systems, but Kebnekaise will be included as well.  
   - In most cases there is little or no difference between UPPMAX's systems and HPC2N's systems, except naming of modules and such. We will mention (and cover briefly) instances when there are larger differences.  


Preliminary schedule
====================

.. list-table:: Preliminary schedule
   :widths: 25 25 50
   :header-rows: 1

   * - Time
     - Topic
     - Activity
   * - 9:00
     - Syllabus 
     -
   * - 9:10
     - Introduction 
     - Lecture
   * - 9:20
     - Loading modules and running Python codes 
     - Lecture + type-along 
   * - 9:35
     - Dealing with packages  
     - Lecture + type-along 
   * - 9:55
     - **Coffee**
     - 
   * - 10:10
     - Dealing with Conda  
     - Lecture + type-along + exercise
   * - 10:30
     - Isolated environments
     - Lecture + type-along + exercise
   * - 10:50
     - **Short leg stretch**
     - 
   * - 10:55
     - SLURM Batch scripts for Python jobs  
     - Lecture + type-along + exercise
   * - 11:20
     - Interactive
     - Lecture + type-along
   * - 11:40
     - Catch-up time and Q/A (no recording)
     - Q/A
   * - 12:00
     - **LUNCH**
     -
   * - 13:00
     - Parallelising simple Python codes
     - Lecture + type-along + exercise
   * - 13:40
     - Using GPU:s for Python
     - Lecture + type-along + exercise
   * - 14:10
     - **Short leg stretch**
     - 
   * - 14:15
     - Using Python for Machine Learning jobs
     - Lecture + type-along + exercise
   * - 14:55
     - **Coffee**
     - 
   * - 15:10
     - Summary 
     -
   * - 15:15
     - Extra time for exercises (no recording)
     - exercises 
   * - 15:35
     - Q&A on-demand (no recording)
     -

   * - 16.00
     - END
     -
    

.. toctree::
   :maxdepth: 2
   :caption: Lessons!

   intro.rst
   load_run.rst
   packages.rst
   isolated.rst
   batch.md
   interactive.md
   parallel.rst
   gpu.md
   ml.md
   summary.rst

.. toctree::
   :maxdepth: 2
   :caption: Extra reading:

   packages_deeper.rst
   isolated_deeper.rst
   jupyter.md
   ML_deeper.rst
   uppmax.rst
   kebnekaise.md
   bianca.rst
   
.. toctree::
   :maxdepth: 2
   :caption: Exercises:

   exercises.rst

  

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
