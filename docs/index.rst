.. Python-at-UPPMAX documentation master file, created by
   sphinx-quickstart on Fri Jan 21 18:24:04 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
Welcome to HPC python course material
=====================================

.. admonition:: This material
   
   Here you will find the content of the workshop Using Python in an HPC environment.

.. admonition:: Content

   - This course aims to give a brief, but comprehensive introduction to using Python in an HPC environment.      - You will learn how to
      - use modules to load Python
      - find site installed Python packages
      - install packages yourself
      - use virtual environments, 
      - write a batch script for running Python
      - use Python in parallel
      - how to use Python for ML and on GPUs. 
   - This course will consist of lectures interspersed with hands-on sessions where you get to try out what you have just learned.    


.. admonition:: Cluster-specific approaches

   - The course is a cooperation between UPPMAX (Rackham, Snowy, Bianca) and HPC2N (Kebnekaise) and will focus on the compute systems at both centres. 
   - For the site-specific part of the course you will be divided into groups depending on which center you will be running your code, as the approach is somewhat different. 


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
   * - 
     - Introduction 
     - Lecture
   * - 
     - Loading modules and running Python codes 
     - Lecture+code along
   * - 
     - Dealing with packages  
     - Lecture+code along
   * - 
     - **Coffee**
     - 
   * - 
     - Isolated environments (general)
     - Lecture+code along
   * - 
     - SLURM Batch scripts for Python jobs  
     - Lecture+code along + exercise
   * - 
     - Interactive
     - Lecture+code along
   * - 
     - **Short leg stretch**
     - 
   * - 
     - Separated sessions 1 for UPPMAX/HPC2N:isolated environments
     - Lecture
   * - 12:00
     - LUNCH 
     -
   * - 13:00
     - Parallelising a simple Python code  
     - Lecture+code along + exercise
   * - 
     - Depoying GPU:s for Python
     - Lecture+code along + exercise
   * - 
     - **Short leg stretch**
     - 
   * - 
     - Using Python for Machine Learning jobs
     - Lecture+code along
   * - 
     - Separated sessions 2 for UPPMAX/HPC2N: Bianca/ML exercises
     - Lecture
   * - 
     - **Coffee**
     - 
   * - 
     - Summary 
     -
   * - 
     - Q&A on-demand
     -
   * - 
     - Additional exercises 
     -
   * - 16.00
     - END
    

.. toctree::
   :maxdepth: 2
   :caption: Lessons:

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
   :caption: UPPMAX Sessions:

   uppmax.rst
   isolatedUPPMAX.rst
   conda.rst
   bianca.rst

.. toctree::
   :maxdepth: 2
   :caption: HPC2N Sessions:

   kebnekaise.md
   isolatedHPC2N.rst
   mlHPC2N.rst

.. toctree::
   :maxdepth: 2
   :caption: Exercises:

   exercises.rst

  

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
