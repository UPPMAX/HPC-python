.. Python-at-UPPMAX documentation master file, created by
   sphinx-quickstart on Fri Jan 21 18:24:04 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
Welcome to HPC python documentation
===================================

Here you will find the content of the workshop Using Python in an HPC environment.

.. admonition:: Content

This course aims to give a brief, but comprehensive introduction to using Python in an HPC environment. You will learn how to use modules to load Python, how to find site installed Python packages, as well as how to install packages yourself. In addition, you will learn how to use virtual environments, write a batch script for running Python, use Python in parallel, and how to use Python for ML and on GPUs. 

The course is a cooperation between UPPMAX (Rackham, Snowy, Bianca) and HPC2N (Kebnekaise) and will focus on the compute systems at both centres. For the site-specific part of the course you will be divided into groups depending on which center you will be running your code, as the approach is somewhat different. 

This course will consist of lectures interspersed with hands-on sessions where you get to try out what you have just learned.    

Preliminary schedule
====================

.. list-table:: Preliminary schedule
   :widths: 25 25 50
   :header-rows: 1

   * - Time
     - Topic
     - Activity
   * - 9:15
     - Introduction to Python on UPPMAX and HPC2N systems 
     - Lecture
   * - 
     - Loading modules and running Python codes 
     - Lecture+code along
   * - 
     - Dealing with packages  
     - Lecture+code along
   * - 
     - Creating isolated environments
     - Lecture+code along
   * - 
     - Separated session for Kebnekaise/Bianca
     - Lecture
   * - 12:00
     - LUNCH 
     -
   * - 13:15
     - Batch scripts for Python jobs  
     - Lecture+code along + exercise
   * - 
     - Parallelising a simple Python code  
     - Lecture+code along + exercise
   * - 
     - Using Python for Machine Learning jobs
     - Lecture+code along
   * - 
     - Using GPUs with Python
     - Lecture+code along + exercise
   * - 
     - Q&A on-demand and Summary 
     -
    

.. toctree::
   :maxdepth: 2
   :caption: Lessons:

   intro.rst
   load_run.rst
   packages.rst
   isolated.rst
   interactive.md
   batch.md
   parallel.rst
   ml.md
   gpu.md
   summary.rst
   
.. toctree::
   :maxdepth: 2
   :caption: Divided Sessons:

   bianca.rst
   kebnekaise.md

   
.. toctree::
   :maxdepth: 2
   :caption: Exercises:

   exercises.rst

  

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
