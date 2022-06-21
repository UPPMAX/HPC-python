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
     -
     -
   * - 
     - Introduction to Python on UPPMAX and HPC2N systems 
     -
   * - 
     - Loading modules and running Python codes 
     -
     
     

| Time | Topic | Activity | Description | Goal |
| -----| ----- | -------- |------------ | ---- |
**Morning 2 hours**| 45 + 40m
10	|Intro	|Lecture |	Outline	| Get overview and LOs
10	|Load and run|	Lecture+code along	| Show modules |	Learn how to load a python version
25	|packages	|Lecture + code along + (exercise)	|Check current, Pip, Conda | 	List packages, do pip installation,do conda installation 
10	|Isolated environ	|Lecture + code along |	Venv |	Understand virtual environ
25	| Kebnekaise/Bianca separated session| Lecture | Cluster specific practice and installed packages, User interaction | Understand cluster limitations		
5	|Summary|	Lecture|	Describe when to do what|	Keypoints
**Afternoon 2 hours** | 45+45m
30|   batch with python and conda |Lecture+code along + exercise | python with batch | write batch script with right envs
15|   interactive jupyter| Lecture + code along +exercise | run jupyter on calculation nodes| run jupyter on calculation nodes
30|   workflows?
15|  Summary

.. toctree::
   :maxdepth: 2
   :caption: Guide:

   intro.rst
   load_run.rst
   packages.rst
   isolated.rst
   bianca.rst
   kebnekaise.md
   interactive.md
   jupyter.md
   batch.md
   parallel.md
   summary.rst

# .. toctree::
#    :maxdepth: 2
#   :caption: Workshop 2nd hour:
#   
#   uppmax.rst

   

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
