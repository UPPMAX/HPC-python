.. _day2-intro:
Introduction
============

`Welcome page and syllabus <https://uppmax.github.io/HPC-python/index.html>`_
   - Also link at House symbol |:house:| at top of page 

.. admonition:: **Course learning outcomes**
   
    <!-- TODO: B: Update -->

   - Load Python modules and site-installed Python packages
   - Create a virtual environment
   - Install Python packages with pip (Kebnekaise, Rackham, Snowy, COSMOS)
   - Write a batch script for running Python
   - Use the compute nodes interactively
   - Use Python in parallel
   - Use Python for ML
   - Use GPUs with Python

.. admonition:: **Learning outcomes**
   
   - Learners understand how this day is organized
   - Learners can find their NAISS project
   - Learners can find how to download and extract the exercises

.. admonition:: **For teachers: Lesson plan **
    :class: dropdown

    Prior questions:
    - What is Pyton?


<!-- TODO: B: should everyone do an intro on each day? Discuss in Matrix/meeting -->
   
.. admonition:: What is python again?
    :class: dropdown

    As you probably already know…
        
        - “Python combines remarkable power with very clear syntax.
        - It has modules, classes, exceptions, very high level dynamic data types, and dynamic typing. 
        - There are interfaces to many system calls and libraries, as well as to various windowing systems. …“

    In particular, what sets Python apart from other languages is its fantastic
    open-source ecosystem for scientific computing and machine learning with
    libraries like NumPy, SciPy, scikit-learn and Pytorch.

    - `Official Python documentation <https://www.python.org/doc/>`_ 
    - `Python forum <https://python-forum.io/>`_
    - `A nice introduction to packages <https://aaltoscicomp.github.io/python-for-scicomp/dependencies/>`_
    - The youtube video `Thinking about Concurrency <https://www.youtube.com/watch?v=Bv25Dwe84g0>`_ is a good introduction to writing concurrent programs in Python
    - The book `High Performance Python <https://www.oreilly.com/library/view/high-performance-python/9781492055013/>`_ is a good resource for ways of speeding up Python code.
        
.. important::

   Project ID and storage directory 

   - UPPMAX: 
       - Project ID: uppmax2025-2-296
       - Storage directory: /proj/hpc-python-uppmax  
   - HPC2N: 
       - Project ID: hpc2n2025-076
       - Storage directory: /proj/nobackup/hpc-python-spring
   - LUNARC: 
       - Project ID: lu2025-7-34
       - Storage directory: /lunarc/nobackup/projects/lu2024-17-44  
   - NSC: 
       - Project ID: naiss2025-22-403
       - Storage directory: /proj/hpc-python-spring-naiss  


<!-- 

TODO: R will make this into an exercise (i.e. an exercise and not
a prerequisite, unless we agree in a meeting that
this _is_ a prerequisite), tie in with downloading
and extracting the tarball

R: I predict that will take 30 minutes
B: I predict that will take 5-20 minutes
me and B think we'll get it to work

TODO: B: In meeting, discuss if tarball is a prerequisite,
do this in Matrix. If not sent out to learners this is a prereq,
we'll do this as an exercise.

-->

Login to the center you have an account at, go to the storage directory,
and create a directory below it for you to work in.
You can call this directory what you want, but your username is a good option. 

<!-- TODO: R: merge with exercise -->

.. important::

   Course material 

   - You can get the course material, including exercises, from the course repository on GitHub. You can either (on of these): 
       - Clone it: ``git clone https://github.com/UPPMAX/HPC-python.git``
       - Download the zip file and unzip it: 
           - ``wget https://github.com/UPPMAX/HPC-python/archive/refs/heads/main.zip``  
           - ``unzip main.zip``

   - You should do either of the above from your space under the course directory on the HPC center of your choice. 
