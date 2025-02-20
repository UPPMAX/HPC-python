Introduction
============

`Welcome page and syllabus <https://uppmax.github.io/HPC-python/index.html>`_
   - Also link at House symbol |:house:| at top of page 

.. admonition:: **Learning outcomes**
   
   - Load Python modules and site-installed Python packages
   - Create a virtual environment
   - Install Python packages with pip (Kebnekaise, Rackham, Snowy, Cosmos)
   - Write a batch script for running Python
   - Use the compute nodes interactively
   - Use Python in parallel
   - Use Python for ML
   - Use GPUs with Python
   

What is python?
---------------

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
    
Material for improving your programming skills
::::::::::::::::::::::::::::::::::::::::::::::

.. admonition:: First level

   `The Carpentries <https://carpentries.org/>`_  teaches basic lab skills for research computing.

   - `Programming with Python <https://swcarpentry.github.io/python-novice-inflammation/>`_ 

   - `Plotting and Programming in Python <http://swcarpentry.github.io/python-novice-gapminder/>`_ 

   General introduction to Python by UPPMAX at https://www.uu.se/en/centre/uppmax/study/courses-and-workshops/introduction-to-uppmax


.. admonition:: Second level

   Other course/workshops given by NAISS HPC centres:

   - `Pandas by LUNARC <https://github.com/rlpitts/Intro-to-Pandas>`_
   - `Matplotlib for publication <https://github.com/rlpitts/Matplotlib4Publication>`_


   CodeRefinery develops and maintains training material on software best practices for researchers that already write code. Their material addresses all academic disciplines and tries to be as programming language-independent as possible. 

   - `Lessons <https://coderefinery.org/lessons/>`_ 
   - `Data visualization using Python <https://coderefinery.github.io/data-visualization-python/>`_
   - `Jupyter <https://coderefinery.github.io/jupyter/>`_

   Aalto Scientific Computing

   - `Data analysis workflows with R and Python <https://aaltoscicomp.github.io/data-analysis-workflows-course/>`_

   - `Python for Scientific Computing <https://aaltoscicomp.github.io/python-for-scicomp/>`_

      - `Introduction to packages <https://aaltoscicomp.github.io/python-for-scicomp/dependencies/>`_


.. admonition:: Third level

   `ENCCS (EuroCC National Competence Centre Sweden) <https://enccs.se/>`_ is a national centre that supports industry, public administration and academia accessing and using European supercomputers. They give higher-level training of programming and specific software.

   - `High Performance Data Analytics in Python <https://enccs.github.io/hpda-python/>`_

   - The youtube video `Thinking about Concurrency <https://www.youtube.com/watch?v=Bv25Dwe84g0>`_ is a good introduction to writing concurrent programs in Python 

   - The book `High Performance Python <https://www.oreilly.com/library/view/high-performance-python/9781492055013/>`_ is a good resource for ways of speeding up Python code.
    
Documentations at other NAISS centres
-------------------------------------

.. seealso::

   - LUNARC
      - `Python <https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/>`_
      - `Jupyter <https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/?query=jupyter#jupyter-lab>`_
   - C3SE
      - `Python <https://www.c3se.chalmers.se/documentation/applications/python/>`_
      - `Jupyter <https://www.c3se.chalmers.se/documentation/applications/jupyter/>`_
      - `Python <https://www.nsc.liu.se/software/python/>`_
   - PDC
      - `Python <https://support.pdc.kth.se/doc/software-docs/python/>`_

.. important::

   Project ID and storage directory 

   - UPPMAX: 
       - Project ID: naiss2024-22-1442
       - Storage directory: /proj/hpc-python-fall  
   - HPC2N: 
       - Project ID: hpc2n2024-142
       - Storage directory: /proj/nobackup/hpc-python-fall-hpc2n
   - LUNARC: 
       - Project ID: lu2024-2-88
       - Storage directory: /lunarc/nobackup/projects/lu2024-17-44  
   - NSC: 
       - Project ID: naiss2024-22-1493
       - Storage directory: /proj/hpc-python-fall-nsc  

   Login to the center you have an account at, go to the storage directory, and create a directory below it for you to work in. You can call this directory what you want, but your username is a good option. 

.. important::

   Course material 

   - You can get the course material, including exercises, from the course repository on GitHub. You can either (on of these): 
       - Clone it: ``git clone https://github.com/UPPMAX/HPC-python.git``
       - Download the zip file and unzip it: 
           - ``wget https://github.com/UPPMAX/HPC-python/archive/refs/heads/main.zip``  
           - ``unzip main.zip``

   - You should do either of the above from your space under the course directory on the HPC center of your choice. 

.. objectives:: 

    We will:
    
    - teach you how to navigate the module system at HPC2N, UPPMAX, LUNARC, and NSC
    - show you how to find out which versions of Python and packages are installed
    - look at the package handler **pip**
    - explain how to create and use virtual environments
    - show you how to run batch jobs 
    - show some examples with parallel computing and using GPUs
    - guide you in how to start Python tools for Machine Learning
 
