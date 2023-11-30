Introduction
==============

`Welcome page and syllabus <https://uppmax.github.io/HPC-python/index.html>`_
   - Also link at House symbol |:house:| at top of page 

.. admonition:: **Learning outcomes**
   
   - Load Python modules and site-installed Python packages
   - Create a virtual environment
   - Install Python packages with pip (Kebnekaise, Rackham, Snowy)
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

- Official Python documentation is found here https://www.python.org/doc/ 
- Python forum is found here https://python-forum.io/ 
- A nice introduction to packages can be found here: https://aaltoscicomp.github.io/python-for-scicomp/dependencies/
- The youtube video `Thinking about Concurrency <https://www.youtube.com/watch?v=Bv25Dwe84g0>`_ is a good introduction to writing concurrent programs in Python and the book `High Performance Python <https://www.oreilly.com/library/view/high-performance-python/9781492055013/>`_ is a good resource for ways of speeding up Python code.
    
    - And, if you feel a little unfamiliar to the LINUX world, have a look at the Introduction to UPPMAX course material here: https://www.uppmax.uu.se/support/courses-and-workshops/introductory-course-winter-2022/
    
The two HPC centers UPPMAX and HPC2N
------------------------------------

   
.. warning:: 

   - At both HPC2N UPPMAX we call the applications available via the *module system* **modules**. 
   - https://www.uppmax.uu.se/resources/software/module-system/ 
   - https://www.hpc2n.umu.se/documentation/environment/lmod
   
   To distinguish these modules from the **python** *modules* that work as libraries we refer to the later ones as **packages**.
   
Material for improving your programming skills
::::::::::::::::::::::::::::::::::::::::::::::

.. admonition:: First level

   `The Carpentries <https://carpentries.org/>`_  teaches basic lab skills for research computing.

   - `Programming with Python <https://swcarpentry.github.io/python-novice-inflammation/>`_ 

   - `Plotting and Programming in Python <http://swcarpentry.github.io/python-novice-gapminder/>`_ 

   General introduction to Python by UPPMAX at https://www.uppmax.uu.se/support/courses-and-workshops/introductory-course-summer-2023/


.. admonition:: Second level

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
    
Other NAISS centres
-------------------

.. seealso::

   - LUNARC
      - `Python <https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/>`_
      - `Jupyter <https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/?query=jupyter#jupyter-lab>`_
   - C3SE
      - `Python <https://www.c3se.chalmers.se/documentation/applications/python/>`_
      - `Jupyter <https://www.c3se.chalmers.se/documentation/applications/jupyter/>`_
      - `Python <https://www.nsc.liu.se/software/python/>`_
   - PDC
      - `Python <https://www.pdc.kth.se/software/software/python/index_general.html>`_


.. objectives:: 

    We will:
    
    - teach you how to navigate the module system at HPC2N and UPPMAX
    - show you how to find out which versions of Python and packages are installed
    - look at the package handler **pip** (and **Conda** for UPPMAX)
    - explain how to create and use virtual environments
    - show you how to run batch jobs 
    - show some examples with parallel computing and using GPUs
    - guide you in how to start Python tools for Machine Learning
 

