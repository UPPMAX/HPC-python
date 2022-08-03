Introduction
==============

.. admonition:: **Welcome!**


.. admonition:: **Learning outcomes**
   
    - **load** and **run** python
    - **find** installed packages
    - install package with **pip**
    - install package with **conda**
    - Understand and create **virtual environments**
    - best practice
    
.. admonition:: Collabration document HackMD

    - Use the HackMD page for the workshop with your questions.
    - Depending on how many helpers there are we'll see how fast there are answers. 
        - Some answers may come after the workshop.
 
    - Type in the left frame 
        - "-" means new bullet and <tab> indents the level.
        - don't focus too much on the formatting if you are new to "Markdown" language!
    - **Have a try with the Icebreaker question**

.. admonition:: **Your expectations?**
   
    - find best practices for using Python at UPPMAX and HPC2N
    - packages
    - use the HPC performance with Python

    
    **Not covered**
    
    - improve python *coding* skills 
    - Tetralith


.. warning::

    - It is good to have a familiarity with the LINUX command line. 
    - Short introductions may be found here: https://uppsala.instructure.com/courses/67267/pages/using-the-command-line-bash?module_item_id=455632
    - Linux "cheat sheet"; https://www.hpc2n.umu.se/documentation/guides/linux-cheat-sheet
    - UPPMAX software library: https://uppsala.instructure.com/courses/67267/pages/uppmax-basics-software?module_item_id=455641
    - Whole intro course material (UPPMAX) can be reached here: https://www.uppmax.uu.se/support/courses-and-workshops/introductory-course-winter-2022/
    - HPC2N's intro course material (including link to recordings) may be found here: https://github.com/hpc2n/intro-course

.. admonition:: Prepare your environment now!
  
   - Please log in to Rackham, Kebnekaise or other cluster that you are using 
     
     - Rackham: ``ssh <user>@rackham.uppmax.uu.se`` 
     - Kebnekaise: ``<user>@kebnekaise.hpc2n.umu.se``     
     - Kebnekaise through ThinLinc, use: ``<user>@kebnekaise-tl.hpc2n.umu.se``
     
   - Create a working directory where you can code along. We recommend creating it under the course project storage directory
   
     - Example. If your username is bbrydsoe and you are at HPC2N, then we recommend you create this folder: 
     
         /proj/nobackup/snic2022-22-641/bbrydsoe/pythonHPC2N
         
     - Example. If your username is mrspock and you are at UPPMAX, this we recommend you create this folder: 
     
         /proj/nobackup/snic2022-22-641/mrspock/pythonUPPMAX
    
What is python?
---------------

As you probably already know…
    
    - “Python combines remarkable power with very clear syntax.
    - It has modules, classes, exceptions, very high level dynamic data types, and dynamic typing. 
    - There are interfaces to many system calls and libraries, as well as to various windowing systems. …“

- Documentation is found here https://www.python.org/doc/ .
- Python forum is found here https://python-forum.io/ .
- A nice introduction to packages can be found here: https://aaltoscicomp.github.io/python-for-scicomp/dependencies/
- CodeRefinery develops and maintains training material on software best practices for researchers that already write code. Ther material addresses all academic disciplines and tries to be as programming language-independent as possible. https://coderefinery.org/lessons/
    
    - And, if you feel a little unfamiliar to the LINUX world, have a look at the Introduction to UPPMAX course material here: https://www.uppmax.uu.se/support/courses-and-workshops/introductory-course-winter-2022/
    
More python?
-----------

- CodeRefinery develops and maintains training material on software best practices for researchers that already write code. Their material addresses all academic disciplines and tries to be as programming language-independent as possible. https://coderefinery.org/lessons/
- Introduction to Python at https://www.uppmax.uu.se/support/courses-and-workshops/introductory-course-winter-2022/
- Introduction to HPC (High performance computing) python at UPPMAX and HPC2N in September. More info to come!


Python at UPPMAX
----------------

The python application at UPPMAX comes with several preinstalled packages.
A very small selection of these are:
``Numpy``, ``Pandas``, ``Scipy``, ``Matplotlib``, ``Jupyter notebook``, ``pip``, ``cython``, ``ipython``

.. questions:: 

    - What to do if you need other packages?
    - How does it work on Bianca without internet?
    - What if I have projects with different requirements in terms of python and packages versions?
    
.. objectives:: 

    We will:
    
    - guide through the python ecosystem on UPPMAX
    - look at the package handlers **pip** and **conda**
    - explain how to create isolated environment 

.. warning:: 
   At UPPMAX we call the applications available via the *module system* **modules**. 
   https://www.uppmax.uu.se/resources/software/module-system/ 
   
   To distinguish these modules from the **python** *modules* that work as libraries we refer to the later ones as **packages**.

.. admonition:: Outline

   - Loading and running Python
   - Packages/modules
   - How to install packages
   - Isolated environments
   - Not this time: jupyter notebook & parallel jobs
        - Check the next SNIC training letter about new collaboration workshop in beginning of September.

Python at HPC2N
----------------

The python application at HPC2N comes with several preinstalled packages - check first before installing yourself!. HPC2N has both Python 2.7.x and Python 3.x installed. We will be using Python 3.x in this course.  For this course, the recommended version of Python to use on Kebnekaise is 3.9.5

NOTE:  HPC2N do NOT recommend (and do not support) using Anaconda/Conda on our systems. You can read more about this here: https://www.hpc2n.umu.se/documentation/guides/anaconda

A selection of the Python packages and libraries installed on HPC2N are:
   - ASE
   - Keras
   - PyTorch
   - SciPy-bundle (Bottleneck, deap, mpi4py, mpmath, numexpr, numpy, pandas, scipy - some of the versions have more)
   - TensorFlow
   - Theano
   - matplotlib
   - scikit-learn
   - scikit-image
   - pip
   - iPython
   - Cython
   - Flask

These are all installed as **modules** and need to be loaded before use. 

.. questions:: 

   - How do I find you which packages and versions are available?
   - What to do if you need other packages?
   - What if I need a graphical interface?
   - What if I have projects with different requirements in terms of python and packages versions?
    
.. objectives:: 

    We will:
    
    - teach you how to navigate the module system at HPC2N
    - show you how to find out which versions of Python and packages are installed
    - look at the package handler **pip** 
    - explain how to create and use virtual environments
    - show you how to run batch jobs 

.. warning:: 
   At HPC2N we call the applications available via the *module system* **modules**. 
   https://www.hpc2n.umu.se/documentation/environment/lmod
   
   To distinguish these modules from the **python** *modules* that work as libraries we refer to the later ones as **packages**.

.. admonition:: Outline

   - Loading and running Python
   - Packages/modules
   - How to install packages
   - Isolated/virtual environments
   - Parallel Python? (will we have this ?????????) 
