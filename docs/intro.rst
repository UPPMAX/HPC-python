Introduction
==============

.. admonition:: Welcome!

    - There is a new documentation for UPPMAX at: https://uppmax.uu.se/support/user-guides/python-user-guide/ 
    - We hope that it will be helpful for your future reference
    - We will approximately follow the outline of it today
    - We hope also to identify improvements of the page for today´s interactions with you!


What is python?
---------------

.. info::

    As you probably already know…
    
    - “Python combines remarkable power with very clear syntax.
    - It has modules, classes, exceptions, very high level dynamic data types, and dynamic typing. 
    - There are interfaces to many system calls and libraries, as well as to various windowing systems. …“

Documentation is found here https://www.python.org/doc/ .

Python forum is found here https://python-forum.io/ .

.. seealso::

    - For other python topics, see python documentation https://www.python.org/doc/.
    - Python forum is found here https://python-forum.io/.
    - A nice introduction to packages can be found here: https://aaltoscicomp.github.io/python-for-scicomp/dependencies/
    - CodeRefinery develops and maintains training material on software best practices for researchers that already write code. Ther material addresses all academic disciplines and tries to be as programming language-independent as possible. https://coderefinery.org/lessons/
    
    - And, if you feel a little unfamiliar to the LINUX world, have a look at the Introdu ction to UPPMAX course material here: https://www.uppmax.uu.se/support/courses-and-workshops/introductory-course-winter-2022/


Python at UPPMAX
----------------

The python application at UPPMAX comes with several preinstalled packages.
A very small selection of these are:
Numpy, Pandas, Scipy, Matplotlib, Jupyter notebook, pip, cython, ipython

.. questions:: 

    - What to do if you need other packages?
    - How does it work on Bianca without internet?
    - What if I have projects with different requirements in terms of python and packages versions?
    
.. objectives:: 

    We will:
    
    - guide through the python ecosystem on UPPMAX
    - look at the package handlers **pip** and **Conda**
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

