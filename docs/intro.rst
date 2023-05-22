Introduction
==============

`Welcome page and syllabus <https://uppmax.github.io/HPC-python/index.html>`_
   - Also link at House symbol |:house:| at top of page 

.. admonition:: **Learning outcomes**
   
   - Load Python modules and site-installed Python packages
   - Create a virtual environment
   - Install Python packages with pip (Kebnekaise, Rackham, Snowy)
   - Install Python packages with conda (Bianca)
   - Write a batchscript for running Python
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
- CodeRefinery develops and maintains training material on software best practices for researchers that already write code. Ther material addresses all academic disciplines and tries to be as programming language-independent as possible. https://coderefinery.org/lessons/
- The youtube video `Thinking about Concurrency <https://www.youtube.com/watch?v=Bv25Dwe84g0>`_ is a good introduction to writing concurrent programs in Python and the book `High Performance Python <https://www.oreilly.com/library/view/high-performance-python/9781492055013/>`_ is a good resource for ways of speeding up Python code.
    
    - And, if you feel a little unfamiliar to the LINUX world, have a look at the Introduction to UPPMAX course material here: https://www.uppmax.uu.se/support/courses-and-workshops/introductory-course-winter-2022/
    
More python?
------------

- CodeRefinery develops and maintains training material on software best practices for researchers that already write code. Their material addresses all academic disciplines and tries to be as programming language-independent as possible. https://coderefinery.org/lessons/
- General introduction to Python at https://www.uppmax.uu.se/support/courses-and-workshops/introductory-course-summer-2022/

- **This course:** Introduction to HPC (High performance computing) python at UPPMAX and HPC2N in September. 

The two HPC centers UPPMAX and HPC2N
------------------------------------

.. admonition:: Two HPC centers

   - We run this course as a collaboration between the HPC centers HPC2N in Umeå and UPPMAX in Uppsala.
      - There are many similarities:
         - Login vs. calculation/compute nodes
         - Environmental module system with software hidden until loaded with ``module load``
         - Slurm batch job and scheduling system
         - ``pip install`` procedure
      - ... and small differences:
         - commands to load Python and python packages
         - isolated environments ``virtualenv`` vs ``venv``
         - slightly different flags to Slurm
      - ... and some bigger differences:
         - UPPMAX has three different clusters 
            - Rackham for general purpose computing on CPUs only
            - Snowy available for local projects and suits long jobs (< 1 month) and has GPUs
            - Bianca for sensitive data and has GPUs
         - HPC2N has Kebnekaise with GPUs 
         - Conda is recommended only for UPPMAX users
    
.. admonition:: How is the workshop run?
  
   - General sessions with small differences shown for UPPMAX and HPC2N in tabs
   - Main focus on the NAISS resources at UPPMAX, but Kebnekaise specifics will be covered

.. warning:: 

   - At both HPC2N UPPMAX we call the applications available via the *module system* **modules**. 
   - https://www.uppmax.uu.se/resources/software/module-system/ 
   - https://www.hpc2n.umu.se/documentation/environment/lmod
   
   To distinguish these modules from the **python** *modules* that work as libraries we refer to the later ones as **packages**.
   
Briefly about the cluster hardware and system at UPPMAX and HPC2N
-----------------------------------------------------------------

**What is a cluster?**

- Login nodes and calculations nodes

- A network of computers, each computer working as a **node**.
     
- Each node contains several processor cores and RAM and a local disk called scratch.

.. figure:: img/node.png
   :align: center

- The user logs in to **login nodes**  via Internet through ssh or Thinlinc.

  - Here the file management and lighter data analysis can be performed.

.. figure:: img/nodes.png
   :align: center

- The **calculation nodes** have to be used for intense computing. 

- Beginner's guide to clusters: https://www.hpc2n.umu.se/documentation/guides/beginner-guide

Common features
###############

- Intel CPUs
- Linux kernel
- Bash shell

.. role:: raw-html(raw)
    :format: html

.. list-table:: Hardware
   :widths: 25 25 25 25 25
   :header-rows: 1

   * - Technology
     - Kebnekaise
     - Rackham
     - Snowy
     - Bianca
   * - Cores per calculation node
     - 28 (72 for largemem part)
     - 20
     - 16
     - 16
   * - Memory per calculation node
     - 128-3072 GB 
     - 128-1024 GB
     - 128-4096 GB
     - 128-512 GB
   * - GPU
     - NVidia K80 and V100 :raw-html:`<br />` (+ 3 NVidia A100 and 2 AMD MI100)
     - None
     - Nvidia T4 
     - 2 NVIDIA A100


.. objectives:: 

    We will:
    
    - teach you how to navigate the module system at HPC2N and UPPMAX
    - show you how to find out which versions of Python and packages are installed
    - look at the package handler **pip** (and **Conda** for UPPMAX)
    - explain how to create and use virtual environments
    - show you how to run batch jobs 
    - show some examples with parallel computing and using GPUs
    - guide you in how to start Python tools for Machine Learning
 

