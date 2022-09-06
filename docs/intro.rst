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


Some practicals
----------------
    
.. admonition:: Collabration document HackMD

    - Use the HackMD page for the workshop with your questions.
      <link>
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
    - Other clusters


.. warning::

    - It is good to have a familiarity with the LINUX command line. 
    - Short introductions : https://uppsala.instructure.com/courses/67267/pages/using-the-command-line-bash?module_item_id=455632
    - Linux "cheat sheet"; https://www.hpc2n.umu.se/documentation/guides/linux-cheat-sheet
    - UPPMAX software library: https://uppsala.instructure.com/courses/67267/pages/uppmax-basics-software?module_item_id=455641
    - Whole intro course material (UPPMAX): https://www.uppmax.uu.se/support/courses-and-workshops/introductory-course-winter-2022/
    - HPC2N's intro course material (including link to recordings): https://github.com/hpc2n/intro-course

.. admonition:: Prepare your environment now!
  
   - Please log in to Rackham, Kebnekaise or other cluster that you are using 

    
.. tabs::

   .. tab:: UPPMAX

      - Rackham: ``ssh <user>@rackham.uppmax.uu.se`` 
      
      - Create a working directory where you can code along. We recommend creating it under the course project storage directory
   
         
      - Example. If your username is "mrspock" and you are at UPPMAX, this we recommend you create this folder: 
     
         /proj/snic2022-22-641/nobackup/mrspock/pythonUPPMAX

   .. tab:: HPC2N

      - Kebnekaise: ``<user>@kebnekaise.hpc2n.umu.se``     
      - Kebnekaise through ThinLinc, use: ``<user>@kebnekaise-tl.hpc2n.umu.se``
   
      - Create a working directory where you can code along. We recommend creating it under the course project storage directory
   
       - Example. If your username is bbrydsoe and you are at HPC2N, then we recommend you create this folder: 
     
         /proj/nobackup/snic2022-22-641/bbrydsoe/pythonHPC2N
         


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
------------

- CodeRefinery develops and maintains training material on software best practices for researchers that already write code. Their material addresses all academic disciplines and tries to be as programming language-independent as possible. https://coderefinery.org/lessons/
- General introduction to Python at https://www.uppmax.uu.se/support/courses-and-workshops/introductory-course-summer-2022/

- **This course:** Introduction to HPC (High performance computing) python at UPPMAX and HPC2N in September. 

The two HPC centers UPPMAX and HPC2N
------------------------------------

.. admonition:: Two HPC centers

   - We run this course as a collaboration between the HPC centers HPC2N in Umeå and UPPMAX in Uppsala.
      - There are many similarities:
         - Login vs. calculation nodes
         - Environmental module system  with softaware hided until loaded with ``module load``
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
  
   - General sessions with small differences shown in UPPMAX vs. HPC2N  in tabs
   - Seperated sessions for UPPMAX and HPC users, respectively.

.. warning:: 

   - At both HPC2N UPPMAX we call the applications available via the *module system* **modules**. 
   - https://www.uppmax.uu.se/resources/software/module-system/ 
   - https://www.hpc2n.umu.se/documentation/environment/lmod
   
   To distinguish these modules from the **python** *modules* that work as libraries we refer to the later ones as **packages**.
   
Briefly about the cluster hardware and system at UPPMAX and HPC2N
-----------------------------------------------------------------

Common features
###############

- Intel CPUs
- Linux kernel
- Bash shell


.. list-table:: Hardware
   :widths: 25 25 25 25 25
   :header-rows: 1

   * - Technology
     - Kebnekaise
     - Rackham
     - Snowy
     - Bianca
   * - Cores per calculation node
     - 28
     - 20
     - 16
     - 16
   * - Memory per calculation node
     -  
     - 128-1024 GB
     - 128-4096 GB
     - 128-512 GB
   * - GPU
     - 
     - None
     - Nvidia T4 
     - NVIDIA A100


.. objectives:: 

    We will:
    
    - teach you how to navigate the module system at HPC2N and UPPMAX
    - show you how to find out which versions of Python and packages are installed
    - look at the package handler **pip** (and **Conda** for UPPMAX)
    - explain how to create and use virtual environments
    - show you how to run batch jobs 
    - show some examples with parallel computing and using GPUs
    - guide you in how to start Python tools for Machine Learning
 

