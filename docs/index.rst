.. Python-at-UPPMAX documentation master file, created by
   sphinx-quickstart on Fri Jan 21 18:24:04 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
Welcome to "Using Python in an HPC environment" course material
===============================================================

.. admonition:: This material
   
   Here you will find the content of the workshop Using Python in an HPC environment.
   
   - Documentation at the HPC centres UPPMAX, HPC2N, and LUNARC
      - UPPMAX: http://docs.uppmax.uu.se/software/python/
      - HPC2N: https://docs.hpc2n.umu.se/tutorials/userinstalls/#python__packages
      - LUNARC: https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/


.. exercise:: Content

   - This course aims to give a brief, but comprehensive introduction to using Python in an HPC environment.
   - You will learn how to
      - use modules to load Python
      - find site installed Python packages
      - install packages yourself
      - use virtual environments, 
      - write a batch script for running Python
      - use Python in parallel
      - use Python for ML and on GPUs. 
   - This course will consist of lectures interspersed with hands-on sessions where you get to try out what you have just learned.    

    **Not covered**
    
    - Improve python *coding* skills 
    - Specifics of other clusters

   - **We aim to give this course in spring and fall every year.**

.. admonition:: **Your expectations?**
   
    - Find best practices for using Python at an HPC centre
    - Learn how to use and install packages
    - Use the HPC capabilities of Python

    
    **Not covered**
    
    - Improve python *coding* skills 
    - Specifics of other clusters

.. warning::

  **Target group**
 
  - The course is for present or presumptive users at UPPMAX, HPC2N, LUNARC, or possibly other clusters in Sweden. 
  - Therefore we apply python solutions on all three clusters, so a broad audience can benefit.
  - We also provide links to the Python/Jupyter documentation at other Swedish HPC centres with personell affiliated to NAISS.

  **Cluster-specific approaches**
  
   - The course is a cooperation between UPPMAX (Rackham, Snowy, Bianca), HPC2N (Kebnekaise), and LUNARC (Cosmos). The main focus will be on UPPMAX's systems, but Kebnekaise and Cosmos will be included as well. If you already have an account at Kebnekaise or Cosmos, you can use those systems for the hands-ons. 
   - In most cases there is little or no difference between UPPMAX's systems, HPC2N's systems, and LUNARC's systems (and the other HPC systems in Sweden), except naming of modules and such. We will mention (and cover briefly) instances when there are larger differences.  

   - See further below a short introduction to the centre-specific cluster architectures of UPPMAX, HPC2N, and LUNARC.

.. admonition:: How is the workshop run?
  
   - General sessions with small differences shown for UPPMAX, HPC2N, and LUNARC in tabs
   - Main focus on the NAISS resources at UPPMAX, but Kebnekaise/Cosmos specifics will be covered
   - Users who already have accounts/projects at HPC2N (Kebnekaise) or LUNARC (Cosmos) are welcome to thoses systems for the exercises. UPPMAX/Rackham will be used for everyone else. 


.. prereq::

   - Python at a basic level
   - user account on either Kebnekaise at HPC2N, Rackham at UPPMAX, or Cosmos at LUNARC
   - having a terminal with X-forwarding available and/or ThinLinc app or two-factor authentication for web-login.
   - familiarity with the LINUX command line
  
      - `Short introduction <http://docs.uppmax.uu.se/getting_started/linux/>`_
      - `Linux tutorial and "cheat sheet" <https://docs.hpc2n.umu.se/tutorials/linuxguide/>`_
      - `HPC2N's Linux intro course material (including link to recordings) <https://github.com/hpc2n/intro-linux>`_
      - `UPPMAX's intro course material <https://www.uu.se/en/centre/uppmax/study/courses-and-workshops/introduction-to-uppmax>`_
      - `UPPMAX software library <http://docs.uppmax.uu.se/cluster_guides/modules/>`_


Some practicals
----------------
        
.. admonition:: Zoom

    - You should have gotten an email with the links    
    - Main room for lectures (recorded)
    - Breakout rooms
       - exercises, including a silent room for those who just want to work on their own without interruptions. 
       - help
    - The **lectures and demos will be recorded**, but **NOT the exercises**. 
       - If you ask questions during the lectures, you may thus be recorded. 
       - If you do not wish to be recorded, then please keep your microphone muted and your camera off during lectures and write your questions in the Q/A document (see more information below about the collaboration documents which are also listed above).
    - Use your REAL NAME.
    - Please MUTE your microphone when you are not speaking
    - Use the “Raise hand” functionality under the “Participants” window during the lecture. 
    - Please do not clutter the Zoom chat. 
    - Behave politely!
    
.. admonition:: Q/A collabration document

    - Use the Q/A page for the workshop with your questions.
        - https://umeauniversity.sharepoint.com/:w:/s/HPC2N630/EZJ2RH0InOlIqCSF1DAhxXgBVirUT04xQsaHka17j2ZJIw?rtime=4q1KLG9z3Eg

    - Depending on how many helpers there are we'll see how fast there are answers. 
        - Some answers may come after the workshop.
 
    - Create a new line for new questions. Take care if others are editing at the same time. 

The three HPC centers UPPMAX, HPC2N, and LUNARC
------------------------------------------------

.. admonition:: Three HPC centers

   - There are many similarities:
   
     - Login vs. calculation/compute nodes
     - Environmental module system with software hidden until loaded with ``module load``
     - Slurm batch job and scheduling system
     - ``pip install`` procedure
     
   - ... and small differences:
   
     - commands to load Python, Python packages
     - slightly different flags to Slurm
     
   - ... and some bigger differences:
   
     - UPPMAX has three different clusters 

       - Rackham for general purpose computing on CPUs only
       - Snowy available for local projects and suits long jobs (< 1 month) and has GPUs
       - Bianca for sensitive data and has GPUs

     - HPC2N has Kebnekaise with GPUs  

     - LUNARC has two systems 
     
       - Cosmos (CPUs and GPUs)
       - Cosmos-SENS (sensitive data) 

     - Conda is recommended only for UPPMAX and LUNARC users
    
.. warning:: 

   - At HPC2N, UPPMAX, and LUNARC we call the applications available via the *module system* **modules**. 

      - <http://docs.uppmax.uu.se/cluster_guides/modules/> 
      - <https://docs.hpc2n.umu.se/documentation/modules/>
      - <https://lunarc-documentation.readthedocs.io/en/latest/manual/manual_modules/>  
   
   - To distinguish these modules from the **python** *modules* that work as libraries we refer to the later ones as **packages**.
   
Briefly about the cluster hardware and system at UPPMAX, HPC2N, and LUNARC
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

**What is a cluster?**

- Login nodes and calculations/computation nodes

- A network of computers, each computer working as a **node**.
     
- Each node contains several processor cores and RAM and a local disk called scratch.

.. figure:: img/node.png
   :align: center

- The user logs in to **login nodes**  via Internet through ssh or Thinlinc.

  - Here the file management and lighter data analysis can be performed.

.. figure:: img/cluster.png
   :align: center

- The **calculation/compute nodes** have to be used for intense computing. 

 
Overview of the UPPMAX systems
##############################

.. mermaid:: mermaid/uppmax2.mmd

Overview of HPC2N's system 
##########################

.. mermaid:: mermaid/kebnekaise.mmd

Overview of LUNARC's systems
############################

.. figure:: img/cosmos-resources.png
   :align: center

Preliminary schedule
====================

.. list-table:: Preliminary schedule Thursday 5 December
   :widths: 20 25 40 15
   :header-rows: 1

   * - Time
     - Topic
     - Content
     - Teacher(s)
   * - 9:00
     - Introduction to the course, log in, load/run Python, find packages
     - Getting started with practical things
     - All
   * - 9:55
     - **Coffee**
     - 
     - 
   * - 10:10
     - Install packages and isolated environments 
     - Install, create and handle
     - Björn
   * - 11.00
     - **Short leg stretch**
     - 
     - 
   * - 11:10
     - Reaching compute nodes with Slurm  
     - Batch jobs vs interactive work in IDEs
     - Birgitte
   * - 11:50
     - Catch-up time and Q/A (no recording)
     - 
     - 
   * - 12:00
     - **LUNCH**
     - 
     - 
   * - 13:00-14:30
     - Analysis with Python
     - Matplotlib, IDEs and plots from scripts
     - Rebecca
   * - 13.55
     - **Short leg stretch** 15m
     - 
     - 
   * - 14:30-15:30
     - Using GPUs for Python
     - Pedro?
     - 
   * - 14.50
     - **Coffee** 15 min
     - 
     - 
   * - 15:35
     - Use cases and Q/A
     - Bring your own problems
     - All
   * - 16.35
     - Summary + Evaluation
     - 
     - 

.. list-table:: Preliminary schedule Friday 6 December
   :widths: 20 25 40 15
   :header-rows: 1

   * - Time
     - Topic
     - Content
     - Teacher
   * - 9:00
     - Analysis with Python part I
     - Pandas
     - Rebecca
   * - 9:50
     - **Coffee**
     - 
     - 
   * - 10:05
     - Analysis with Python part II
     - Pandas & Seaborn
     - Rebecca
   * - 10.55
     - **Short leg stretch**
     - 
     - 
   * - 11:10
     - Parallelism part I: MPI, Processes, Dask
     - Processes, MPI
     - Pedro
   * - 12:00
     - **LUNCH**
     - 
     - 
   * - 13:00
     - Parallelism part II: MPI, Processes, Dask
     - Dask
     - Pedro
   * - 13:15
     - Big Data with Python
     - File formats and packages, Chunking
     - Björn
   * - 13:50
     - **Short leg stretch**
     - 
     - 
   * - 14:05
     - Machine and Deep Learning part I
     - Pytorch, Tensorflow, ScikitLearn
     - Jayant
   * - 14.55
     - **Coffee**
     - 
     - 
   * - 15:10
     - Machine and Deep Learning part II
     - Pytorch, Tensorflow, ScikitLearn
     - Jayant
   * - 15.50
     - **Short leg stretch**
     - 
     - 
   * - 16.00
     - Use cases and Q&A
     - Bring your own problems
     - All
   * - 16.45
     - Ending, with evaluation
     - 
     - 

.. admonition:: Prepare your environment now!
  
   - Please log in to Rackham, Kebnekaise, Cosmos or other cluster that you are using.
   - For graphics, ThinLinc may be the best option.
      - The `ThinLinc app <https://www.cendio.com/thinlinc/download/>`_.
   - Rackham has access for regular SSH, through a regular ThinLinc client and a through a web browser interface with ThinLinc: 
       - SSH: rackham.uppmax.uu.se  
       - ThinLinc client: rackham-gui.uppmax.uu.se 
       - Web browser interface: https://rackham-gui.uppmax.uu.se
       - `ThinLinc user guide at UPPMAX <http://docs.uppmax.uu.se/cluster_guides/thinlinc/>`_
           - 2FA may be needed, which can be handled by logging in with regular SSH, doing 2FA, logging out again, then there is a grace period of some minutes for you to login to ThinLinc. `More here: <https://www.uu.se/en/centre/uppmax/get-started/2-factor>`_
   - Kebnekaise has access for regular SSH, ThinLinc clients, and through a web browser interface with ThinLinc: 
       - SSH: kebnekaise.hpc2n.umu.se 
       - ThinLinc client: kebnekaise-tl.hpc2n.umu.se 
       - From webbrowser: https://kebnekaise-tl.hpc2n.umu.se:300/ 
   - Cosmos: 
       - SSH: cosmos.lunarc.lu.se 
       - ThinLinc: cosmos-dt.lunarc.lu.se 
           - 2FA required! For more info, `go here: <https://lunarc-documentation.readthedocs.io/en/latest/getting_started/login_howto/>`_ 

.. admonition:: Project

    - The course project on UPPMAX (Rackham) is: ``naiss2024-22-1442``
    - The course project on HPC2N (Kebnekaise) is: ``hpc2n2024-142``
    - The course project on LUNARC (Cosmos) is: `` ``


    
.. tabs::

   .. tab:: UPPMAX

      - Rackham: ``ssh <user>@rackham.uppmax.uu.se``       
      - Rackham through ThinLinc, 
       
         - use the App with
             - address: ``rackham-gui.uppmax.uu.se``  NB: leave out the ``https://www.``!
             - user: ``<username-at-uppmax>``  NB: leave out the ``https://www.``!
         - or go to <https://rackham-gui.uppmax.uu.se>

           - here, you'll need two factor authentication.
          
      - Create a working directory where you can code along. We recommend creating it under the course project storage directory
   
         - Example. If your username is "mrspock" and you are at UPPMAX, then we recommend you to create a user folder in the project folder of the course and step into that: 

         - ``cd /proj/hpc-python-fall``
         - ``mkdir mrspock``
         - ``cd mrspock``

   .. tab:: HPC2N

      - Kebnekaise: ``<user>@kebnekaise.hpc2n.umu.se``     
      - Kebnekaise through ThinLinc, use the client and put
        
         - as server: ``kebnekaise-tl.hpc2n.umu.se`` 
         - as user: ``<username-at-HPC2N>`` NOTE: Leave out the ``@hpc2n.umu.se``
      - Create a working directory where you can code along. We recommend creating it under the course project storage directory
   
      - Example. If your username is bbrydsoe and you are at HPC2N, then we recommend you create this folder: 
     
          - ``/proj/nobackup/hpc-python-fall-hpc2n/bbrydsoe``

   .. tab:: LUNARC 

      - Cosmos with SSH: ``cosmos.lunarc.lu.se``
      - Cosmos through ThinLinc: ``cosmos-dt.lunarc.lu.se``

          - as server: ``cosmos-dt.lunarc.lu.se``
          - as user: ``<username-at-lunarc>`` NOTE: leave out the ``@lunarc.lu.se`` 

      - Create a working directory where you can code along.     

.. admonition:: Exercises

   - Stay in/go to the folder you just created above!
   - You can download the exercises from the course GitHub repo, under the "Exercises" directory or clone the whole repo!
 
       - Clone it with: ``git clone https://github.com/UPPMAX/HPC-python.git``
       - Copy the tarball with ``wget https://github.com/UPPMAX/HPC-python/raw/refs/heads/main/exercises.tar.gz`` and then uncompress with ``tar -zxvf exercises.tar.gz``  

   - Get an overview here: ``https://github.com/UPPMAX/HPC-python/tree/main/Exercises``
   
**NOTE** If you downladed the tarball and uncompressed it, the exercises are under ``exercises/`` in the directory you picked. Under that you find Python scripts in ``programs`` and batch scripts in the directories named for the sites. 

**NOTE** If you are downloading / cloning from the course GitHub repo and into the above directory, your Python examples and batch submit file examples will be in a subdirectory of that.

Assuming you created a directory MYDIR-NAME under the project storage, you will find the examples as follows:

.. tabs::

   .. tab:: UPPMAX

        Python programs

        .. code-block:: sh

            /proj/hpc-python-fall/MYDIR-NAME/HPC-python/Exercises/examples/programs/

        Batch submit files 

        .. code-block:: sh

            /proj/hpc-python-fall/MYDIR-NAME/HPC-python/Exercises/examples/uppmax


   .. tab:: HPC2N
 
      Python programs

      .. code-block:: sh

          /proj/nobackup/python-hpc-fall-hpc2n/MYDIR-NAME/HPC-python/Exercises/examples/programs/

      Batch submit files

      .. code-block:: sh

          /proj/nobackup/python-hpc-fall-hpc2n/MYDIR-NAME/HPC-python/Exercises/examples/hpc2n/

   .. tab:: LUNARC

      Python programs

      .. code-block:: sh

      TO_BE_DONE

      Batch submit files

      .. code-block::

      TO_BE_DONE 
    


.. admonition:: Use Thinlinc or terminal?

   - It is up to you!
   - Graphics come easier with Thinlinc, so recomended in the early session when we will plot a figure.
   - For this course, when having many windows open, it may be better to run in terminal in most of the cases, for space issues.
   
   
Content of the course
---------------------

.. toctree::
   :maxdepth: 2
   :caption: Pre-requirements:

   prereqs.md
    
.. toctree::
   :maxdepth: 2
   :caption: Lessons day 1:

   day1/intro.rst
   day1/load_run_packages.rst
   day1/install_packages.rst
   day1/batch.md
   day1/interactive.md
   day1/gpu.md
   summary.rst

.. toctree::
   :maxdepth: 2
   :caption: Lessons day 2:

   day2/parallel.rst
   day2/ml.md
   day2/ML_deeper.rst

.. toctree::
   :maxdepth: 2
   :caption: Extra reading:

   extra/packages_deeper.rst
   extra/isolated_deeper.rst
   extra/interactive_deeper.rst
   #jupyter.md
   uppmax.rst
   kebnekaise.md
   bianca.rst
   
.. toctree::
   :maxdepth: 2
   :caption: Exercises:

   exercises.rst

  

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
