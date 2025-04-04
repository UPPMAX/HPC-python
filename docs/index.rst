.. Python-at-UPPMAX documentation master file, created by
   sphinx-quickstart on Fri Jan 21 18:24:04 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
Welcome to "Using Python in an HPC environment" course material
===============================================================

.. admonition:: Content

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
   - **We aim to give this course in spring and fall every year.**

Schedule Spring 2025
--------------------

.. list-table:: 
   :widths: 25 25
   :header-rows: 1

   * - Day
     - Language
   * - Thursday 24 April
     - Intro to Python 
   * - Friday 25 April
     - Packages and basic analysis
   * - Monday 28 April
     - Advanced analysis and batch jobs
   * - Tuesday 29 April
     - PArallelism, GPUs and Machine Learning

.. warning:: 

   **Not covered**
    
    - Improve Python *coding* skills 
    - Specifics of other clusters

.. admonition:: Course approach to deal with multiple HPC centers

   **Target group**

   - The course is for present or presumptive users at The NAISS resources at PDC and NSC as well as UPPMAX, HPC2N, LUNARC, or possibly other clusters in Sweden. 
   - The course is a NAISS cooperation with teachers from **UPPMAX** (Rackham, Snowy, Bianca), **HPC2N** (Kebnekaise), and **LUNARC** (Cosmos) and will focus on systems at the Swedish academic HPC centres with NAISS personnel.
   - Although there are differences we will only have **few seperate sessions**.
   - Most participants will use The NAISS resources **Tetralith** system at NSC or **Dardel** system at PDC for the course.

       - Alvis at **C3SE (Chalmers)** may be added in the future.

   - Users with the "local" affiliation below, can work at the following clusters:

      - Kebnekaise: UmU, IRF, MIUN, SLU, LTU. 
      - Cosmos: LU. 
      - Rackham/Snowy: UU. 

   - The general information given in the course will be true for all/most HPC centres in Sweden. 

      - The examples will often have specific information, like module names and versions, which may vary. What you learn here should help you to make any changes needed for the other centres. 
      - When present, links to the Python documentation at other NAISS centres are given in the corresponding session.

.. admonition:: Cluster documentation of Python
   
   - Documentation at the HPC centres
      - `UPPMAX <https://docs.uppmax.uu.se/software/python/>`_
      - `HPC2N <https://docs.hpc2n.umu.se/tutorials/userinstalls/#python__packages>`_
      - `LUNARC <https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/>`_
      - `NSC <https://www.nsc.liu.se/support/>`_
      - `PDC <https://support.pdc.kth.se/doc/applications/python/>`_ 

.. admonition:: Some practicals

   - `Code of Conduct <https://github.com/UPPMAX/R-matlab-julia-HPC/blob/main/CODE_OF_CONDUCT.md>`_
      - Be nice to each other!
 
   - Zoom
       - You should have gotten an email with the links
      
       - Zoom policy:

           - Zoom chat (maintained by co-teachers):

                - technical issues of zoom
                - technical issues of your settings
                - direct communication 
                - each teacher may have somewhat different approach

            - collaboration document (see below):

                - "explain again"         
                - elaborating the course content
                - solutions for your own work

       - Recording policy:
           - **All lectures and demos** will be available on HPC2Ns **Youtube** channel after the course.

           - Due to different teaching styles, the teacher of a session decides if a lesson will be:
               - recorded without learners
               - recorded live with learners

           - The teachers will be clear if it is recorded or not and it will be visible in the ZOOM.
           - For the live-recordings

               - The questions asked per microphone during these sessions will be recorded
               - If you don't want your voice to appear: 

                   - use the collaboration document (see below)

       - The Zoom main room is used for most lectures
       - Some sessions use breakout rooms for exercises, some of which use a silent room
            


.. admonition:: Q/A collabration document

   - Use the Q/A page for the workshop with your questions.
       - https://umeauniversity.sharepoint.com/:w:/s/HPC2N630/EeQr33ZhejNHsv0crUSoaewBpGlpIdQUhqRsPdktS3xF9w
   - Use this page for the workshop with your questions
   - It helps us identify content that is missing in the course material
   - We answer those questions as soon as possible

.. warning::

   - **Please be sure that you have gone through the** `pre-requirements <https://uppmax.github.io/R-matlab-julia-HPC/prereqs.html>`_
      - It mentions the familiarity with the LINUX command line.
      - The applications to connect to the clusters: terminals and ThinLinc (remote graphical desktop)
   - This course does not aim to improve your coding skills. Rather you will learn to understand the ecosystems and navigations for the the different languages on a HPC cluster.


Make working directories 
    
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
      - Create a working directory where you can code along. We recommend creating it under the course project storage directory: /proj/nobackup/hpc2-python-fall-hpc2n
   
      - Example. If your username is bbrydsoe and you are at HPC2N, then we recommend you create this folder: 
     
          - ``/proj/nobackup/hpc-python-fall-hpc2n/bbrydsoe``

   .. tab:: LUNARC 

      - Cosmos with SSH: ``cosmos.lunarc.lu.se``
      - Cosmos through ThinLinc: ``cosmos-dt.lunarc.lu.se``

          - as server: ``cosmos-dt.lunarc.lu.se``
          - as user: ``<username-at-lunarc>`` NOTE: leave out the ``@lunarc.lu.se`` 

      - Create a working directory where you can code along, under the storage directory. The storage directory is located under: ``/lunarc/nobackup/projects/lu2024-17-44``    

   .. tab:: NSC 

      - Tetralith with SSH: ``tetralith.nsc.liu.se``
      - Tetralith through ThinLinc: ``tetralith.nsc.liu.se``

          - as server: tetralith.nsc.liu.se``
          - as user: ``<username-at-nsc>`` NOTE: leave out the ``@nsc.liu.se``

          - Create a working directory where you can code along, under ``/proj/hpc-python-fall-nsc``. 

      - 2FA is needed. Info here about setup: <https://www.nsc.liu.se/support/2fa/>   

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

          /proj/nobackup/hpc-python-fall-hpc2n/MYDIR-NAME/HPC-python/Exercises/examples/programs/

      Batch submit files

      .. code-block:: sh

          /proj/nobackup/hpc-python-fall-hpc2n/MYDIR-NAME/HPC-python/Exercises/examples/hpc2n/

   .. tab:: LUNARC

      Python programs

      .. code-block:: sh

      TO_BE_DONE

      Batch submit files

      .. code-block::

      TO_BE_DONE 

   .. tab:: NSC

      Python programs

      .. code-block:: sh

         /proj/hpc-python-fall-nsc/MYDIR-NAME/HPC-python/Exercises/examples/programs/

      Batch submit files 

      .. code-block:: sh 

      /proj/hpc-python-fall-nsc/MYDIR-NAME/HPC-python/Exercises/examples/nsc    
    

   
Content of the course
---------------------

.. toctree::
   :maxdepth: 2
   :caption: Pre-requirements:

   prereqs.rst
   preparations.rst

.. toctree::
   :maxdepth: 2
   :caption: Common:

   schedule.md
   common/login.rst
   common/use_tarball.rst
   common/understanding_clusters.rst
   common/naiss_projects_overview.rst
    
.. toctree::
   :maxdepth: 2
   :caption: Lessons day 1 (Intro to Python):

   common/day1.rst

.. toctree::
   :maxdepth: 2
   :caption: Lessons day 2 (packages and analysis):

   day2/intro.rst
   day2/load_run_packages.rst
   day2/install_packages.rst
   day2/use_isolated_environments.rst
   day2/interactive.md
   day2/ondemand-desktop.rst
   day2/IDEs.rst
   day2/Matplotlib60min.rst
   summary1.rst
   day2/python_at_hpc_centers.rst

.. toctree::
   :maxdepth: 2
   :caption: Lessons day 3 (advanced analysis):

   day3/pandas.rst
   day3/Seaborn-Intro.rst
   day3/batch.md
   day3/big_data.md
   summary2.rst

.. toctree::
   :maxdepth: 2
   :caption: Lessons day 4 (parallel and ML):

   day4/parallel.rst
   day4/gpu.md
   day4/ml.md

.. toctree::
   :maxdepth: 2
   :caption: Extra:

   extra/other_courses.rst
   extra/packages_deeper.rst
   extra/isolated_deeper.rst
   extra/interactive_deeper.rst
   extra/jupyterHPC2N.rst
   day3/ML_deeper.rst
   
   uppmax.rst
   kebnekaise.md
   bianca.rst
   
.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
