Load and run python and use packages
====================================

At UPPMAX, HPC2N, LUNARC, and NSC (and most other Swedish HPC centres) we call the applications available via the module system modules. 
    - http://docs.uppmax.uu.se/cluster_guides/modules/
    - https://docs.hpc2n.umu.se/documentation/modules/
    - https://lunarc-documentation.readthedocs.io/en/latest/manual/manual_modules/ 
    - https://www.nsc.liu.se/software/modules/

.. objectives:: 

   - Show how to load Python
   - Show how to run Python scripts and start the Python command line

.. admonition:: Short cheat sheet
    
    - See which modules exists: ``module spider`` or ``ml spider``
    - Find module versions for a particular software: ``module spider <software>``
    - Modules depending only on what is currently loaded: ``module avail`` or ``ml av``
    - See which modules are currently loaded: ``module list`` or ``ml``
    - Load a module: ``module load <module>/<version>`` or ``ml <module>/<version>``
    - Unload a module: ``module unload <module>/<version>`` or ``ml -<module>/<version>``
    - More information about a module: ``module show <module>/<version>`` or ``ml show <module>/<version>``
    - Unload all modules except the 'sticky' modules: ``module purge`` or ``ml purge``
    
.. warning::
   
   - Note that the module systems at UPPMAX, HPC2N, LUNARC, and NSC are slightly different. 
   - While all modules at 
       - UPPMAX not directly related to bio-informatics are shown by ``ml avail`` 
       - NSC are show by ``ml avail``
       - HPC2N and LUNARC are hidden until one has loaded a prerequisite like the compiler ``GCC``.


- For reproducibility reasons, you should always load a specific version of a module instead of just the default version
- Many modules have prerequisite modules which needs to be loaded first (at HPC2N/LUNARC/NSC this is also the case for the Python modules). When doing ``module spider <module>/<version>`` you will get a list of which other modules needs to be loaded first


Check for Python versions
-------------------------

.. tip::
    
   **Type along!**

.. tabs::

   .. tab:: UPPMAX

      Check all available Python versions with:

      .. code-block:: console

          $ module avail python

      NOTE that python is written in lower case!


   .. tab:: HPC2N
   
      Check all available version Python versions with:

      .. code-block:: console
 
         $ module spider Python
      
      To see how to load a specific version of Python, including the prerequisites, do 

      .. code-block:: console
   
         $ module spider Python/<version>

      Example for Python 3.11.3 

      .. code-block:: console

         $ module spider Python/3.11.3

   .. tab:: LUNARC 

      Check all available Python versions with: 

      .. code-block:: console 

         $ module spider Python 

      To see how to load a specific version of Python, including the prerequisites, do 

      .. code-block:: console 

         $ module spider Python/<version>

      Example for Python 3.11.5 

      .. code-block:: console

         $ module spider Python/3.11.5

   .. tab:: NSC

      Check all available Python versions with: 

      .. code-block:: console

         $ module spider Python

      To see how to load a specific version of Python, including the prerequisites, do 

      .. code-block:: console 

         $ module spider Python/<version>

      Example for Python 3.11.5

      .. code-block:: console

         $ module spider Python/3.11.5


.. admonition:: Output at UPPMAX as of May 14, 2024
   :class: dropdown
    
       .. code-block::  console
    
           ----------------------------------- /sw/mf/rackham/applications -----------------------------------
              python_GIS_packages/3.10.8      python_ML_packages/3.9.5-gpu         wrf-python/1.3.1
              python_ML_packages/3.9.5-cpu    python_ML_packages/3.11.8-cpu (D)
           
           ------------------------------------ /sw/mf/rackham/compilers -------------------------------------
              python/2.7.6     python/3.4.3    python/3.9.5         python3/3.6.8     python3/3.11.8
              python/2.7.9     python/3.5.0    python/3.10.8        python3/3.7.2     python3/3.12.1 (D)
              python/2.7.11    python/3.6.0    python/3.11.4        python3/3.8.7
              python/2.7.15    python/3.6.8    python/3.11.8        python3/3.9.5
              python/3.3       python/3.7.2    python/3.12.1 (D)    python3/3.10.8
              python/3.3.1     python/3.8.7    python3/3.6.0        python3/3.11.4

          Where:
           D:  Default Module

           Use module spider" to find all possible modules and extensions.
           Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".

.. admonition:: Output at HPC2N as of May 14, 2024
    :class: dropdown

        .. code-block:: console

           b-an01 [~]$ module spider Python
           ----------------------------------------------------------------------------
           Python:
           ----------------------------------------------------------------------------
           Description:
               Python is a programming language that lets you work more quickly and
               integrate your systems more effectively.
    
            Versions:
                Python/2.7.15   
                Python/2.7.16  
                Python/2.7.18-bare 
                Python/2.7.18  
                Python/3.7.2   
                Python/3.7.4   
                Python/3.8.2   
                Python/3.8.6   
                Python/3.9.5-bare  
                Python/3.9.5   
                Python/3.9.6-bare  
                Python/3.9.6   
                Python/3.10.4-bare
                Python/3.10.4
                Python/3.10.8-bare
                Python/3.10.8
                Python/3.11.3
                Python/3.11.5
            Other possible modules matches:
                Biopython  Boost.Python  GitPython  IPython  flatbuffers-python  ...
           ----------------------------------------------------------------------------
           To find other possible module matches execute:
               $ module -r spider '.*Python.*'
           ----------------------------------------------------------------------------
           For detailed information about a specific "Python" package (including how to load the modules) use the module's full name.
               Note that names that have a trailing (E) are extensions provided by other modules.
       
           For example:
            $ module spider Python/3.9.5
           ----------------------------------------------------------------------------

.. admonition:: Output at LUNARC as of Nov 5, 2024
    :class: dropdown

        .. code-block:: console

           $ module spider Python

           --------------------------------------------------------------------------------------------------------
             Python:
           --------------------------------------------------------------------------------------------------------
               Description:
                 Python is a programming language that lets you work more quickly and integrate your systems more effectively.

                Versions:
                   Python/2.7.18-bare
                   Python/2.7.18
                   Python/3.8.6
                   Python/3.9.5-bare
                   Python/3.9.5
                   Python/3.9.6-bare
                   Python/3.9.6
                   Python/3.10.4-bare 
                   Python/3.10.4
                   Python/3.10.8-bare
                   Python/3.10.8
                   Python/3.11.3
                   Python/3.11.5
                   Python/3.12.3
                Other possible modules matches:
                   Biopython  GitPython  IPython  Python-bundle  Python-bundle-PyPI  bx-python  flatbuffers-python  ...

           --------------------------------------------------------------------------------------------------------
              To find other possible module matches execute:

                 $ module -r spider '.*Python.*'

           --------------------------------------------------------------------------------------------------------
             For detailed information about a specific "Python" package (including how to load the modules) use the module's full name.
             Note that names that have a trailing (E) are extensions provided by other modules.
             For example:

                $ module spider Python/3.12.3
           --------------------------------------------------------------------------------------------------------

.. admonition:: Output at NSC (Tetralith) as of Nov 20, 2024
    :class: dropdown

        .. code-block:: console

           $ module spider Python
           ####################################################################################################################################
           # NOTE: At NSC the output of 'module spider' is generally not helpful as all relevant software modules are shown by 'module avail' #
           # Some HPC centers hide software until the necessary dependencies have been loaded. NSC does not do that.                          #
           ####################################################################################################################################

           ----------------------------------------------------------------------------
             Python:
           ----------------------------------------------------------------------------
                Versions:
                   Python/recommendation
                   Python/2.7.18-bare-hpc1-gcc-2022a-eb
                   Python/2.7.18-bare
                   Python/3.10.4-bare-hpc1-gcc-2022a-eb
                   Python/3.10.4-bare
                   Python/3.10.4-env-hpc1-gcc-2022a-eb
                   Python/3.10.4-env-hpc2-gcc-2022a-eb
                   Python/3.10.4
                   Python/3.10.8-bare
                   Python/3.10.8
                   Python/3.11.3
                   Python/3.11.5
                Other possible modules matches:
                   IPython  netcdf4-python

           ----------------------------------------------------------------------------
             To find other possible module matches execute:

                 $ module -r spider '.*Python.*'

           ----------------------------------------------------------------------------
             For detailed information about a specific "Python" package (including how to load the modules) use the module's full name.
             Note that names that have a trailing (E) are extensions provided by other modules.
              For example:

                $ module spider Python/3.11.5
           ----------------------------------------------------------------------------


.. note:: 

   Unless otherwise said, we recomment using Python 3.11.x in this course. 


Load a Python module
--------------------

For reproducibility, we recommend ALWAYS loading a specific module instad of using the default version! 

.. tip::
    
   **Type along!**


.. tabs::

   .. tab:: UPPMAX
   
      Go back and check which Python modules were available. To load version 3.11.8, do:

      .. code-block:: console

        $ module load python/3.11.8
        
      Note: Lowercase ``p``.
      For short, you can also use: 

      .. code-block:: console

         $ ml python/3.11.8

 
   .. tab:: HPC2N

      To load Python version 3.11.3, do: 	    
 
      .. code-block:: console

         $ module load GCC/12.3.0 Python/3.11.3

      Note: Uppercase ``P``.   
      For short, you can also use: 

      .. code-block:: console

         $ ml GCC/12.3.0 Python/3.11.3

   .. tab:: LUNARC

      To load Python version 3.11.5, do:

      .. code-block:: console

         $ module load GCC/13.2.0 Python/3.11.5

      Note: Uppercase ``P``.
      For short, you can also use:

      .. code-block:: console

         $ ml GCC/13.2.0 Python/3.11.5

   .. tab:: NSC (Tetralith)

      To load Python version 3.11.5, do:

      .. code-block:: console

         $ module load buildtool-easybuild/4.8.0-hpce082752a2  GCCcore/13.2.0 GCC/13.2.0 Python/3.11.5

      Note: Uppercase ``P``.
      For short, you can also use:

      .. code-block:: console

         $ ml buildtool-easybuild/4.8.0-hpce082752a2  GCCcore/13.2.0 GCC/13.2.0 Python/3.11.5


.. warning::

   + UPPMAX: Don’t use system-installed python (2.7.5)
   + UPPMAX: Don't use system installed python3 (3.6.8)
   + HPC2N: Don’t use system-installed python (2.7.18)
   + HPC2N: Don’t use system-installed python3  (3.8.10)
   + LUNARC: Don’t use system-installed python/python3 (3.9.18)  
   + NSC: Don't use system-installed python/python3 (3.9.18) 
   + ALWAYS use python module

.. admonition:: Why are there both Python/2.X.Y and Python/3.Z.W modules?

   - Some existing software might use `Python2` and some will use `Python3`. 
   - Some of the Python packages have both `Python2` and `Python3` versions. 
   - Check what your software as well as the installed modules need when you pick!   
    
.. admonition:: UPPMAX: Why are there both python/3.X.Y and python3/3.X.Y modules?

   - Sometimes existing software might use `python2` and there's nothing you can do about that.
   - In pipelines and other toolchains the different tools may together require both `python2` and `python3`.
   - Here's how you handle that situation:
    
    + You can run two python modules at the same time if ONE of the module is ``python/2.X.Y`` and the other module is ``python3/3.X.Y`` (not ``python/3.X.Y``).
    


.. admonition:: LUNARC: Are python and python3 equivalent, or does the former load Python/2.X.Y?

   The answer depends on which module is loaded. If Python/3.X.Y is loaded, then python is just an alias for python3 and it will start the same command line. However, if Python/2.7.X is loaded, then python will start the Python/2.7.X command line while python3 will start the system version (3.9.18). If you load Python/2.7.X and then try to load Python/3.X.Y as well, or vice-versa, the most recently loaded Python version will replace anything loaded prior, and all dependencies will be upgraded or downgraded to match. Only the system’s Python/3.X.Y version can be run at the same time as a version of Python/2.7.X.


Run
---

Run Python script
#################

.. hint::

   - There are many ways to edit your scripts.
   - If you are rather new.

      - Graphical: ``$ gedit <script> &`` 
   
         - (``&`` is for letting you use the terminal while editor window is open)

         - Requires ThinLinc or ``ssh -X``

      - Terminal: ``$ nano <script>``

   - Otherwise you would know what to do!
   - |:warning:| The teachers may use their common editor, like ``vi``/``vim``
      - If you get stuck in ``vim``, press: ``<esc>`` and then ``:q`` !
 

.. type-along::

   - Let's make a script with the name ``example.py``  

   .. code-block:: console

      $ nano example.py

   - Insert the following text

   .. code-block:: python

      # This program prints Hello, world!
      print('Hello, world!')

   - Save and exit. In nano: ``<ctrl>+O``, ``<ctrl>+X``

   You can run a python script in the shell like this:

   .. code-block:: bash

      $ python example.py
      # or 
      $ python3 example.py

.. warning::

   - *ONLY* run jobs that are short and/or do not use a lot of resources from the command line. 
   - Otherwise use the batch system (see the `batch session <https://uppmax.github.io/HPC-python/batch.html>`_)
    

Run an interactive Python shell
###############################

- You can start a simple python terminal by:

.. code-block:: console

   $ python 
    
**Example**

.. code-block:: python

   >>> a=3
   >>> b=7
   >>> c=a+b
   >>> c
   10

- Exit Python with <Ctrl-D>, ``quit()`` or ``exit()`` in the python prompt

.. code-block:: python

    >>> <Ctrl-D>
    >>> quit()
    >>> exit()





For more interactiveness you can run Ipython.

.. tip::
    
   **Type along!**



.. tabs::

   .. tab:: UPPMAX

      NOTE: remember to load a python module first. Then start IPython from the terminal
      
      .. code-block:: console

         $ ipython 
    
      or 

      .. code-block:: console

         $ ipython3 
         
      UPPMAX has also ``jupyter-notebook`` installed and available from the loaded Python module. Start with
       
      .. code-block:: console

         $ jupyter-notebook 
         
      You can decide on your own favorite browser and add ``--no-browser`` and open the given URL from the output given.
      From python/3.10.8 and forward, also jupyterlab is available.
         
    
   .. tab:: HPC2N
      
      NOTE: remember to load an IPython module first. You can see possible modules with 

      .. code-block:: console

         $ module spider IPython

      And load one of them (here 8.14.0) with

      .. code-block:: console
	 
        $ ml GCC/12.3.0 IPython/8.14.0 
         
      Then start Ipython with (lowercase):
      
      .. code-block:: console

         $ ipython 

      HPC2N also has ``JupyterLab`` installed. It is available as a module, but the process of using it is somewhat involved. We will cover it more under the session on <a href="https://uppmax.github.io/HPC-python/interactive.html">Interactive work on the compute nodes</a>. Otherwise, see this tutorial: 

      - https://www.hpc2n.umu.se/resources/software/jupyter 

   .. tab:: LUNARC 

      NOTE: remember to load an IPython module first. You can see possible modules with 

      .. code-block:: console

         $ module spider IPython

      And load one of them (here 8.14.0) with

      .. code-block:: console
         
        $ ml GCC/12.3.0 IPython/8.14.0 
         
      Then start Ipython with (lowercase):
      
      .. code-block:: console

         $ ipython 

      LUNARC also has ``JupyterLab``, ``JupyterNotebook``, and ``JupyterHub`` installed.  

   .. tab:: NSC (Tetralith) 

      NOTE: remember to load an IPython module first. You can see possible modules with 

      .. code-block:: console

         $ module spider IPython

      And load one of them (here 8.5.0) with

      .. code-block:: console
         
        $ ml buildtool-easybuild/4.8.0-hpce082752a2  GCC/11.3.0 IPython/8.5.0
         
      Then start Ipython with (lowercase):
      
      .. code-block:: console

         $ ipython 


- Exit IPython with <Ctrl-D>, ``quit()`` or ``exit()`` in the python prompt


iPython

.. code-block:: ipython

    In [2]: <Ctrl-D>
    In [12]: quit()
    In [17]: exit()


Packages/Python modules
-----------------------


.. admonition:: Python modules AKA Python packages

   - Python **packages broaden the use of python** to almost infinity! 

   - Instead of writing code yourself there may be others that have done the same!

   - Many **scientific tools** are distributed as **python packages**, making it possible to run a script in the prompt and there define files to be analysed and arguments defining exactly what to do.

   - A nice **introduction to packages** can be found here: `Python for scientific computing <https://aaltoscicomp.github.io/python-for-scicomp/dependencies/>`_

.. questions::

   - How do I find which packages and versions are available?
   - What to do if I need other packages?
   - Are there differences between HPC2N, LUNARC, UPPMAX, and NSC?
   
.. objectives:: 

   - Show how to check for Python packages
   - show how to install own packages on the different clusters

Check current available packages
-------------------------------- 

General for all four centers
############################# 

Some python packages are working as stand-alone tools, for instance in bioinformatics. The tool may be already installed as a module. Check if it is there by:

.. code-block:: console

   $ module spider <tool-name or tool-name part> 
    
Using ``module spider`` lets you search regardless of upper- or lowercase characters and regardless of already loaded modules (like ``GCC`` on HPC2N and ``bioinfo-tools`` on UPPMAX).

.. tabs::

   .. tab:: UPPMAX

	Check the pre-installed packages of a specific python module:

	.. code-block:: console

	   $ module help python/<version> 
  
	
	
   .. tab:: HPC2N
   
	At HPC2N, a way to find Python packages that you are unsure how are names, would be to do

	.. code-block:: console

	   $ module -r spider ’.*Python.*’
   
	or

	.. code-block:: console

	   $ module -r spider ’.*python.*’
   
	Do be aware that the output of this will not just be Python packages, some will just be programs that are compiled with Python, so you need to check the list carefully.   

   .. tab:: LUNARC 

      At LUNARC, a way to find Python packages that you are unsure how are names, would be to do

        .. code-block:: console

           $ module -r spider ’.*Python.*’
   
        or

        .. code-block:: console

           $ module -r spider ’.*python.*’
   
        Do be aware that the output of this will not just be Python packages, some will just be programs that are compiled with Python, so you need to check the list carefully.   

   
Check the pre-installed packages of a loaded python module, in shell:

.. code-block:: console

   $ pip list

To see which Python packages you, yourself, has installed, you can use ``pip list --user`` while the environment you have installed the packages in are active.

You can also test from within python to make sure that the package is not already installed:

.. code-block:: python 

    >>> import <package>
    
Does it work? Then it is there!

Otherwise, you can either use ``pip`` or ``conda``.

.. admonition:: Check path to the package you are using,

   - In a python session, type:

   .. code-block:: python
   
      import [a_module]
      print([a_module].__file__)

   - The print-out tells you the path to the `.pyc` file, but should give you a hint where it belongs.

.. exercise:: Check packages (5 min)

   - See if the following packages are installed. Use python version ``3.11.8`` on Rackham and ``3.11.3`` on Kebnekaise/Cosmos (remember: the Python module on kebnekaise/cosmos has a prerequisite). 

      - ``numpy``
      - ``mpi4py``
      - ``distributed``
      - ``multiprocessing``
      - ``time``
      - ``dask``
      
.. solution::

   - Rackham has for ordinary python/3.11.8 module already installed: 
      - ``numpy`` |:white_check_mark:|
      - ``pandas`` |:white_check_mark:|
      - ``mpi4py`` |:x:|
      - ``distributed`` |:x:|
      - ``multiprocessing`` |:white_check_mark:|  (standard library)
      - ``time`` |:white_check_mark:|  (standard library)
      - ``dask`` |:white_check_mark:|

   - Kebnekaise has for ordinary Python/3.11.3 module already installed:
      - ``numpy`` |:x:|
      - ``pandas`` |:x:| 
      - ``mpi4py`` |:x:|
      - ``distributed`` |:x:|
      - ``multiprocessing`` |:white_check_mark:|  (standard library)
      - ``time`` |:white_check_mark:|  (standard library)
      - ``dask``  |:x:|

   - Cosmos has for ordinary Python/3.11.3 module already installed: 
      - ``numpy`` |:x:|
      - ``pandas`` |:x:| 
      - ``mpi4py`` |:x:|
      - ``distributed`` |:x:|
      - ``multiprocessing`` |:white_check_mark:|  (standard library)
      - ``time`` |:white_check_mark:|  (standard library)
      - ``dask``  |:x:|

   - See next session how to find more pre-installed packages!

**NOTE**: at HPC2N and LUNARC, the available Python packages needs to be loaded as modules/module-bundles before using! See a list of some of them below, under the HPC2N/LUNARC tab or find more as mentioned above, using ``module spider -r ...``

A selection of the Python packages and libraries installed on UPPMAX and HPC2N are given in extra reading: `UPPMAX clusters <https://uppmax.github.io/HPC-python/uppmax.html>`_ and `Kebnekaise cluster <https://uppmax.github.io/HPC-python/kebnekaise.html>`_

.. tabs::

   .. tab:: UPPMAX

      - The python application at UPPMAX comes with several preinstalled packages. 
      - You can check them here: `UPPMAX packages <https://uppmax.github.io/HPC-python/uppmax.html#uppmax-packages>`_.
      - In addition there are packages available from the module system as `python tools/packages <https://uppmax.github.io/HPC-python/uppmax.html#uppmax-packages>`_
      - Note that bioinformatics-related tools can be reached only after loading ``bioinfo-tools``. 
      - Two modules contains topic specific packages. These are:
         
         - Machine learning: ``python_ML_packages`` (cpu and gpu versions and based on python/3.9.5 and python/3.11.8)
	 - GIS: ``python_GIS_packages`` (cpu version based on python/3.10.8)

   .. tab:: HPC2N

      - The python application at HPC2N comes with several preinstalled packages - check first before installing yourself! 
      - HPC2N has both Python 2.7.x and Python 3.x installed. 
      - We will be using Python 3.x in this course.  For this course, the recommended version of Python to use on Kebnekaise is 3.11.3.

	NOTE:  HPC2N do NOT recommend (and do not support) using Anaconda/Conda on our systems. You can read more about this here: `Anaconda <https://docs.hpc2n.umu.se/tutorials/anaconda/>`_.


      - This is a selection of the packages and libraries installed at HPC2N. These are all installed as **modules** and need to be loaded before use. 
	
	  - ``ASE``
	  - ``Keras``
	  - ``PyTorch``
	  - ``SciPy-bundle`` (Bottleneck, deap, mpi4py, mpmath, numexpr, numpy, pandas, scipy - some of the versions have more)
	  - ``TensorFlow``
	  - ``Theano``
	  - ``matplotlib``
	  - ``scikit-learn``
	  - ``scikit-image``
	  - ``iPython``
	  - ``Cython``
	  - ``Flask``
          - ``JupyterLab``  
          - ``Python-bundle-PyPI`` (Bundle of Python packages from PyPi)

   .. tab:: LUNARC 

      - The python application at LUNARC comes with several preinstalled packages - check first before installing yourself! 
      - LUNARC has both Python 2.7.x and Python 3.x installed. 
      - We will be using Python 3.x in this course.  For this course, the recommended version of Python to use on Kebnekaise is 3.11.3.

      - This is a selection of the packages and libraries installed at LUNARC. These are all installed as **modules** and need to be loaded before use. 

          - ``PyTorch``
          - ``SciPy-bundle`` (Bottleneck, deap, mpi4py, mpmath, numexpr, numpy, pandas, scipy - some of the versions have more)
          - ``TensorFlow``
          - ``matplotlib``
          - ``scikit-learn``
          - ``scikit-image``
          - ``iPython``
          - ``Cython``
          - ``Biopython``  
          - ``JupyterLab`` 
          - ``Python-bundle`` (NumPy, SciPy, Matplotlib, JupyterLab, MPI4PY, ...)  
  

Demo/Type-along 
---------------

This is an exercise that combines loading, running, and using site-installed packages. Later, during the ML session, we will look at running the same exercise, but as a batch job. There is also a follow-up exercise of an extended version of the script, if you want to try run that as well (see further down on the page). 

.. note:: 

    You need the data-file ``scottish_hills.csv`` which can be found in the directory ``Exercises/examples/programs``. If you have cloned the git-repo for the course, or copied the tar-ball, you should have this directory. The easiest thing to do is just change to that directory and run the exercise there. 

    Since the exercise opens a plot, you need to login with ThinLinc (or otherwise have an x11 server running on your system and login with ``ssh -X ...``). 

The exercise is modified from an example found on https://ourcodingclub.github.io/tutorials/pandas-python-intro/. 

.. warning::

   **Not relevant if using UPPMAX. Only if you are using HPC2N or LUNARC!**

   You need to also load Tkinter. 

   **For HPC2N:**

   .. code-block:: console 

      ml GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 Tkinter/3.11.3

   **For LUNARC**

   .. code-block:: console

      ml GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 matplotlib/3.8.2 Tkinter/3.11.5 

   In addition, you need to add the following two lines to the top of your python script/run them first in Python, for both HPC2N and LUNARC:

   .. code-block:: python

      import matplotlib
      matplotlib.use('TkAgg')

.. exercise:: Python example with packages pandas and matplotlib 

   We are using Python version ``3.11.x``. To access the packages ``pandas`` and ``matplotlib``, you may need to load other modules, depending on the site where you are working. 
     
   .. tabs:: 

      .. tab:: UPPMAX

         Here you only need to load the ``python`` module, as the relevant packages are included (as long as you are not using GPUs, but that is talked about later in the course). Thus, you just do: 

        .. code-block:: console

           $ ml python/3.11.8

      .. tab:: HPC2N

         On Kebnekaise you also need to load ``SciPy-bundle`` and ``matplotlib`` (and their prerequisites). These versions will work well together (and with the Tkinter/3.11.3): 

         .. code-block:: console

            $ ml GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2
   
      .. tab:: LUNARC

         On Cosmos you also need to load ``SciPy-bundle`` and ``matplotlib`` (and their prerequisites). These versions will work well together (and with the Tkinter/3.11.5): 

         .. code-block:: console

            $ ml GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 matplotlib/3.8.2     
   1. From inside Python/interactive (if you are on Kebnekaise/Cosmos, mind the warning above about loading a compatible Tkinter and adding the two lines importing matplotlib and setting TkAgg at the top):

      Start python and run these lines: 

      .. code-block:: python

         import pandas as pd

      .. code-block:: python

         import matplotlib.pyplot as plt

      .. code-block:: python

         dataframe = pd.read_csv("scottish_hills.csv")

      .. code-block:: python

         x = dataframe.Height

      .. code-block:: python

         y = dataframe.Latitude

      .. code-block:: python 

         plt.scatter(x, y)

      .. code-block:: python

         plt.show()

      If you change the last line to ``plt.savefig("myplot.png")`` then you will instead get a file ``myplot.png`` containing the plot. This is what you would do if you were running a python script in a batch job. 

      - On UPPMAX and LUNARC you can view png files with the program ``eog``
	   - Test: ``eog myplot.png &``
      - On HPC2N you can view png files with the program ``eom``
	   - Test: ``eom myplot.png &``

   2. As a Python script (if you are on Kebnekaise/LUNARC, mind the warning above):

      Copy and save this script as a file (or just run the file ``pandas_matplotlib-<system>.py`` that is located in the ``<path-to>/Exercises/examples/programs`` directory you got from the repo or copied. Where <system> is either ``rackham`` or ``kebnekaise``. 

      .. tabs::

	 .. tab:: rackham

	    .. code-block:: python
 
  	       import pandas as pd
               import matplotlib.pyplot as plt

               dataframe = pd.read_csv("scottish_hills.csv")
               x = dataframe.Height
               y = dataframe.Latitude
               plt.scatter(x, y)
               plt.show()

	 .. tab:: kebnekaise

	    .. code-block:: python

	       import pandas as pd
	       import matplotlib
	       import matplotlib.pyplot as plt
	      
               matplotlib.use('TkAgg')

	       dataframe = pd.read_csv("scottish_hills.csv")
               x = dataframe.Height
               y = dataframe.Latitude
               plt.scatter(x, y)
               plt.show()
	      
         .. tab:: Cosmos 

            .. code-block:: python 

               import pandas as pd
               import matplotlib
               import matplotlib.pyplot as plt
              
               matplotlib.use('TkAgg')

               dataframe = pd.read_csv("scottish_hills.csv")
               x = dataframe.Height
               y = dataframe.Latitude
               plt.scatter(x, y)
               plt.show()
      
If you have time, you can also try and run these extended versions, which also requires the ``scipy`` packages (included with python at UPPMAX and with the same modules loaded as for ``pandas`` for HPC2N/LUNARC):

Exercises  (C. 10 min)
----------------------



.. exercise:: Python example that requires ``pandas``, ``matplotlib``, and ``scipy`` packages.

   You can either save the scripts or run them line by line inside Python. The scripts are also available in the directory ``<path-to>/Exercises/examples/programs``, as ``pandas_matplotlib-linreg.py`` and ``pandas_matplotlib-linreg-pretty.py``.

   NOTE that there are separate versions for rackham, kebnekaise, and cosmos and that you for kebnekaise and cosmos need to again add the same lines as mentioned under the warning before the previous exercise. 

   Remember that you also need the data file ``scottish_hills.csv`` located in the above directory. 

   Examples are from https://ourcodingclub.github.io/tutorials/pandas-python-intro/

   ``pandas_matplotlib-linreg.py``

   .. code-block:: python 

      import pandas as pd
      import matplotlib.pyplot as plt
      from scipy.stats import linregress

      dataframe = pd.read_csv("scottish_hills.csv")

      x = dataframe.Height
      y = dataframe.Latitude

      stats = linregress(x, y)

      m = stats.slope
      b = stats.intercept

      plt.scatter(x, y)
      plt.plot(x, m * x + b, color="red")   # I've added a color argument here

      plt.show()

   ``pandas_matplotlib-linreg-pretty.py``

   .. code-block:: python

      import pandas as pd
      import matplotlib.pyplot as plt
      from scipy.stats import linregress

      dataframe = pd.read_csv("scottish_hills.csv")

      x = dataframe.Height
      y = dataframe.Latitude

      stats = linregress(x, y)

      m = stats.slope
      b = stats.intercept

      # Change the default figure size
      plt.figure(figsize=(10,10))

      # Change the default marker for the scatter from circles to x's
      plt.scatter(x, y, marker='x')

      # Set the linewidth on the regression line to 3px
      plt.plot(x, m * x + b, color="red", linewidth=3)

      # Add x and y lables, and set their font size
      plt.xlabel("Height (m)", fontsize=20)
      plt.ylabel("Latitude", fontsize=20)

      # Set the font size of the number lables on the axes
      plt.xticks(fontsize=18)
      plt.yticks(fontsize=18)

      plt.show()

.. keypoints::

   - Before you can run Python scripts or work in a Python shell, first load a python module and probable prerequisites
   - Start a Python shell session either with ``python`` or ``ipython``
   - Run scripts with ``python3 <script.py>``
   - You can check for packages 
   
   	- from the Python shell with the ``import`` command
	- from BASH shell with the 
	
		- ``pip list`` command at all three centers
		- ``ml help python/<version>`` at UPPMAX
		
   - Installation of Python packages can be done either with **PYPI** or **Conda**
   - You install own packages with the ``pip install`` command (This is the recommended way on HPC2N)
   - At UPPMAX and LUNARC Conda is also available (See Conda section)

