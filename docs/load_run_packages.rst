Load and run python and use packages
====================================

At both UPPMAX and HPC2N we call the applications available via the module system modules. 
    - https://www.uppmax.uu.se/resources/software/module-system/ 
    - https://www.hpc2n.umu.se/documentation/environment/lmod 

   
.. objectives:: 

   - Show how to load Python
   - Show how to run Python scripts and start the Python command line

.. admonition:: Short cheat sheet
    :class: dropdown 
    
    - See which modules exists: ``module spider`` or ``ml spider``
    - Find module versions for a particular software: ``module spider <software>``
    - Modules depending only on what is currently loaded: ``module avail`` or ``ml av``
    - See which modules are currently loaded: ``module list`` or ``ml``
    - Load a module: ``module load <module>/<version>`` or ``ml <module>/<version>``
    - Unload a module: ``module unload <module>/<version>`` or ``ml -<module>/<version>``
    - More information about a module: ``module show <module>/<version>`` or ``ml show <module>/<version>``
    - Unload all modules except the 'sticky' modules: ``module purge`` or ``ml purge``
    
.. warning::
   
   - Note that the module systems at UPPMAX and HPC2N are slightly different. 
   - While all modules at UPPMAX not directly related to bio-informatics are shown by ``ml avail``, modules at HPC2N are hidden until one has loaded a prerequisite like the compiler ``GCC``.


- For reproducibility reasons, you should always load a specific version of a module instead of just the default version
- Many modules have prerequisite modules which needs to be loaded first (at HPC2N this is also the case for the Python modules). When doing ``module spider <module>/<version>`` you will get a list of which other modules needs to be loaded first


Check for Python versions
-------------------------

.. tip::
    
   **Type along!**

.. tabs::

   .. tab:: UPPMAX

     Check all available Python versions with:

      .. code-block:: console

          $ module avail python


   .. tab:: HPC2N
   
      Check all available version Python versions with:

      .. code-block:: console
 
         $ module spider Python
      
      To see how to load a specific version of Python, including the prerequisites, do 

      .. code-block:: console
   
         $ module spider Python/<version>

      Example for Python 3.9.5

      .. code-block:: console

         $ module spider Python/3.9.5 

.. admonition:: Output at UPPMAX as of Nov 30 2023
   :class: dropdown
    
       .. code-block::  tcl
    
          -------------------------------------- /sw/mf/rackham/applications ---------------------------------------
           python_GIS_packages/3.10.8      python_ML_packages/3.9.5-gpu (D)
           python_ML_packages/3.9.5-cpu    wrf-python/1.3.1

           --------------------------------------- /sw/mf/rackham/compilers ----------------------------------------
           python/2.7.6     python/3.3      python/3.6.0    python/3.9.5           python3/3.7.2
           python/2.7.6     python/3.3.1    python/3.7.2         python3/3.6.0    python3/3.10.8
           python/2.7.9     python/3.4.3    python/3.8.7         python3/3.6.8    python3/3.11.4 (D)
           python/2.7.11    python/3.5.0    python/3.9.5         python3/3.7.2
           python/2.7.15    python/3.6.0    python/3.10.8        python3/3.8.7
           python/3.3       python/3.6.8    python/3.11.4 (D)    python3/3.9.5

           Where:
           D:  Default Module

           Use module spider" to find all possible modules and extensions.
           Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".

.. admonition:: Output at HPC2N as of 30 Nov 2023
    :class: dropdown

        .. code-block:: tcl

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
                Python/3.11.3
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

Load a Python module
--------------------

For reproducibility, we recommend ALWAYS loading a specific module instad of using the default version! 

For this course, we recommend using Python 3.9.5.

.. tip::
    
   **Type along!**


.. tabs::

   .. tab:: UPPMAX
   
      Go back and check which Python modules were available. To load version 3.9.5, do:

      .. code-block:: console

        $ module load python/3.9.5
        
      Note: Lowercase ``p``.
      For short, you can also use: 

      .. code-block:: console

         $ ml python/3.9.5

 
   .. tab:: HPC2N

 
      .. code-block:: console

         $ module load GCC/10.3.0 Python/3.9.5

      Note: Uppercase ``P``.   
      For short, you can also use: 

      .. code-block:: console

         $ ml GCC/10.3.0 Python/3.9.5

.. warning::

   + UPPMAX: Don’t use system-installed python (2.7.5)
   + UPPMAX: Don't use system installed python3 (3.6.8)
   + HPC2N: Don’t use system-installed python (2.7.18)
   + HPC2N: Don’t use system-installed python3  (3.8.10)
   + ALWAYS use python module

.. admonition:: Why are there both Python/2.X.Y and Python/3.Z.W modules?

    Some existing software might use `Python2` and some will use `Python3`. Some of the Python packages have both `Python2` and `Python3` versions. Check what your software as well as the installed modules need when you pick!   
    
.. admonition:: UPPMAX: Why are there both python/3.X.Y and python3/3.X.Y modules?

    Sometimes existing software might use `python2` and there's nothing you can do about that. In pipelines and other toolchains the different tools may together require both `python2` and `python3`.
    Here's how you handle that situation:
    
    + You can run two python modules at the same time if ONE of the module is ``python/2.X.Y`` and the other module is ``python3/3.X.Y`` (not ``python/3.X.Y``).
    
Run
---

Run Python script
#################

    
You can run a python script in the shell like this:

.. code-block:: console

   $ python example.py
   # or 
   $ python3 example.py


since python is a symbolic link to python3 in this case. 

Or, if you loaded a python3 module, you can use:

.. code-block:: console

   $ python3 example.py

NOTE: *only* run jobs that are short and/or do not use a lot of resources from the command line. Otherwise use the batch system (see the [batch session](https://uppmax.github.io/HPC-python/batch.html))
    
.. note::

   Real cases will be tested in the [**batch session**](https://uppmax.github.io/R-python-julia-HPC/python/batchPython.html). 

Run an interactive Python shell
###############################

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
      From python/3.10.8 also jupyter-lab is available.
         
    
   .. tab:: HPC2N
      
      NOTE: remember to load an IPython module first. You can see possible modules with 

      .. code-block:: console

         $ module spider IPython
         $ ml IPython/7.25.0
         
      Then start Ipython with (lowercase):
      
      .. code-block:: console

         $ ipython 

      HPC2N also has ``JupyterLab`` installed. It is available as a module, but the process of using it is somewhat involved. We will cover it more under the session on <a href="https://uppmax.github.io/HPC-python/interactive.html">Interactive work on the compute nodes</a>. Otherwise, see this tutorial: 

      - https://www.hpc2n.umu.se/resources/software/jupyter 


**Example**

.. code-block:: python

   >>> a=3
   >>> b=7
   >>> c=a+b
   >>> c
   10


- Exit Python or IPython with <Ctrl-D>, ``quit()`` or ``exit()`` in the python prompt

Python

.. code-block:: python

    >>> <Ctrl-D>
    >>> quit()
    >>> exit()

iPython

.. code-block:: ipython

    In [2]: <Ctrl-D>
    In [12]: quit()
    In [17]: exit()

## Packages/Python modules

.. admonition:: Python modules AKA Python packages

   - Python **packages broaden the use of python** to almost infinity! 

   - Instead of writing code yourself there may be others that have done the same!

   - Many **scientific tools** are distributed as **python packages**, making it possible to run a script in the prompt and there define files to be analysed and arguments defining exactly what to do.

   - A nice **introduction to packages** can be found here: `Python for scientific computing <https://aaltoscicomp.github.io/python-for-scicomp/dependencies/>`_

.. questions::

   - How do I find which packages and versions are available?
   - What to do if I need other packages?
   - Are there differences between HPC2N and UPPMAX?
   
.. objectives:: 

   - Show how to check for Python packages
   - show how to install own packages on the different clusters
Check current available packages
--------------------------------

General for both centers
########################

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
   
Check the pre-installed packages of a loaded python module, in shell:

.. code-block:: console

   $ pip list

To see which Python packages you, yourself, has installed, you can use ``pip list --user`` while the environment you have installed the packages in are active.

You can also test from within python to make sure that the package is not already installed:

.. code-block:: python 

    >>> import <package>
    
Does it work? Then it is there!
Otherwise, you can either use ``pip`` or ``conda``.


**NOTE**: at HPC2N, the available Python packages needs to be loaded as modules before using! See a list of some of them below, under the HPC2N tab or find more as mentioned above, using ``module spider -r ....``

A selection of the Python packages and libraries installed on UPPMAX and HPC2N are give in extra reading: `UPPMAX clusters <https://uppmax.github.io/HPC-python/uppmax.html>`_ and `Kebnekaise cluster <https://uppmax.github.io/HPC-python/kebnekaise.html>`_

.. tabs::

   .. tab:: UPPMAX

      - The python application at UPPMAX comes with several preinstalled packages. 
      - You can check them here: `UPPMAX packages <https://uppmax.github.io/HPC-python/uppmax.html#uppmax-packages>`_.
      - In addition there are packages available from the module system as `python tools/packages <https://uppmax.github.io/HPC-python/uppmax.html#uppmax-packages>`_
      - Note that bioinformatics-related tools can be reached only after loading ``bioinfo-tools``. 
      - Two modules contains topic specific packages. These are:
         
         - Machine learning: ``python_ML_packages`` (cpu and gpu versions and based on python/3.9.5)
	 - GIS: ``python_GIS_packages`` (cpu vrson based on python/3.10.8)

   .. tab:: HPC2N

      - The python application at HPC2N comes with several preinstalled packages - check first before installing yourself!. 
      - HPC2N has both Python 2.7.x and Python 3.x installed. 
      - We will be using Python 3.x in this course.  For this course, the recommended version of Python to use on Kebnekaise is 3.9.5

	NOTE:  HPC2N do NOT recommend (and do not support) using Anaconda/Conda on our systems. You can read more about this here: `Anaconda <https://www.hpc2n.umu.se/documentation/guides/anaconda>`_.


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
	  - ``pip``
	  - ``iPython``
	  - ``Cython``
	  - ``Flask``


.. keypoints::

   - Before you can run Python scripts or work in a Python shell, first load a python module and probable prerequisites
   - Start a Python shell session either with ``python`` or ``ipython``
   - Run scripts with ``python3 <script.py>``
   - You can check for packages 
   
   	- from the Python shell with the ``import`` command
	- from BASH shell with the 
	
		- ``pip list`` command at both centers
		- ``ml help python/3.9.5`` at UPPMAX
		
   - Installation of Python packages can be done either with **PYPI** or **Conda**
   - You install own packages with the ``pip install`` command (This is the recommended way on HPC2N)
   - At UPPMAX Conda is also available (See Conda section)

