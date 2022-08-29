Load and run python
===================

At both UPPMAX and HPC2N we call the applications available via the module system modules. 
    - https://www.uppmax.uu.se/resources/software/module-system/ 
    - https://www.hpc2n.umu.se/documentation/environment/lmod 

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
    
- For reproducibility reasons, you should always load a specific version of a module instead of just the default version
- Many modules have prerequisite modules which needs to be loaded first (at HPC2N this is also the case for the Python modules). When doing ``module spider <module>/<version>`` you will get a list of which other modules needs to be loaded first

.. tabs::

   .. tab:: UPPMAX

      Load latest python module by:

      .. code-block:: sh

        $ module load python
    
      Check all available versions with:

      .. code-block:: sh

          $ module available python



   .. tab:: HPC2N

      For reproducibility, at HPC2N we recommend ALWAYS loading a specific module instad of using the default version! 

      For this course, we recommend using Python 3.9.5 at HPC2N. To load this version, load the prerequisites and then the module: 

      .. code-block:: sh

         $ module load GCC/10.3.0 Python/3.9.5

      For short, you can also use: 

      .. code-block:: sh

        $ ml GCC/10.3.0 Python/3.9.5

      Check all available version Python versions with:

      .. code-block:: sh
 
         $ module spider Python

     





.. admonition:: Output at UPPMAX as of March 9 2022
    :class: dropdown
    
    .. prompt::  text
    
        -------------------------------------- /sw/mf/rackham/applications ---------------------------------------
           python_ML_packages/3.9.5    wrf-python/1.3.1

        --------------------------------------- /sw/mf/rackham/compilers ----------------------------------------
           python/2.7.6     python/3.3      python/3.6.0    python/3.9.5  (D)    python3/3.8.7
           python/2.7.9     python/3.3.1    python/3.6.8    python3/3.6.0        python3/3.9.5 (D)
           python/2.7.11    python/3.4.3    python/3.7.2    python3/3.6.8
           python/2.7.15    python/3.5.0    python/3.8.7    python3/3.7.2

          Where:
          D:  Default Module

        Use module spider" to find all possible modules and extensions.
        Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".


Load specific version (recommendation for reproducibility) with:

.. code-block:: sh

    $ module load python/X.Y.Z

.. warning::

    + Don’t use system-installed python/2.7.5
    + ALWAYS use python module

.. admonition:: Why are there both python/3.X.Y and python3/3.X.Y modules?

    Sometimes existing software might use `python2` and there's nothing you can do about that. In pipelines and other toolchains the different tools may together require both `python2` and `python3`.
    Here's how you handle that situation:
    
    + You can run two python modules at the same time if ONE of the module is ``python/2.X.Y`` and the other module is ``python3/3.X.Y`` (not ``python/3.X.Y``).
    
Load (HPC2N)
------------
For reproducibility, at HPC2N we recommend ALWAYS loading a specific module instad of using the default version! 

For this course, we recommend using Python 3.9.5 at HPC2N. To load this version, load the prerequisites and then the module: 

.. code-block:: sh

   $ module load GCC/10.3.0 Python/3.9.5

For short, you can also use: 

.. code-block:: sh

    $ ml GCC/10.3.0 Python/3.9.5

Check all available version Python versions with:

.. code-block:: sh
 
   $ module spider Python

.. admonition:: Output as of 27 July 2022
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
            Other possible modules matches:
                Biopython  Boost.Python  GitPython  IPython  flatbuffers-python  ...
           ----------------------------------------------------------------------------
           To find other possible module matches execute:
               $ module -r spider '.*Python.*'
           ----------------------------------------------------------------------------
           For detailed information about a specific "Python" package (including how to load the modules) use the module's full name.
               Note that names that have a trailing (E) are extensions provided by other modules.
       
           For example:
            $ module spider Python/3.9.6
           ----------------------------------------------------------------------------

To see how to load a specific version of Python, including the prerequisites, do 

.. code-block:: sh
   
   $ module spider Python/<version>

Example for Python 3.9.5

.. code-block:: sh

   $ module spider Python/3.9.6

.. warning::

    + Do not use the system-installed Python 2.7.18
    + ALWAYS use a Python module

.. admonition:: Why are there both Python/2.X.Y and Python/3.Z.W modules?

    Some existing software might use `Python2` and some will use `Python3`. Some of the Python packages have both `Python2` and `Python3` versions. Check what your software as well as the installed modules need when you pick!   

Run (UPPMAX)
------------

You can run a python script in the shell like this:

.. code-block:: sh

    $ python example.py

or, if you loaded a python3 module:

.. code-block:: sh

    $ python3 example.py

You start a python session/prompt ( >>> ) by typing:

.. code-block:: sh

    $ python  # or python3

    #for interactive 
    ipython # or ipython3 
    
Exit with <Ctrl-D>, "quit()" or 'exit()’ in python prompt

.. code-block:: python

    >>> <Ctrl-D>
    >>> quit()
    >>> exit()

Run (HPC2N)
------------

You can run a python script in the shell like this:

.. code-block:: sh

   $ python example.py

or, if you loaded a python3 module, you can use:

.. code-block:: sh

   $ python3 example.py

since python is a symbolic link to python3 in this case. 

NOTE: *only* run jobs that are short and/or do not use a lot of resources from the command line. Otherwise use the batch system!

You start a python session/prompt ( >>> ) by typing:

.. code-block:: sh

    $ python  
    
or 
    
.. code-block:: sh

    $ python3

Exit Python with <Ctrl-D>, "quit()" or 'exit()’ in the python prompt

.. code-block:: python

    >>> <Ctrl-D>
    >>> quit()
    >>> exit()

In addition to loading Python, you will also often need to load site-installed modules for Python packages, or use own-installed Python packages. The work-flow would be something like this: 

1) Load Python and prerequisites: `module load <pre-reqs> Python/<version>``
2) Load site-installed Python packages (optional): ``module load <pre-reqs> <python-package>/<version>``
3) Activate your virtual environment (optional): ``source <path-to-virt-env>/bin/activate``
4) Install any extra Python packages (optional): ``pip install --no-cache-dir --no-build-isolation <python-package>``
5) Start Python: ``python``

Installed Python modules (modules and own-installed) can be accessed within Python with ``import <package>`` as usual. 

The command ``pip list`` given within Python will list the available modules to import. 

More about virtual/isolated environment to follow in later sections of the course! 

For interactive Python, IPython, start a session with 

.. code-block:: sh

    $ ipython 
    
or 

.. code-block:: sh

    $ ipython3 
    
NOTE: remember to load an IPython module first. You can see possible modules with 

.. code-block:: sh

    $ module spider IPython
    

More information will follow later in the course on running Python from within a **batch job**. 
