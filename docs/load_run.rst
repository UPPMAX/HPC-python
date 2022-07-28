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


Load (UPPMAX)
-------------
Load latest python module by:

.. prompt:: bash $

    module load python
    
Check all available versions with:

.. prompt:: bash $

    module available python

.. admonition:: Output as of March 9 2022
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

.. prompt:: bash $

    module load python/X.Y.Z

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

    ``module load GCC/10.3.0 Python/3.9.5``

For short, you can also use: 

    ``ml GCC/10.3.0 Python/3.9.5``

Check all available version Python versions with:

    ``module spider Python``

.. admonition:: Output as of 27 July 2022
    :class: dropdown
        :verbatim:

        b-an01 [~]$ module spider Python

        ----------------------------------------------------------------------------
        Python:
        ----------------------------------------------------------------------------
        Description:
            Python is a programming language that lets you work more quickly and
            integrate your systems more effectively.
    
         Versions:
             | Python/2.7.15   
             | Python/2.7.16  
             | Python/2.7.18-bare 
             | Python/2.7.18  
             | Python/3.7.2   
             | Python/3.7.4   
             | Python/3.8.2   
             | Python/3.8.6   
             | Python/3.9.5-bare  
             | Python/3.9.5   
             | Python/3.9.6-bare  
             | Python/3.9.6   
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

    ``module spider Python/<version>``

Example for Python 3.9.5

    ``module spider Python/3.9.6``

.. warning::

    + Do not use the system-installed Python 2.7.18
    + ALWAYS use a Python module

.. admonition:: Why are there both Python/2.X.Y and Python/3.Z.W modules?

    Some existing software might use `Python2` and some will use `Python3`. Some of the Python packages have both `Python2` and `Python3` versions. Check what your software as well as the installed modules need when you pick!   

Run
---

You can run a python script in the shell by:

.. prompt:: bash $

    python example.py

or, if you loaded a python3 module:

.. prompt:: bash $

    python3 example.py

You start a python session/prompt ( >>> ) by typing:

.. prompt:: bash $

    python  # or python3

    #for interactive 
    ipython # or ipython3 
    
Exit with <Ctrl-D>, "quit()" or 'exit()’ in python prompt

.. prompt:: python >>>

    <Ctrl-D>
    quit()
    exit()
