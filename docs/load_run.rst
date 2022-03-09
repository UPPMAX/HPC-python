Load and run Python
===================

At UPPMAX we call the applications available via the module system modules. 
https://www.uppmax.uu.se/resources/software/module-system/ 

Load
----------
Load latest Python module by:

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

        Use "module spider" to find all possible modules and extensions.
        Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".


Load specific version with:

.. prompt:: bash $

    module load python/X.Y.Z

.. warning::

    + Don’t use system-installed python/2.7.5
    + ALWAYS use python module

.. note::

    + Note that there are both 'python' and 'python3' modules for version '3.X.Y'.
    + You can run two python modules at the same time if ONE of the module is 'python/**2**.X.X'. The other module has then to be 'python3/3.Y.Z' (not python/3.X.Y).
    + Sometimes necessary in pipelines and other toolchains where the different tools require different python versions.

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
