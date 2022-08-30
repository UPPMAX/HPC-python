Isolated environments
=====================

.. note::
   Isolated environments solve a couple of problems:
   
   - You can install specific, also older, versions into them.
   - You can create one for each project and no problem if the two projects require different versions.
   - You can remove the environment and create a new one, if not needed or with errors.
   
``conda`` works as an isolated environment. Below we present the ``pip`` way with "virtual environments", as well as installing using setup.py! Installing with a virtual environment is the only recommended way at HPC2N! 

.. questions::

   - How to work with isolated environments at HPC2N and UPPMAX?
   - How do you structure a lesson effectively for teaching?

   
.. objectives:: 
   - Give a general 'theoretical* introduction to isolated environments 
   - Site-specific procedures are given at the separated sessions.

General procedures   
------------------
    
**Make an overview general for both clusters**
- general procedure
- the tools
   - venv
   - *virtualenv*
   - Conda
- point to separated sessions

.. admonition:: venv vs. virtualenv
These are almost completely interchangeable, the difference being that **virtualenv supports older python versions** and has a few more minor unique features, while **venv is in the standard library**.


Virtual environment - venv (UPPMAX)
-----------------------------------


Virtual environment - vpyenv (HPC2N) **should be called virtualenv?**
---------------------------------------------

Create a ``vpyenv``. First load the python version you want to base your virtual environment on:

.. code-block:: sh

    $ module load python/<version>
    $ virtualenv --system-site-packages vpyenv
    
"vpyenv" is the name of the virtual environment. You can name it whatever you want. The directory “vpyenv” is created in the present working directory.

**NOTE**: since it may take up a bit of space if you are installing many Python packages to your virtual environment, we **strongly** recommend you place it in your project storage! 

**NOTE**: if you need are for instance working with both Python 2 and 3, then you can of course create more than one virtual environment, just name them so you can easily remember which one has what. 

 
More info
'''''''''

More on virtual environment: https://docs.python.org/3/tutorial/venv.html 
HPC2N's documentation pages about installing Python packages and virtual environments: https://www.hpc2n.umu.se/resources/software/user_installed/python

.. keypoints::

   - With a virtual environment you can tailor an environment with specific versions for Python and packages, not interfering with other installed python versions and packages.
   - Make it for each project you have for reproducibility.
   - There are different tools to create virtual environemnts.
     - UPPMAX have Conda and venv
     - HPC2N has virtualenv.
     - More details in the seperated sessions!
 
   
