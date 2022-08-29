Isolated environments
=====================

.. questions::

   - What syntax is used to make a lesson?
   - How do you structure a lesson effectively for teaching?

   ``questions`` are at the top of a lesson and provide a starting
   point for what you might learn.  It is usually a bulleted list.
   (The history is a holdover from carpentries-style lessons, and is
   not required.)
   
.. objectives:: 

   - Show how to load Python
   - show how to run Python scripts and start the Python commandline

At both UPPMAX and HPC2N we call the applications available via the module system modules. 
    - https://www.uppmax.uu.se/resources/software/module-system/ 
    - https://www.hpc2n.umu.se/documentation/environment/lmod 
    
    
.. note::
   Isolated environments solve a couple of problems:
   
   - You can install specific, also older, versions into them.
   - You can create one for each project and no problem if the two projects require different versions.
   - You can remove the environment and create a new one, if not needed or with errors.
   
``conda`` works as an isolated environment. Below we present the ``pip`` way with "virtual environments", as well as installing using setup.py! Installing with a virtual environment is the only recommended way at HPC2N! 

Virtual environment - venv (UPPMAX)
-----------------------------------


More on virtual environment: https://docs.python.org/3/tutorial/venv.html 

Virtual environment - vpyenv (HPC2N)
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

   - What the learner should take away
   - point 2
   
