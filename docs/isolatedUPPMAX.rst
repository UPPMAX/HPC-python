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


.. note::
   Isolated environments solve a couple of problems:
   
   - You can install specific, also older, versions into them.
   - You can create one for each project and no problem if the two projects require different versions.
   - You can remove the environment and create a new one, if not needed or with errors.
   
``conda`` works as an isolated environment. Below we present the ``pip`` way with "virtual environments", as well as installing using setup.py! Installing with a virtual environment is the only recommended way at HPC2N! 

Virtual environment - venv (UPPMAX)
-----------------------------------

Create a ``venv``. First load the python version you want to base your virtual environment on:

.. code-block:: sh

    $ module load python/<version>
    $ python -m venv Example
    
"Example" is the name of the virtual environment. The directory “Example” is created in the present working directory.

If you want it in a certain place like "~/test/":

.. code-block:: sh

    $ python -m venv ~/test/Example 
    
Activate it.

.. code-block:: sh

    $ source <path/>Example/bin/activate

Note that your prompt is changing to start with (Example) to show that you are within an environment.

Install your packages with ``pip`` and the correct versions, like:

.. prompt:: 
    :language: bash
    :prompts: (Example) $

    pip install numpy==1.13.1 matplotlib==2.2.2

Deactivate it.

.. prompt:: 
    :language: bash
    :prompts: (Example) $

    deactivate

Everytime you need the tools available in the virtual environment you activate it as above.

.. prompt:: bash $

    source <path/>Example/bin/activate

More on virtual environment: https://docs.python.org/3/tutorial/venv.html 

.. keypoints::

   - What the learner should take away
   - point 2
    
