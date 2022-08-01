Isolated ("virtual") environments
=====================

.. note::
   Isolated environments solve a couple of problems:
   
   - You can install specific, also older, versions into them.
   - You can create one for each project and no problem if the two projects require different versions.
   - You can remove the environment and create a new one, if not needed or with errors.
   
``conda`` works as an isolated environment. Below we present the ``pip`` way with "virtual environments"! This is the only recommended way at HPC2N! 

Virtual environment - venv (UPPMAX)
----

Create a ``venv``. First load the python version you want to base your virtual environment on:

.. code-block:: sh

    $ module load python/<version>
    $ python -m venv Example
    
"Example" is the name of the virtual environment. The directory “Example” is created in the present working directory.

If you want it in a certain place like "~/test/":

.. code-block:: sh

    $ python -m venv ~/test/Example 
    
Activate it.

.. prompt:: bash $

    source <path/>Example/bin/activate

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

More on virtual environment: https://docs.python.org/3/tutorial/venv.html 

Virtual environment - vpyenv (HPC2N)
----

Create a ``vpyenv``. First load the python version you want to base your virtual environment on:

.. code-block:: sh

    $ module load python/<version>
    $ virtualenv --system-site-packages vpyenv
    
"vpyenv" is the name of the virtual environment. You can name it whatever you want. The directory “vpyenv” is created in the present working directory.

**NOTE**: since it may take up a bit of space if you are installing many Python packages to your virtual environment, we **strongly** recommend you place it in your project storage! 

To place it in a directory below your project storage (again calling it "vpyenv"): 

.. code-block:: sh

   $ virtualenv --system-site-packages /proj/nobackup/<your-project-storage>/vpyenv

**NOTE** To save space, you should load any other Python modules you will need that are system installed before installing your own packages! Remember to choose ones that are compatible with the Python version you picked! 

Activate the environment.

.. code-block:: sh

    $ source <path/to/virt-environment>/vpyenv/bin/activate

Note that your prompt is changing to start with (vpyenv) to show that you are within an environment.

Install your packages with ``pip``. You should give the correct versions you want, to ensure compatibility: 

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

More on virtual environment: https://docs.python.org/3/tutorial/venv.html 

pyenv
-----

This approach is more advanced and should be, in our opinion, used only if the above are not enough for the purpose. 
This approach allows you to install your **own python version** and much more… 

Have a look on this manual https://www.uppmax.uu.se/support/user-guides/python-modules-guide/
