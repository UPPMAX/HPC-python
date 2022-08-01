Isolated environments
=====================

.. note::
   Isolated environments solve a couple of problems:
   
   - You can install specific, also older, versions into them.
   - You can create one for each project and no problem if the two projects require different versions.
   - You can remove the environment and create a new one, if not needed or with errors.
   
``conda`` works as an isolated environment. Below we present the ``pip`` way with "virtual environments"! This is the only recommended way at HPC2N! 

Virtual environment - venv
----

Create a ``venv``. First load the python version you want to base your virtual environment on:

.. code-

    module load python/<version>
    python -m venv Example
    
"Example" is the name of the virtual environment. The directory “Example” is created in the present working directory.

NOTE: since it may take up a bit of space if you are installing many Python packages to your virtual environment, we recommend you place it in your project storage! 

If you want it in a certain place like "~/test/":

.. prompt:: bash $

    python -m venv ~/test/Example 

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

pyenv
-----

This approach is more advanced and should be, in our opinion, used only if the above are not enough for the purpose. 
This approach allows you to install your **own python version** and much more… 

Have a look on this manual https://www.uppmax.uu.se/support/user-guides/python-modules-guide/
