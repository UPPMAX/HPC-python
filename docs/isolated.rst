Isolated environments
=====================

.. note::
   Isolated environments solve a couple of problems:
   You can install specific, also older, versions into them.
   You can create one for each project and no problem if the two projects require different versions.
   You can remove the environment and create a new one, if not needed or with errors.

venv
----

Example with virtual environment
Create a "venv". First load the python version you want to base your virtual environment on.
module load python/x.y.z
python –m venv Example

Example is the name of the virtual environment.
It creates the directory “Example”

Activate it.
	source Example/bin/activate

Install your packages with pip and the correct versions, like:
pip install numpy==1.13.1 matplotlib==2.2.2

Deactivate it.
	deactivate

Everytime you need the tools available in the virtual environment you activate it as above.
More on virtual environment: https://docs.python.org/3/tutorial/venv.html 

pyenv
-----

This approach allows you to install your own python version and much more… have a look on this manual https://www.uppmax.uu.se/support/user-guides/python-modules-guide/
