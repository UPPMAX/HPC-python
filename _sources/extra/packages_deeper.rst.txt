More about packages
===================

Using setup.py
--------------

Some Python packages are only available as downloads, for instance via github, to install with setup.py or similar file. If that is the case for the package you need, this is how you do it: 

- Pick a location for your installation (change below to fit - I am installing under a project storage)

UPPMAX:
   - ``mkdir /proj/hpc-python/<username>/mypythonpackages``
   - ``cd /proj/hpc-python/<username>/mypythonpackages``
   
HPC2N: 
   - ``mkdir /proj/nobackup/python-hpc/<username>/mypythonpackages``
   - ``cd /proj/nobackup/python-hpc/<username>/mypythonpackages``

- Load Python + (on Kebnekaise) site-installed prerequisites and site-installed packages you need (SciPy-bundle, matplotlib, etc.)
- Install any remaining prerequisites. Remember to activate your Virtualenv if installing with pip!
- Download Python package, place it in your chosen installation dir, then untar/unzip it
- cd into the source directory of the Python package

   - Run ``python setup.py build``
   - Then install with: ``python setup.py install --prefix=<path to install dir>``
   
- Add the path to $HOME/.bash_profile (note that it will differ by Python version): 

   - ``export PYTHONPATH=$PYTHONPATH:<path to your install directory>/lib/python3.9/site-packages``
   
You can use it as normal inside Python (remember to load dependent modules as well as activate virtual environment if it depends on some packages you installed with pip): ``import <python-module>``

