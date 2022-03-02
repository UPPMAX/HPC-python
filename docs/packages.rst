Packages
========

Check current available packages
--------------------------------

Check the pre-installed packages of a specific python module version
module help python/<version> 
or with python module loaded, in shell 
pip list

Very nice introduction can be found here: https://aaltoscicomp.github.io/python-for-scicomp/dependencies/ 

To make sure that the package is not already installed, type in python:
>>> import <package>
Does it work? Then it is there!
Otherwise, you can either use "pip" or "Conda".


Install with pip
----------------

You use pip this way, in Linux shell or python shell: pip install –user <package>
Use pip3 if you loaded python3

Then the package ends up in ~/.local/lib/python<version>/site-packages/ .


Install with Conda
------------------

module load condaThis grants you access to the latest version of Conda and all major repositories on all UPPMAX systems.
Check the text output as conda is loaded, see next page!
First time
export CONDA_ENVS_PATH=/a/path/to/a /place/in/your/project/
example: export CONDA_ENVS_PATH=/proj/...

The variable CONDA_ENVS_PATH contains the location of your environments. Set it to your project's environments folder if you have one.
Otherwise, the default is ~/.conda/envs. 

You may run "source conda_init.sh" to initialise your shell to be able to run "conda activate" and "conda deactivate" etc.
Just remember that this command adds stuff to your shell outside the scope of the module system.

REMEMBER TO 'conda clean -a' once in a while to remove unused and unnecessary files

Example: conda create --name python36-env python=3.6 numpy=1.13.1 matplotlib=2.2.2
Activate the Conda environment by:
	conda activate python36-env
For older version of Conda, try:
	source activate python36-env
Deactivate
	conda deactivate
More info
https://uppmax.uu.se/support/user-guides/conda-user-guide/ 


On Bianca cluster
-----------------

First try Conda, as above.
If packages are not available, follow the guideline below.
Make an installation on Rackham and then use the wharf to copy it over to your directory on Bianca
(~/.local/lib/python<version>/site-packages/ ). 
You may have to:
cp –a
… or tar/untar to include all possible symbolic links:
tar cfz <tarfile.tar.gz> <files> 	in source dirtar xfz <tarfile.tar.gz> 			in target dir
