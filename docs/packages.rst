Packages
========

.. Note::

    - Python packages broadens the use of python to almost infinity! 

    - Instead of writing codes yourself there may be others that has done the same!

    - Many scientific tools are distributed as python packages making it possible to run a script in the prompt and there defining files to be analysed and arguments defining exactly what to do.

    - A nice introduction to packages can be found here: https://aaltoscicomp.github.io/python-for-scicomp/dependencies/ 

.. admonition:: There are two package installation systems

    + PyPI is traditionally for Python-only packages but it is no problem to also distribute packages written in other languages as long as they provide a Python interface.

    + Conda is more general and while it contains many Python packages and packages with a Python interface, it is often used to also distribute packages which do not contain any Python (e.g. C or C++ packages).

    + Many libraries and tools are distributed in both ecosystems.


Check current available packages
--------------------------------

Some python packages are working as dstand-alone tools, for instance in bioinformatics. The tool may be already installed as a module. Check if it is there by:

.. prompt:: bash $

    module spider <tool-name or tool-name part> 
    
Using 'module spider' lets you search regardless of upper- or lowercase characters.

Check the pre-installed packages of a specific python module:

.. prompt:: bash $

    module help python/<version> 
  
or with python module loaded (more certain), in shell:

.. prompt:: bash $

    pip list

You can also test from within python to make sure that the package is not already installed:

.. prompt:: python >>>

    import <package>
    
Does it work? Then it is there!
Otherwise, you can either use "pip" or "Conda".


Install with pip
----------------

You use pip this way, in Linux shell OR python shell: 

.. prompt:: bash $

    pip install –-user <package>
    
Use pip3 if you loaded python3.

Then the package ends up in ~/.local/lib/python<version>/site-packages/ .

Install with Conda
------------------

.. Note::

    We have mirrored all major conda repositories directly on UPPMAX, on both Rackham and Bianca. These are updated every third day.
    We have the following channels available:
    
    - bioconda
    - biocore
    - conda-forge
    - dranew
    - free
    - main
    - pro
    - qiime2
    - r
    - r2018.11
    - scilifelab-lts
    
    You reach them all by loading the conda module. You don't have to state the specific channel.

1. First load our conda module (there is no need to install you own miniconda, for instance)

  .. prompt:: bash $

        module load conda
    
  - This grants you access to the latest version of Conda and all major repositories on all UPPMAX systems.

  - Check the text output as conda is loaded, especially the first time, see below:
  

2. First time

  - output when conda is loaded: 
  
    - The variable CONDA_ENVS_PATH contains the location of your environments. Set it to your project's environments folder if you have one.

    - Otherwise, the default is ~/.conda/envs. 

    - You may run "source conda_init.sh" to initialise your shell to be able to run "conda activate" and "conda deactivate" etc.

    - Just remember that this command adds stuff to your shell outside the scope of the module system.

    - REMEMBER TO 'conda clean -a' once in a while to remove unused and unnecessary files
    
 .. prompt:: bash $
      export CONDA_ENVS_PATH=/a/path/to/a /place/in/your/project-dir/
 
 - example: export CONDA_ENVS_PATH=/proj/snic2020-5-XXX
 
 - run 'conda init bash' to initialise your shell (bash) to be able to run "conda activate" and "conda deactivate" etc...

 .. prompt:: bash $

     conda init bash

3. Create the conda environment

  - Example:
  
    .. prompt:: bash $

        conda create --name python36-env python=3.6 numpy=1.13.1 matplotlib=2.2.2

4. Activate the Conda environment by:

    .. prompt:: bash $

	conda activate python36-env

    - You will see that your prompt is changing to start with (python-36-env) to show that you are within an environment.
    
    
5. Now do your work!

6. Deactivate

 .. prompt:: 
    :language: bash
    :prompts: (python-36-env) $

	conda deactivate

More info
https://uppmax.uu.se/support/user-guides/conda-user-guide/ 


On Bianca cluster
-----------------

.. Note::

    Since we have mirrored conda repositories locally Conda will work also on Bianca!


- First try Conda, as above.


- If packages are not available, follow the guideline below.


- Make an installation on Rackham and then use the wharf to copy it over to your directory on Bianca

  - (~/.local/lib/python<version>/site-packages/ ). 

- You may have to:

  - in source directory:

    .. prompt:: bash $

        cp –a
	
    - … or tar/untar to include all possible symbolic links:

      .. prompt:: bash $

        tar cfz <tarfile.tar.gz> <files> 	
	
  - and in target directory:
    
    .. prompt:: bash $

             tar xfz <tarfile.tar.gz> 		
	     

