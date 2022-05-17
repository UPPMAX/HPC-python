Packages
========

- Python **packages broaden the use of python** to almost infinity! 

- Instead of writing code yourself there may be others that have done the same!

- Many **scientific tools** are distributed as **python package**s making it possible to run a script in the prompt and there defining files to be analysed and arguments defining exactly what to do.

- A nice **introduction to packages** can be found here: https://aaltoscicomp.github.io/python-for-scicomp/dependencies/ 

.. admonition:: There are two package installation systems

    - **PyPI** (``pip``) is traditionally for Python-only packages but it is no problem to also distribute packages written in other languages as long as they provide a Python interface.

    - **Conda** (``conda``) is more general and while it contains many Python packages and packages with a Python interface, it is often used to also distribute packages which do not contain any Python (e.g. C or C++ packages).
    	- Creates its own environment that does not interact with other python installations

    - Many libraries and tools are distributed in both ecosystems.


Check current available packages
--------------------------------

Some python packages are working as stand-alone tools, for instance in bioinformatics. The tool may be already installed as a module. Check if it is there by:

.. prompt:: bash $

    module spider <tool-name or tool-name part> 
    
Using ``module spider`` lets you search regardless of upper- or lowercase characters.

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
Otherwise, you can either use ``pip`` or ``conda``.


Install with pip
----------------

You use ``pip`` this way, in Linux shell OR python shell: 

.. prompt:: bash $

    pip install –-user <package>
    
Use ``pip3`` if you loaded python3.

Then the package ends up in ~/.local/lib/python<version>/site-packages/ .

Install with conda
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

  - Check the text output as conda is loaded, especially the first time, see below
  
   .. admonition:: Conda load output
       :class: dropdown

       - The variable CONDA_ENVS_PATH contains the location of your environments. Set it to your project's environments folder if you have one.

       - Otherwise, the default is ~/.conda/envs. 

       - You may run ``source conda_init.sh`` to initialise your shell to be able to run ``conda activate`` and ``conda deactivate`` etc.

       - Just remember that this command adds stuff to your shell outside the scope of the module system.

       - REMEMBER TO ``conda clean -a`` once in a while to remove unused and unnecessary files


2. First time
        
  - The variable CONDA_ENVS_PATH contains the location of your environments. Set it to your project's environments folder if you have one.
  - Otherwise, the default is ~/.conda/envs. 
  - Example:
  
      .. prompt:: bash $
 
          export CONDA_ENVS_PATH=/proj/snic2020-5-XXX
  
   .. admonition:: By choice
      :class: dropdown
 
      Run ``source conda_init.sh`` to initialise your shell (bash) to be able to run ``conda activate`` and ``conda deactivate`` etcetera instead of ``source activate``. It will modify (append) your ``.bashrc`` file.


3. Create the conda environment

  - Example:
  
    .. prompt:: bash $

        conda create --name python36-env python=3.6 numpy=1.13.1 matplotlib=2.2.2
	
    .. admonition:: The ``mamba`` alternative 
        :class: dropdown
    
	- ``mamba`` is a fast drop-in alternative to conda, using "libsolv" for dependency resolution. It is available from the ``conda`` module.
	- Example:  
	
          .. prompt:: bash $

	      mamba create --name python37-env python=3.7 numpy=1.13.1 matplotlib=2.2.2

4. Activate the conda environment by:

    .. prompt:: bash $

	source activate python36-env

    - You will see that your prompt is changing to start with ``(python-36-env)`` to show that you are within an environment.
    
5. Now do your work!

6. Deactivate

 .. prompt:: 
    :language: bash
    :prompts: (python-36-env) $
    
    conda deactivate

.. warning::
 
    - Conda is known to create **many** *small* files. Your diskspace is not only limited in GB, but also in number of files (typically ``300000`` in $home). 
    - Check your disk usage and quota limit with ``uquota``
    - Do a ``conda clean -a`` once in a while to remove unused and unnecessary files
    
    
More info
https://uppmax.uu.se/support/user-guides/conda-user-guide/ 



