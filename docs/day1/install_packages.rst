Install packages
================

.. objectives:: 

   Learners can 

   - work (create, activate, work, deactivate) with virtual environments
   - install a python package
   - export and import a virtual environment

Introduction
------------

There are 2-3 ways to install missing python packages at a HPC cluster.

- Local installation, always available for the version of Python you had active when doing the installation
    - ``pip install --user [package name]``
- Isolated environment. Use some packages just needed for a specific use case.
    - ``venv``/``virtualenv`` in combination with ``pip`` 
        - recommended in all HPC centers in Sweden
    - ``conda``
        - just recommended in some HPC centers in Sweden


Local (general installation)
............................

.. note::

   ``pip install --user [package name]`` 

    - The package end up in ``~/.local``
    - target directory can be changed by ``--prefix=[root_folder of installation]``

Isolated environments
.....................

As an example, maybe you have been using TensorFlow 1.x.x for your project and now you need to install a package that requires TensorFlow 2.x.x but you will still be needing the old version of TensorFlow for another package, for instance. This is easily solved with isolated environments.

.. note::
  
   Isolated/virtual environments solve a couple of problems:
   
   - You can install specific, also older, versions into them.
   - You can create one for each project and no problem if the two projects require different versions.
   - You can remove the environment and create a new one, if not needed or with errors.

- Isolated environments lets you create separate workspaces for different versions of Python and/or different versions of packages. 
- You can activate and deactivate them one at a time, and work as if the other workspace does not exist.

**The tools**

   - venv            UPPMAX+HPC2N+LUNARC+NSC
   - virtualenv      UPPMAX+HPC2N+LUNARC+NSC
   - Conda           LUNARC + UPPMAX (recommended only for Bianca cluster)

.. warning::

   **About Conda on HPC systems**

   - Conda is good in many ways but can interact negatively when trying to use the pytrhon modules in the HPC systems.
   - LUNARC seems to have working solutions
   - At UPPMAX Conda is installed but we have many users that get into problems. 
	- However, on Bianca this is the most straight-forward way to install packages (no ordinary internet)

.. admonition:: Conda in HPC

   - `Anaconda at LUNARC <https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/#anaconda-distributions>`_
   - `Conda at UPPMAX <https://docs.uppmax.uu.se/software/conda/>`_ 
      - `Conda on Bianca <https://uppmax.github.io/bianca_workshop/intermediate/install/#install-packages-principles>`_

Virtual environment - venv & virtualenv
---------------------------------------

.. admonition:: Workflow

1. You load the Python module you will be using, as well as any site-installed package modules (requires the ``--system-site-packages`` option later)
2. You create the isolated environment with something like venv, virtualenv (use the ``--system-site-packages`` to include all "non-base" packages)
3. You activate the environment
4. You install (or update) the environment with the packages you need
5. You work in the isolated environment
6. You deactivate the environment after use 

.. admonition:: venv vs. virtualenv

   - These are almost completely interchangeable
   - The difference being that **virtualenv supports older python versions** and has a few more minor unique features, while **venv is in the standard library**.
   - Step 1:
       - Virtualenv: ``virtualenv --system-site-packages Example``
       - venv: ``python -m venv --system-site-packages Example2``
   - Next steps are identical and involves "activating" and ``pip installs``
   - We recommend ``venv`` in the course. Then we are just needing the Python module itself!

.. keypoints::

   - With a virtual environment you can tailor an environment with specific versions for Python and packages, not interfering with other installed python versions and packages.
   - Make it for each project you have for reproducibility.
   - There are different tools to create virtual environments.
       - ``conda``, only recommended for personal use and at some clusters
       - ``virtualenv``, may require to load extra python bundle modules.
       - **``venv``**, most straight-forward and available at all HPC centers. **Recommended**
   - More details to follow!

Example
#######

.. tip::
    
   **Do not type along!**

Create a ``venv``. First load the python version you want to base your virtual environment on:

.. tabs::

   .. tab:: UPPMAX

      .. code-block:: console

         $ module load python/3.11.8 
         $ python -m venv --system-site-packages Example2
    
     "Example2" is the name of the virtual environment. The directory "Example2" is created in the present working directory. The ``-m`` flag makes sure that you use the libraries from the python version you are using.

   .. tab:: HPC2N

      .. code-block:: console

         $ module load GCC/12.3.0 Python/3.11.3
         $ python -m venv --system-site-packages Example2

      "Example2" is the name of the virtual environment. You can name it whatever you want. The directory “Example2” is created in the present working directory.

   .. tab:: LUNARC 

      .. code-block:: console

         $ module load GCC/12.3.0 Python/3.11.3
         $ python -m venv --system-site-packages Example2

      "Example2" is the name of the virtual environment. You can name it whatever you want. The directory “Example2” is created in the present working directory.

   .. tab:: NSC 

      .. code-block:: console

         $ ml buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5
         $ python -m venv --system-site-packages Example2

      "Example2" is the name of the virtual environment. You can name it whatever you want. The directory “Example2” is created in the present working directory.
      
.. note::

   To save space, you should load any other Python modules you will need that are system installed before installing your own packages! Remember to choose ones that are compatible with the Python version you picked! 
   ``--system-site-packages`` includes the packages already installed in the loaded python module.

   At HPC2N, NSC and LUNARC, you often have to load SciPy-bundle. This is how you could create a venv (Example3) with a SciPy-bundle included which is compatible with Python/3.11.3:
   
   .. code-block:: console

         $ module load GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 # for HPC2N and LUNAR
         $ module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 # for NSC
         $ python -m venv --system-site-packages Example3


**NOTE**: since it may take up a bit of space if you are installing many Python packages to your virtual environment, we **strongly** recommend you place it in your project storage! 

**NOTE**: if you need to for instance working with both Python 2 and 3, then you can of course create more than one virtual environment, just name them so you can easily remember which one has what. 
      
.. admonition:: If you want your virtual environment in a certain place...

   - Example for course project location and ``$USER`` being you user name. 
       - If your directory in the project has another name, replace ``$USER`` with that one!
   
   - UPPMAX: 
       - Create: ``python -m venv --system-site-packages /proj/hpc-python-fall/$USER/Example``
       - Activate: ``source /proj/hpc-python-fall/<user-dir>/Example/bin/activate``
   - HPC2N: 
       - Create: ``$ python -m venv --system-site-packages /proj/nobackup/hpc-python-fall-hpc2n/$USER/Example``
       - Activate: ``source /proj/nobackup/hpc-python-fall-hpc2n/<user-dir>/Example/bin/activate``
   - LUNARC: 
       - Create: ``$ python -m venv --system-site-packages /lunarc/nobackup/projects/lu2024-17-44/$USER/Example``
       - Activate: ``source /lunarc/nobackup/projects/lu2024-17-44/<user-dir>/Example/bin/activate``
   - NSC: 
       - Create: ``$ python -m venv --system-site-packages /proj/hpc-python-fall-nsc/$USER/Example``
       - Activate: ``source /proj/hpc-python-fall-nsc/<user-dir>/Example/bin/activate``
        
   Note that your prompt is changing to start with (Example) to show that you are within an environment.

.. note::

   - ``source`` can most often be replaced by ``.``, like in ``. Example/bin/activate``. Note the important <space> after ``.``
   - For clarity we use the ``source`` style here.


Install packages to the virtual environment with pip
....................................................

.. tip::

   **Do not type along!**
   
Install your packages with ``pip``. While not always needed, it is often a good idea to give the correct versions you want, to ensure compatibility with other packages you use. This example assumes your venv is activated: 

.. code-block:: console
      
    (Example) $ pip install --no-cache-dir --no-build-isolation numpy==1.20.2 matplotlib==3.2.2

The ``--no-cache-dir"`` option is required to avoid it from reusing earlier installations from the same user in a different environment. The ``--no-build-isolation`` is to make sure that it uses the loaded modules from the module system when building any Cython libraries.

Deactivate the venv.

.. code-block:: console
      
    (Example) $ deactivate
    


Everytime you need the tools available in the virtual environment you activate it as above (after also loading the modules).

.. prompt:: console

   source /proj/<your-project-id>/<your-dir>/Example/bin/activate
   
   
.. note::

   - You can use "pip list" on the command line (after loading the python module) to see which packages are available and which versions. 
   - Some packaegs may be inhereted from the moduels yopu have loaded
   - You can do ``pip list --local`` to see what is instaleld by you in the environment.
   - Some IDE:s like Spyder may only find those "local" packages

Prepare the course environment
------------------------------

We will need to install the LightGBM Python package for one of the examples in the ML section. 

.. tip::
    
   **Type along!**

Create a virtual environment called ``vpyenv``. First load the python version you want to base your virtual environment on, as well as the site-installed ML packages. 

.. tabs::

   .. tab:: UPPMAX
      
      .. code-block:: console

          $ module load uppmax 
          $ module load python/3.11.8
	  $ module load python_ML_packages/3.11.8-cpu
	  $ python -m venv --system-site-packages /proj/hpc-python/<user-dir>/vpyenv
    
      Activate it.

      .. code-block:: console

         $ source /proj/hpc-python/<user-dir>/vpyenv/bin/activate

      Note that your prompt is changing to start with (vpyenv) to show that you are within an environment.

      Install your packages with ``pip`` (``--user`` not needed as you are in your virtual environment) and (optionally) giving the correct versions, like:

      .. code-block:: console
      
         (vpyenv) $ pip install --no-cache-dir --no-build-isolation scikit-build-core cmake 
         (vpyenv) $ pip install --no-cache-dir --no-build-isolation lightgbm

      The reason for the other packages (``scikit-build-core`` and ``cmake``) that are being installed first, is that they are prerequisites for ``lightgbm``. 	 

      Check what was installed

      .. code-block:: console
      
         (vpyenv) $ pip list

      Deactivate it.

      .. code-block:: console
      
         (vpyenv) $ deactivate

      Everytime you need the tools available in the virtual environment you activate it as above, after loading the python module.

      .. code-block:: console 

         $ source /proj/hpc-python/<user-dir>/vpyenv/bin/activate

      More on virtual environment: https://docs.python.org/3/tutorial/venv.html 

   .. tab:: HPC2N
     
      **First go to the directory you want your environment in.**

      Load modules for Python, SciPy-bundle, matplotlib, create the virtual environment, activate the environment, and install lightgbm and scikit-learn (since the versions available are not compatible with this Python) on Kebnekaise at HPC2N 
   
      .. code-block:: console
           
         $ module load GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2
         $ python -m venv --system-site-packages vpyenv
         $ source vpyenv/bin/activate
         (vpyenv) $ pip install --no-cache-dir --no-build-isolation lightgbm scikit-learn 
   
      Deactivating a virtual environment.

      .. code-block:: console

         (vpyenv) $ deactivate

      Every time you need the tools available in the virtual environment you activate it as above (after first loading the modules for Python, Python packages, and prerequisites)

      .. code-block:: console

         $ source vpyenv/bin/activate
    

Using the self-installed packages in Python
###########################################

- To use the Python packages you have installed under your virtual environment, load your Python module + prerequisites, load any site-installed Python packages you used, and then activate the environment.
- Now your own packages can be accessed from within Python, just like any other Python package. 

**Test it!**

.. tip::
    
   **Type along!**


Using the virtual environment created under "Preparing the course environment" and the ``lightgbm`` we installed there. 

.. admonition:: UPPMAX
   :class: dropdown
   
   Load modules for python, and python_ML_packages, then activate the environment and start python. Try and import the library ``lightgbm``. 

   .. code-block:: console
         
      $ module load uppmax python/3.11.8 python_ML_packages/3.11.8-cpu
      $ source /proj/hpc-python/<user-dir>/vpyenv/bin/activate
      (vpyenv) $ python
      Python 3.11.8 (main, Feb  8 2024, 11:48:52) [GCC 12.3.0] on linux
      Type "help", "copyright", "credits" or "license" for more information.
      >>> import lightgbm 
      >>> 

.. admonition:: HPC2N
    :class: dropdown

    Load modules for Python, SciPy-bundle, matplotlib. Then activate the environment created under "Preparing the course environment" and import the library ``lightgbm`` we installed. 

    .. code-block:: console
         
       $ module load GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2
       $ source /proj/nobackup/python-hpc/<user-dir>/vpyenv/bin/activate
       (vpyenv)$ python
       Python 3.11.3 (main, Apr  2 2024, 14:00:42) [GCC 12.3.0] on linux
       Type "help", "copyright", "credits" or "license" for more information.
       >>> import lightgbm 
       >>> 
 

- To use self-installed Python packages in a batch script, you also need to load the above mentioned modules and activate the environment. An example of this will follow later in the course. 

- To see which Python packages you, yourself, have installed, you can use ``pip list --user`` while the environment you have installed the packages in are active. To see all packages, use ``pip list``. 


Working with virtual environments defined from files
----------------------------------------------------

- First create and activate an environment (see above)
- Create an environment based on dependencies given in an environment file::
  
.. code-block:: console

   $ pip install -r requirements.txt
   
- Create file from present virtual environment::

.. code-block:: console

   $ pip freeze > requirements.txt
  
- That includes also the *system site packages* if you included them with ``--system-site-packages``
- You can list packages specific for the virtualenv by ``pip list --local`` 

- So, creating a file from just the local environment::

.. code-block:: console

   $ pip freeze --local > requirements.txt

``requirements.txt`` (used by the virtual environment) is a simple text file which looks similar to this::

   numpy
   matplotlib
   pandas
   scipy

``requirements.txt`` with versions that could look like this::

    numpy==1.20.2
    matplotlib==3.2.2
    pandas==1.1.2
    scipy==1.6.2

.. admonition:: More on dependencies

   - `Dependency management from course Python for Scientific computing <https://aaltoscicomp.github.io/python-for-scicomp/dependencies/>`_


.. note:: 

   **pyenv**

   - This approach is more advanced and should, in our opinion, be used only if the above are not enough for the purpose. 
   - ``pyenv`` allows you to install your **own python version**, like 3.10.2, and much more… 
   - `Pyenv at UPPMAX <http://docs.uppmax.uu.se/software/python_pyenv/>`_
   - Probably Conda will work well for you anyway...

Jupyter in a virtual environment
--------------------------------

.. warning:: 

   **Running Jupyter in a virtual environment**

   You could also use ``jupyter`` (``-lab`` or ``-notebook``) in a virtual environment.

   **UPPMAX**: 

   If you decide to use the --system-site-packages configuration you will get ``jupyter`` from the python module you created your virtual environment with.
   However, you **won't find your locally installed packages** from that jupyter session. To solve this reinstall jupyter within the virtual environment by force::

      $ pip install -I jupyter

   - This overwrites the first version as "seen" by the environment.
   - Then run::

      $ jupyter-notebook
   
   Be sure to start the **kernel with the virtual environment name**, like "Example", and not "Python 3 (ipykernel)".

   **HPC2N**

   To use Jupyter at HPC2N, follow this guide: https://www.hpc2n.umu.se/resources/software/jupyter
   To use it with extra packages, follow this guide after setting it up as in the above guide: https://www.hpc2n.umu.se/resources/software/jupyter-python


Python packages in HPC and ML
-----------------------------

It is difficult to give an exhaustive list of useful packages for Python in HPC, but this list contains some of the more popular ones: 

.. list-table:: Popular packages
   :widths: 8 10 10 20 
   :header-rows: 1

   * - Package
     - Module to load, UPPMAX
     - Module to load, HPC2N
     - Brief description 
   * - Dask
     - ``python``
     - ``dask``
     - An open-source Python library for parallel computing.
   * - Keras
     - ``python_ML_packages``
     - ``Keras``
     - An open-source library that provides a Python interface for artificial neural networks. Keras acts as an interface for both the TensorFlow and the Theano libraries. 
   * - Matplotlib
     - ``python`` or ``matplotlib``
     - ``matplotlib``
     - A plotting library for the Python programming language and its numerical mathematics extension NumPy.
   * - Mpi4Py
     - Not installed
     - ``SciPy-bundle``
     - MPI for Python package. The library provides Python bindings for the Message Passing Interface (MPI) standard.
   * - Numba 
     - ``python``
     - ``numba``
     - An Open Source NumPy-aware JIT optimizing compiler for Python. It translates a subset of Python and NumPy into fast machine code using LLVM. It offers a range of options for parallelising Python code for CPUs and GPUs. 
   * - NumPy
     - ``python``
     - ``SciPy-bundle``
     - A library that adds support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. 
   * - Pandas
     - ``python`` 
     - ``SciPy-bundle``
     - Built on top of NumPy. Responsible for preparing high-level data sets for machine learning and training. 
   * - PyTorch/Torch
     - ``PyTorch`` or ``python_ML_packages``
     - ``PyTorch``
     - PyTorch is an ML library based on the C programming language framework, Torch. Mainly used for natural language processing or computer vision.  
   * - SciPy
     - ``python``
     - ``SciPy-bundle``
     - Open-source library for data science. Extensively used for scientific and technical computations, because it extends NumPy (data manipulation, visualization, image processing, differential equations solver).  
   * - Seaborn 
     - ``python``
     - Not installed
     - Based on Matplotlib, but features Pandas’ data structures. Often used in ML because it can generate plots of learning data. 
   * - Sklearn/SciKit-Learn
     - ``scikit-learn``
     - ``scikit-learn``
     - Built on NumPy and SciPy. Supports most of the classic supervised and unsupervised learning algorithms, and it can also be used for data mining, modeling, and analysis. 
   * - StarPU
     - Not installed 
     - ``StarPU``
     - A task programming library for hybrid architectures. C/C++/Fortran/Python API, or OpenMP pragmas. 
   * - TensorFlow
     - ``TensorFlow``
     - ``TensorFlow``
     - Used in both DL and ML. Specializes in differentiable programming, meaning it can automatically compute a function’s derivatives within high-level language. 
   * - Theano 
     - Not installed 
     - ``Theano``
     - For numerical computation designed for DL and ML applications. It allows users to define, optimise, and gauge mathematical expressions, which includes multi-dimensional arrays.  

Remember, in order to find out how to load one of the modules, which prerequisites needs to be loaded, as well as which versions are available, use ``module spider <module>`` and ``module spider <module>/<version>``. 

Often, you also need to load a python module, except in the cases where it is included in ``python`` or ``python_ML_packages`` at UPPMAX or with ``SciPy-bundle`` at HPC2N. 

NOTE that not all versions of Python will have all the above packages installed! 

More info
---------

- UPPMAX's documentation pages about installing Python packages and virtual environments: http://docs.uppmax.uu.se/software/python/#installing-python-packages
- HPC2N's documentation pages about installing Python packages and virtual environments: https://www.hpc2n.umu.se/resources/software/user_installed/python




.. admonition:: Summary of workflow

   In addition to loading Python, you will also often need to load site-installed modules for Python packages, or use own-installed Python packages. The work-flow would be something like this: 
   
 
   1) Load Python and prerequisites: `module load <pre-reqs> Python/<version>``
   2) Load site-installed Python packages (optional): ``module load <pre-reqs> <python-package>/<version>``
   3) Activate your virtual environment (optional): ``source <path-to-virt-env>/bin/activate``
   4) Install any extra Python packages (optional): ``pip install --no-cache-dir --no-build-isolation <python-package>``
   5) Start Python or run python script: ``python``
   6) Do your work
   7) Deactivate

   - Installed Python modules (modules and own-installed) can be accessed within Python with ``import <package>`` as usual. 
   - The command ``pip list`` given within Python will list the available modules to import. 
   - More about packages and virtual/isolated environment to follow in later sections of the course! 

.. challenge:: Create a virtual environment with a requirements file below

   - Create a virtual environment with python-3.9.5 (UPPMAX) and Python/3.8.6 (HPC2N) with the name ``analysis``.
   - Install packages definde by this ``requirements.txt`` file (save it).
  
   .. code-block:: console
   
      numpy==1.20.2
      matplotlib==3.2.2
      pandas==1.2.0
    
   - Check that the packages were installed.
   - Don't forget to deactivate afterwards.

.. solution:: Solution for UPPMAX
   :class: dropdown
    
   .. code-block:: console

      $ module load python/3.9.5
      $ python -m venv --system-site-packages /proj/naiss2023-22-1126/<user-dir>/analysis
    
   Activate it.

   .. code-block:: console

      $ source /proj/naiss2023-22-1126/<user-dir>/analysis/bin/activate

   - Note that your prompt is changing to start with (analysis) to show that you are within an environment.
   - Install the packages from the file and then check if the right packages were installed::
      
        pip install -r requirements.txt
      
   .. code-block:: console

         $ pip list
	 $ deactivate
      
.. solution:: Solution for HPC2N
   :class: dropdown
    
   .. code-block:: console

      $ module load GCC/10.2.0 Python/3.8.6 
      $ python -m venv --system-site-packages /proj/nobackup/<your-proj-dir>/analysis 
      
   Activate it.

   .. code-block:: console

      $ source /proj/nobackup/<your-proj-dir>/analysis/bin/activate

   - Note that your prompt is changing to start with (analysis) to show that you are within an environment.
   - Install the packages from the file and then check if the right packages were installed::
      
        pip install -r requirements.txt
      
   .. code-block:: console

      $ pip list
      $ deactivate
      


.. keypoints::

   - With a virtual environment you can tailor an environment with specific versions for Python and packages, not interfering with other installed python versions and packages.
   - Make it for each project you have for reproducibility.
   - There are different tools to create virtual environemnts.
   
      - UPPMAX has ``conda`` and ``venv`` and ``virtualenv``
      - HPC2N has ``venv`` and ``virtualenv``
