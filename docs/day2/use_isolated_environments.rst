.. _use-isolated-environments:

Use isolated environments
=========================

.. admonition:: Learning objectives

    - Practice using the documentation of your HPC cluster
    - Find out which isolated environment tool to use on your HPC cluster
    - Work (create, activate, work, deactivate) with isolated environments
      in the way recommended by your HPC cluster
    - (optional) work (create, activate, work, deactivate) with isolated environments
      in the other way (if any) possible on your HPC cluster
    - (optional) export and import a virtual
      environment

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

   - venv: uses pip       
   - virtualenv: uses pip   
   - conda/forge: uses conda/mamba     

.. warning::

   **About Conda on HPC systems**

   - Conda is good in many ways but can interact negatively when trying to use the pytrhon modules in the HPC systems.
   - LUNARC seems to have working solutions
   - At UPPMAX Conda is installed but many users that get into problems. 
	- However, on Bianca this is the most straight-forward way to install packages (no ordinary internet)


https://pixi.sh/latest/misc/FAQ/#what-is-the-difference-with-conda-mamba-poetry-pip

Virtual environment - venv & virtualenv
---------------------------------------

**FIX: shorten**

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

.. seealso::

   - UPPMAX's documentation pages about installing Python packages and virtual environments: http://docs.uppmax.uu.se/software/python/#installing-python-packages
   - HPC2N's documentation pages about installing Python packages and virtual environments: https://www.hpc2n.umu.se/resources/software/user_installed/python



.. keypoints::

   - With a virtual environment you can tailor an environment with specific versions for Python and packages, not interfering with other installed python versions and packages.
   - Make it for each project you have for reproducibility.
   - There are different tools to create virtual environments.
       - ``conda``, only recommended for personal use and at some clusters
       - ``virtualenv``, may require to load extra python bundle modules.
       - ``venv``, most straight-forward and available at all HPC centers. **Recommended**
   - More details to follow!

.. tip::

   - Try with ``venv`` first
   - If very troublesome, try with ``conda``

Conda
-----

**FIX: some intro text**

.. keypoints::

   - Conda is an installer of packages but also bigger toolkits
   - Conda creates isolated environments not clashing with other installations of python and other versions of packages
   - Conda environment requires that you install all packges needed by yourself. That is,  you cannot load the python module and use the packages therein inside you Conda environment.

+------------+---------------------------------+
| HPC cluster| Conda vs venv                   | 
+============+=================================+
| Alvis      | venv, conda in container        |
+------------+---------------------------------+
| Bianca     | conda/latest, venv via wharf    |
+------------+---------------------------------+
| COSMOS     | Anaconda3/2024.02-1             |
+------------+---------------------------------+
| Dardel     | miniconda3/24.7.1-0-cpeGNU-23.12|
+------------+---------------------------------+
| Kebnekaise | venv only                       |
+------------+---------------------------------+
| LUMI       | ?                               |
+------------+---------------------------------+
| Rackham    | venv, conda/latest              |
+------------+---------------------------------+
| Tetralith  | Anaconda3/2024.02-1             |
+------------+---------------------------------+

NSC:

- https://www.nsc.liu.se/software/python/
- https://www.nsc.liu.se/software/anaconda/

PDC:

- https://www.kth.se/blogs/pdc/2020/11/working-with-python-virtual-environments/
- https://hackmd.io/@pmitev/conda_on_Rackham

LUNARC

- https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/#anaconda-distributions

UPPMAX

- https://docs.uppmax.uu.se/software/conda/


.. admonition:: Conda in HPC

   - `Anaconda at LUNARC <https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/#anaconda-distributions>`_
   - `Conda at UPPMAX <https://docs.uppmax.uu.se/software/conda/>`_ 
      - `Conda on Bianca <https://uppmax.github.io/bianca_workshop/intermediate/install/#install-packages-principles>`_



.. admonition:: Other tools

   - pixi: package management tool for developers https://pixi.sh/latest/

   - uv: An extremely fast Python package and project manager, written in Rust. https://docs.astral.sh/uv/

Install from file/Set up course environment
-------------------------------------------

**FIX intro**


.. note::

   - All centers has had different approaches in what is included in the module system and not.
   - Therefore the solution to complete the necessary packages needed for the course lessons, different approaches has to be made.
   - This is left as exercise for you


We will need to install the LightGBM Python package for one of the examples in the ML section. 

.. tip::
    
   **Follow the track where you are working right now**


.. tabs::

   .. tab:: venv

      .. tabs::

         .. tab:: NSC

            **If you do not have matplotlib already outside any virtual environment**

            - Install matplotlib in your ``.local`` folder, not in a virtual environment.
            - Do: 

            .. code-block:: console

               ml buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 
               pip install --user matplotlib

            - Check that matplotlib is there by ``pip list``

            **Check were to find environments needed for the lessons in the afternoon tomorrow**

            - browse ``/proj/hpc-python-spring-naiss/`` to see the available environments. 
            - their names are
                - ``venvNSC-TF``
                - ``venvNSC-torch``
                - ``venvNSC-numba``
                - ``venv-spyder-only``

         .. tab:: LUNARC 

            - Everything will work by just loading modules, see each last section

            - Extra exercise can be to reproduce the examples above.

         .. tab:: UPPMAX

            **Check were to find environments needed for the lessons in the afternoon tomorrow**

            - browse ``/proj/hpc-python-uppmax/`` to see the available environments. 
            - their names are, for instance
                - ``venv-spyder``
                - ``venv-TF``
                - ``venv-torch``

            - Extra exercise can be to reproduce the examples above.

         .. tab:: HPC2N

            **Check were to find possible environments needed for the lessons in the afternoon tomorrow**

            - browse ``/proj/nobackup/hpc-python-spring/`` to see the available environments.
            - It may be empty for now but may show up by tomorrow
            - their names may be, for instance
                - ``venv-TF``
                - ``venv-torch``


   .. tab:: Conda


Own design isolated environments
--------------------------------

.. tabs::

   .. tab: venv




   .. tab: conda 





.. keypoints::

   - It is worth it to organize your code for publishing, even if only you are using it.

   - PyPI is a place for Python packages

   - conda is similar but is not limited to Python

.. note::

   - To use self-installed Python packages in a batch script, you also need to load the above mentioned modules and activate the environment. An example of this will follow later in the course. 

   - To see which Python packages you, yourself, have installed, you can use ``pip list --user`` while the environment you have installed the packages in are active. To see all packages, use ``pip list``. 





Exercises
---------

.. challenge:: Exercise 1: which system to pick

.. challenge:: Exercise 2: Isnatall package

    - Confirm package is absent
    - Create environment
    - Activate environment
    - Confirm package is absent
    - Install package in isolated environment
    - Confirm package is now present
    - Deactivate environment
    - Confirm package is now absent again



.. challenge:: (optional) Exercise 3: like 2, but for other tool

.. challenge:: (optional) Exercise 4: 

?export and import a virtual environment

Example
.......

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
       - Create: ``python -m venv /proj/hpc-python-uppmax/$USER/Example``
       - Activate: ``source /proj/hpc-python-uppmax/<user-dir>/Example/bin/activate``
   - HPC2N: 
       - Create: ``python -m venv /proj/nobackup/hpc-python-spring/$USER/Example``
       - Activate: ``source /proj/nobackup/hpc-python-spring/<user-dir>/Example/bin/activate``
   - LUNARC: 
       - Create: ``python -m venv /lunarc/nobackup/projects/lu2024-17-44/$USER/Example``
       - Activate: ``source /lunarc/nobackup/projects/lu2024-17-44/<user-dir>/Example/bin/activate``
   - NSC: 
       - Create: ``python -m venv /proj/hpc-python-spring-naiss/$USER/Example``
       - Activate: ``source /proj/hpc-python-spring-naiss/<user-dir>/Example/bin/activate``
        
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
      
    (Example) $ pip install --no-cache-dir --no-build-isolation numpy matplotlib

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

Working with virtual environments defined from files
----------------------------------------------------

.. admonition:: Python packages in HPC and ML
   :class: dropdown

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

.. admonition:: Summary of workflow

   In addition to loading Python, you will also often need to load site-installed modules for Python packages, or use own-installed Python packages. The work-flow would be something like this: 
   
 
   1. Load Python and prerequisites: ``module load <pre-reqs> Python/<version>``
   2. Load site-installed Python packages (optional): ``module load <pre-reqs> <python-package>/<version>``
   3. Create the virtual environment: ``python -m venv [PATH]/Example``
   4. Activate your virtual environment: ``source <path-to-virt-env>/Example/bin/activate``
   5. Install any extra Python packages: ``pip install --no-cache-dir --no-build-isolation <python-package>``
   6. Start Python or run python script: ``python``
   7. Do your work
   8. Deactivate

   - Installed Python modules (modules and own-installed) can be accessed within Python with ``import <package>`` as usual. 
   - The command ``pip list`` given within Python will list the available modules to import. 
   - More about packages and virtual/isolated environment to follow in later sections of the course! 


Exercises
---------

.. challenge:: 1. Make a test environment

   1. make a virtual environment with the name ``venv1``. Do not include packages from the the loaded module(s)
   2. activate
   3. install ``matplotlib``
   4. make a requirements file of the content
   5. deactivate
   6. make another virtual environment with the name ``venv2``
   7. activate that
   8. install with the aid of the requirements file
   9. check the content
   10. open python shell from command line and try to import
   11. exit python
   12. deactivate
   
.. solution:: Solution 
   :class: dropdown
    
   - First load the required Python module(s) if not already done so in earlier lessons. Remember that this steps differ between the HPC centers

   1. make the first environment

   .. code-block:: console

      $ python -m venv venv1
    
   2. Activate it.

   .. code-block:: console

      $ source venv1/bin/activate

      - Note that your prompt is changing to start with ``(venv1)`` to show that you are within an environment.
   
   3. install ``matplotlib``

   .. code-block:: console

      pip install matplotlib

   4. make a requirements file of the content

   .. code-block:: console

      pip freeze --local > requirements.txt

   5. deactivate

   .. code-block:: console

      deactivate

   6. make another virtual environment with the name ``venv2``

   .. code-block:: console

      python -m venv venv2

   7. activate that

   .. code-block:: console

      source venv2/bin/activate

   8. install with the aid of the requirements file

   .. code-block:: console

      pip install -r requirements.txt

   9. check the content

   .. code-block:: console

      pip list

   10. open python shell from command line and try to import

   .. code-block:: console

      python

   .. code-block:: python

      import matplotlib

   11. exit python

   .. code-block:: python

      exit()
      
   12. deactivate

   .. code-block:: console

      deactivate

      

.. seealso::

   - want to share your work? :ref:`devel_iso`

