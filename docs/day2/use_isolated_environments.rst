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
---------------------

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

   - Conda is good in many ways but can interact negatively when trying to use the python modules (module load).
   - LUNARC seems to have working solutions
   - At UPPMAX Conda is installed but many users that get into problems. 
       - However, on Bianca this is the most straight-forward way to install packages (no ordinary internet)

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
| LUMI       | conda-containerize              |
+------------+---------------------------------+

.. tip::

   - Try with ``venv`` first
   - If very troublesome, try with ``conda``

.. admonition:: Other tools perhaps covered in the future

   - `pixi <https://pixi.sh/latest/>`_: package management tool for developers 
       - It allows the developer to install libraries and applications in a reproducible way. Use pixi cross-platform, on Windows, Mac and Linux.
       - could replace conda/mamba

   - `uv <https://docs.astral.sh/uv/>`_: An extremely fast Python package and project manager, written in Rust. 
       - A single tool to replace pip, pip-tools, pipx, poetry, pyenv, twine, virtualenv, and more

Virtual environment - venv & virtualenv
---------------------------------------

With this tool you can download and install with ``pip`` from the `PyPI repository <https://pypi.org/>`_

Typical workflow
................

   1. You load the Python module you will be using, as well as any site-installed package modules (requires the ``--system-site-packages`` option later)
   2. You create the isolated environment with something like ``venv``, ``virtualenv`` (use the ``--system-site-packages`` to include all "non-base" packages)
   3. You activate the environment with ``source <path to virtual environment>/bin activate``
   4. You install (or update) the environment with the packages you need with the ``pip`` command
   5. You work in the isolated environment
   6. You deactivate the environment after use with ``deactivate``

.. admonition:: venv vs. virtualenv
   :class: dropdown   

   - These are almost completely interchangeable
   - The difference being that **virtualenv supports older python versions** and has a few more minor unique features, while **venv is in the standard library**.
   - Step 1:
       - Virtualenv: ``virtualenv --system-site-packages Example``
       - venv: ``python -m venv --system-site-packages Example2``
   - Next steps are identical and involves "activating" and ``pip installs``
   - We recommend ``venv`` in the course. Then we are just needing the Python module itself!

.. admonition:: Draw-backs

   - Only works for python environments
   - Only works with python versions already installed


Conda
-----

- Conda is an installer of packages but also bigger toolkits and is useful also for R packages and C/C++ installations.
- Conda creates isolated environments not clashing with other installations of python and other versions of packages.
- Conda environment requires that you install all packges needed by yourself. That is,  you cannot load the python module and use the packages therein inside you Conda environment.

.. warning::
 
    - Conda is known to create **many** *small* files. Your diskspace is not only limited in GB, but also in number of files (typically ``300000`` in $home). 
    - Check your disk usage and quota limit with ``uquota`` or **FIX**, depending on system
    - Do a ``conda clean -a`` once in a while to remove unused and unnecessary files

    


Typical workflow
................

The first 2 steps are cluster dependent and will therefore be slightly different.

1. Make conda available from a software module, like ``ml load conda`` or similar, or use own installation of miniconda or miniforge.
2. First time

   .. admonition:: First time
      :class: dropdown   

      - The variable CONDA_ENVS_PATH contains the location of your environments. Set it to your project's environments folder, if you have one, instead of the HOME folder.
      - Otherwise, the default is ~/.conda/envs. 
      - Example:
  
      .. code-block:: console
 
         $ export CONDA_ENVS_PATH=/proj/<your-project-id>/nobackup/<username>
  
      .. admonition:: By choice
         :class: dropdown
 
      Run ``source conda_init.sh`` to initialise your shell (bash) to be able to run ``conda activate`` and ``conda deactivate`` etcetera instead of ``source activate``. It will modify (append) your ``.bashrc`` file.
      
3. Create the conda environment
4. Activate the conda environment by: source activate <conda-env-name>
5. Now do your work!
6. Deactivate

 .. prompt:: 
    :language: bash
    :prompts: (python-36-env) $
    
    conda deactivate

.. admonition:: Conda base env

   - When conda is loaded you will by default be in the base environment, which works in the same way as other conda environments. It includes a Python installation and some core system libraries and dependencies of Conda. It is a “best practice” to avoid installing additional packages into your base software environment.

.. admonition:: Conda cheat sheet    
   
   - List packages in present environment:	``conda list``
   - List all environments:			``conda info -e`` or ``conda env list``
   - Install a package: ``conda install somepackage``
   - Install from certain channel (conda-forge): ``conda install -c conda-forge somepackage``
   - Install a specific version: ``conda install somepackage=1.2.3``
   - Create a new environment: ``conda create --name myenvironment``
   - Create a new environment from requirements.txt: ``conda create --name myenvironment --file requirements.txt``
   - On e.g. HPC systems where you don’t have write access to central installation directory: conda create --prefix /some/path/to/env``
   - Activate a specific environment: ``conda activate myenvironment``
   - Deactivate current environment: ``conda deactivate``

.. admonition:: Conda vs mamba etc...

   - `what-is-the-difference-with-conda-mamba-poetry-pip <https://pixi.sh/latest/misc/FAQ/#what-is-the-difference-with-conda-mamba-poetry-pip>`_

.. warning::

   - If you experience unexpected problems with the conda provided by the module system on Rackham or anaconda3 on Dardel, you can easily install your own and maintain it yourself.
   - Read more at `Pavlin Mitev's page about conda on Rackham/Dardel <https://hackmd.io/@pmitev/conda_on_Rackham>`_ and change paths to relevant one for your system.
   - Or `Conda - "best practices" - UPPMAX <https://hackmd.io/@pmitev/module_conda_Rackham>`_

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

         .. tab:: PDC 

            .. code-block:: console

               $ module load PDC/21.11
               $ module load Anaconda3/2021.05
               $ cd /cfs/klemming/home/u/username
               $ python3 -m venv my-venv-dardel

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

      .. tabs::

         .. tab:: NSC

            - content

         .. tab:: PDC

            - content

         .. tab:: LUNARC

            - content

         .. tab:: UPPMAX: Rackham

            - content

         .. tab:: UPPMAX: Bianca

            - content

Own design isolated environments
--------------------------------

.. tabs::

   .. tab:: venv

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

         .. tab:: PDC 

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
         - PDC: 
             - Create: ``python -m venv /cfs/klemming/projects/snic/hpc-python-spring-naiss/$USER/Example``
             - Activate: ``source /cfs/klemming/projects/snic/hpc-python-spring-naiss/$USER/Example/bin/activate``

         Note that your prompt is changing to start with (Example) to show that you are within an environment.

      .. note::

         - ``source`` can most often be replaced by ``.``, like in ``. Example/bin/activate``. Note the important <space> after ``.``
         - For clarity we use the ``source`` style here.

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


   .. tab:: conda 





.. keypoints::

   - It is worth it to organize your code for publishing, even if only you are using it.

   - PyPI is a place for Python packages

   - conda is similar but is not limited to Python

.. note::

   - To use self-installed Python packages in a batch script, you also need to load the above mentioned modules and activate the environment. An example of this will follow later in the course. 

   - To see which Python packages you, yourself, have installed, you can use ``pip list --user`` while the environment you have installed the packages in are active. To see all packages, use ``pip list``. 



Exercises
---------

.. challenge:: Exercise 0: Make a decision between venv or conda.


Breakout room according to grouping

.. challenge:: Exercise 1: Cover the documentation

   First try to find it by navigating.

   - NSC: https://www.nsc.liu.se
   - PDC: https://support.pdc.kth.se/doc/
   - LUNARC: https://lunarc-documentation.readthedocs.io/en/latest/
   - UPPMAX: https://docs.uppmax.uu.se/
   - HPC2N: https://docs.hpc2n.umu.se/
   - LUMI: https://docs.lumi-supercomputer.eu/software

   .. solution::

      **FIX add links to venvs**
      **FIX conda tab and venv tab?? Or make 1a and 1b**

      NSC:

      - https://www.nsc.liu.se/software/python/
      - https://www.nsc.liu.se/software/anaconda/

      PDC:

      - https://www.kth.se/blogs/pdc/2020/11/working-with-python-virtual-environments/

      LUNARC

      - https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/#anaconda-distributions

      UPPMAX

      - https://docs.uppmax.uu.se/software/conda/

      LUMI

      - https://docs.lumi-supercomputer.eu/software/installing/container-wrapper/#examples-of-using-the-lumi-container-wrapper

.. challenge:: Exercise 2: Prepare the course environment

   - venv or conda
   - solution in drop-down

.. challenge:: Exercise 3a: Install package (venv)

   - Coose a package of the ones below:

   **Prepare** list

    - Confirm package is absent
    - Create environment
    - Activate environment
    - Confirm package is absent
    - Install package in isolated environment
    - Confirm package is now present
    - Deactivate environment
    - Confirm package is now absent again

.. challenge:: (optional) 4a. Make a test environment (venv)

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

.. challenge:: 3b. Make a test environment (conda)

.. challenge:: (optional) Exercise 4: like 3, but for other tool


.. admonition:: Already installed Python packages in HPC and ML
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



Summary
.......

.. keypoints::

   - With a virtual environment you can tailor an environment with specific versions for Python and packages, not interfering with other installed python versions and packages.
   - Make it for each project you have for reproducibility.
   - There are different tools to create virtual environments.
       - ``conda``, only recommended for personal use and at some clusters
       - ``virtualenv``, may require to load extra python bundle modules.
       - ``venv``, most straight-forward and available at all HPC centers. **Recommended**

.. admonition:: Summary of Venv workflow

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

NSC:

- https://www.nsc.liu.se/software/python/
- https://www.nsc.liu.se/software/anaconda/

PDC:

- https://support.pdc.kth.se/doc/applications/python/

LUNARC

- https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/#anaconda-distributions

UPPMAX

- https://docs.uppmax.uu.se/software/conda/
- https://hackmd.io/@pmitev/conda_on_Rackham

LUMI

- https://docs.lumi-supercomputer.eu/software/installing/container-wrapper/#examples-of-using-the-lumi-container-wrapper


.. seealso::

   - want to share your work? :ref:`devel_iso`
   - uploading files
      - `NAISS transfer course <https://uppmax.github.io/naiss_file_transfer_course/sessions/intro/>`_

