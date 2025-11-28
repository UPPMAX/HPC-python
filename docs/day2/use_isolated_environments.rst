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

.. admonition:: For teachers

   - Introduction 5 m
   - venv 5 m
   - Conda 5
   - Exercises 30 m

Isolated environments
---------------------

- As an example, maybe you have been using TensorFlow 1.x.x for your project and 
    - now you need to install a package that requires TensorFlow 2.x.x 
    - but you will still be needing the old version of TensorFlow. 
- This is easily solved with isolated environments.

- Another example is when a reviewer want you to remake a figure. 
    - You have already started to use a newer Python version or newer packages and 
    - realise that your earlier script does not work anymore. 
- Having freezed the environment would have solved you from this issue!

.. note::
  
   Isolated/virtual environments solve a couple of problems:
   
   - You can install specific, also older, package versions into them.
   - You can create one for each project and no problem if the two projects require different versions.
   - You can remove the environment and create a new one, if not needed or with errors.
   - Good for reproducibility!

- Isolated environments let you create separate workspaces for different versions of Python and/or different versions of packages. 
- You can activate and deactivate them one at a time, and work as if the other workspace does not exist.

**The tools**

- Python's built-in ``venv`` module: uses pip       
- ``virtualenv`` (can be installed): uses pip   
- ``conda``/``forge``: uses ``conda``/``mamba``    

What happens at activation?
...........................

- Python version is defined by the environment.
    - Check with ``which python``, should show at path to the environment.
    - In conda you can define python version as well
    - Since ``venv`` is part of Python you will get the python version used when running the ``venv`` command.
- Packages are defined by the environent.
    - Check with ``pip list``
    - Conda can only see what you installed for it.
    - ``venv`` and ``virtualenv`` also see other packages if you allowed for that when creating the environment (``--system-site-packages``). 
- You can work in a Python shell or IDE (coming session)
- You can run scripts dependent on packages now instaleld in your environment.

.. warning::

   **About Conda on HPC systems**

   - Conda is good in many ways but can interact negatively when 
      - using the python modules (module load) at the same time
      - having base environment always active
   - Not recommended at HPC2N
   - At the other clusters, handle with care!

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
| Kebnekaise | venv **only**                   |
+------------+---------------------------------+
| LUMI       | venv, conda in container        |
+------------+---------------------------------+
| Pelle      | venv, Miniforge3/24.11.3-0      |
+------------+---------------------------------+
| Tetralith  | Anaconda3/2024.02-1             |
+------------+---------------------------------+
| LUMI       | conda-containerize              |
+------------+---------------------------------+

.. tip::

   - Try with ``venv`` first
   - If very troublesome, try with ``conda``

   - To use self-installed Python packages in a batch script, you also need to load the above mentioned modules and activate the environment. An example of this will follow later in the course. 
   - To see which Python packages you, yourself, have installed, you can use ``pip list --user`` while the environment you have installed the packages in is active. To see all packages, use ``pip list``. 

.. admonition:: Other tools perhaps covered in the future
   :class: dropdown

   - `pixi <https://pixi.sh/latest/>`_: package management tool for developers 
       - It allows the developer to install libraries and applications in a reproducible way. Use pixi cross-platform, on Windows, Mac and Linux.
       - could replace conda/mamba

   - `uv <https://docs.astral.sh/uv/>`_: An extremely fast Python package and project manager, written in Rust. 
       - A single tool to replace pip, pip-tools, pipx, poetry, pyenv, twine, virtualenv, and more

Virtual environment - venv & virtualenv
---------------------------------------

With this tool you can download and install with ``pip`` from the `PyPI repository <https://pypi.org/>`_

.. admonition:: venv vs. virtualenv
   :class: dropdown   

   - These are almost completely interchangeable
   - The difference being that **virtualenv supports older python versions** and has a few more minor unique features, while **venv is in the standard library**.
   - Step 1:
       - Virtualenv: ``virtualenv --system-site-packages Example``
       - venv: ``python -m venv --system-site-packages Example2``
   - Next steps are identical and involves "activating" and ``pip install``
   - We recommend ``venv`` in the course. Then we are just needing the Python module itself!

.. admonition:: Example NSC

   .. code-block:: console

      ml buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 
      which python
      python -V
      cd /proj/courses-fall-2025/users/<username>
      python -m venv env-matplotlib
      source activate  env-matplotlib
      pip install matplotlib
      python

   .. code-block:: python

      >>> import matplotlib

.. note::

   - You can use "pip list" on the command line (after loading the python module) to see which packages are available and which versions. 
   - Some packaegs may be inhereted from the moduels yopu have loaded
   - You can do ``pip list --local`` to see what is installed by you in the environment.
   - Some IDE:s like Spyder may only find those "local" packages

Conda
-----

- `Conda <https://anaconda.org/anaconda/conda>`_ is an installer of packages but also bigger toolkits and is useful also for R packages and C/C++ installations.

- Conda creates isolated environments not clashing with other installations of python and other versions of packages.
- Conda environment requires that you install all packages needed by yourself. 
    - That is,  you cannot load the python module and use the packages therein inside you Conda environment.

.. admonition:: Conda channels
   :class: dropdown

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
    
    You reach them all by loading the conda module. You don't have to state the specific channel when using UPPMAX. Otherwise you do with ``conda -c <channel> ...``
   

.. warning:: 

   Drawbacks
    
   - Conda cannot use already install packages from the Python modules and libraries already installed, and hence installs them anyway
   - Conda is therefore known for creating **many** *small* files. Your diskspace is not only limited in GB, but also in number of files (typically ``300000`` in $HOME). 
   - Check your disk usage and quota limit
       - Do a ``conda clean -a`` once in a while to remove unused and unnecessary files

.. tip::

   - The conda environemnts inclusing many small files are by default stored in ``~/.conda`` folder that is in your $HOME directory with limited storage.
   - Move your ``.conda`` directory to your project folder and make a soft link to it from ``$HOME``
   - Do the following (``mkdir -p`` ignores error output and will not recfreate anothe folder if it already exists):
        - (replace what is inside ``<>`` with relevant path)

   - Solution 1

      This works nicely if you have several projects. Then you can change these varables according to what you are currently working with.

      .. code-block:: bash
   
         export CONDA_ENVS_PATH="path/to/your/project/(subdir)"
         export CONDA_PKG_DIRS="path/to/your/project/(subdir)"
         mamba create --prefix=$CONDA_ENVS_PATH/<conda env name>

   - Solution 2 

      - This may not be a good idea if you have several projects.

      .. code-block:: bash

         $ mkdir -p ~/.conda
         $ mv ~/.conda /<path-to-project-folder>/<username>/
         $ ln -s /<path-to-project-folder>/<username>/.conda ~/.conda

.. admonition:: Example NSC

   .. code-block:: console

      module load Miniforge/24.7.1-2-hpc1
      export CONDA_PKG_DIRS=/proj/courses-fall-2025/users/$USER
      export CONDA_ENVS_PATH=/proj/courses-fall-2025/users/$USER
      mamba create --prefix=$CONDA_ENVS_PATH/numpy-proj-39 python=3.9.5 -c conda-forge
      mamba activate nsc-example
      # A prompt "(/path-to/nsc-example/)" should show up
      # double-check we are using python from the Conda environment!
      which python  # should point to the conda environment!
      python -V     # should give python version 3.9.5
      mamba install numpy

   .. code-block:: python

      >>> import numpy

.. admonition:: Comments
   :class: dropdown

   - When pinning with Conda, use single ``=`` instead of double (as used by pip)

.. admonition:: Conda base env
   :class: dropdown

   - When conda is loaded you will by default be in the base environment, which works in the same way as other conda environments. 
   - It includes a Python installation and some core system libraries and dependencies of Conda. It is a “best practice” to avoid installing additional packages into your base software environment.

.. admonition:: Conda cheat sheet
   
   - List packages in present environment:	         ``conda list``
   - List all environments:			                 ``conda info -e`` or ``conda env list``
   - Install a package:                              ``conda install somepackage``
   - Install from certain channel (conda-forge):     ``conda install -c conda-forge somepackage``
   - Install a specific version:                     ``conda install somepackage=1.2.3``
   - Create a new environment:                       ``conda create --name myenvironment``
   - Create a new environment from requirements.txt: ``conda create --name myenvironment --file requirements.txt``
   - On e.g. HPC systems where you don’t have write access to central installation directory: ``conda create --prefix /some/path/to/env``
   - Activate a specific environment:                ``conda activate myenvironment``
   - Deactivate current environment:                 ``conda deactivate``

.. admonition:: Conda vs mamba etc...
   :class: dropdown

   - `what-is-the-difference-with-conda-mamba-poetry-pip <https://pixi.sh/latest/misc/FAQ/#what-is-the-difference-with-conda-mamba-poetry-pip>`_

.. admonition:: What to do when a problem arises?
   :class: dropdown

   - If you experience unexpected problems with the conda provided by the module system on Rackham or anaconda3 on Dardel, you can easily install your own and maintain it yourself.
   - Read more at `Pavlin Mitev's page about conda on Rackham/Dardel <https://hackmd.io/@pmitev/conda_on_Rackham>`_ and change paths to relevant one for your system.
   - Or `Conda - "best practices" - UPPMAX <https://hackmd.io/@pmitev/module_conda_Rackham>`_

Install from file
------------------

- All centers has had different approaches in what is included in the module system and not.
- Therefore the solution to complete the necessary packages needed for the course lessons, different approaches has to be made.
- This is left as exercise for you, see Exercise 4 and 5.

venv
....

Make a requirements file:

.. code-block:: 

   pip freeze --local > requirements.txt

.. admonition:: How does it look like?
   :class: dropdown

   .. code-block:: text

      contourpy==1.3.3
      cycler==0.12.1
      fonttools==4.60.1
      kiwisolver==1.4.9
      matplotlib==3.10.7
      numpy==2.3.5
      pillow==12.0.0
      pyparsing==3.2.5
      python-dateutil==2.9.0.post0
      six==1.17.0

Install packages from a file 

.. code-block:: 

   pip install -r requirements.txt

conda/forge
...........

Make environment file:

.. code-block:: 

    conda env export > environment.yml

.. admonition:: How does it look like?
   :class: dropdown

   .. code-block:: yaml

      name: /lunarc/nobackup/projects/lu2025-17-52/bjornc/example
      channels:
        - conda-forge
      dependencies:
        - _libgcc_mutex=0.1=conda_forge
        - _openmp_mutex=4.5=2_gnu
        - alsa-lib=1.2.14=hb9d3cd8_0
        - brotli=1.2.0=hed03a55_1
        - brotli-bin=1.2.0=hb03c661_1
        - bzip2=1.0.8=hda65f42_8
        - ca-certificates=2025.11.12=hbd8a1cb_0
        - cairo=1.18.4=h3394656_0
        - contourpy=1.3.3=py312hd9148b4_3
        - cycler=0.12.1=pyhd8ed1ab_1
        - cyrus-sasl=2.1.28=hd9c7081_0
        - dbus=1.16.2=h3c4dab8_0
        - double-conversion=3.3.1=h5888daf_0
        - font-ttf-dejavu-sans-mono=2.37=hab24e00_0
        - font-ttf-inconsolata=3.000=h77eed37_0
        - font-ttf-source-code-pro=2.038=h77eed37_0
        - font-ttf-ubuntu=0.83=h77eed37_3
        - fontconfig=2.15.0=h7e30c49_1
        - fonts-conda-ecosystem=1=0
        - fonts-conda-forge=1=hc364b38_1
        - fonttools=4.60.1=py312h8a5da7c_0
        - freetype=2.14.1=ha770c72_0
        - graphite2=1.3.14=hecca717_2
        - harfbuzz=12.2.0=h15599e2_0
        - icu=75.1=he02047a_0
        - keyutils=1.6.3=hb9d3cd8_0
        - kiwisolver=1.4.9=py312h0a2e395_2
        - krb5=1.21.3=h659f571_0
        - lcms2=2.17=h717163a_0
        - ld_impl_linux-64=2.45=default_hbd61a6d_104
        - lerc=4.0.0=h0aef613_1
        - libblas=3.11.0=2_h4a7cf45_openblas
        ...
        this was just like 30%

Create an environment from a file. Do this on another computer or rename.

.. code-block:: 

   conda env create -f environment.yml
   # if renaming is necessary
   conda env create -f environment.yml -p /path/renamed

.. discussion::

   - Did you note the difference in file length?
   - What does it mean?

Exercises
---------

.. challenge:: Exercise 0: Make a decision between ``venv`` or ``conda``.

   - We recommend `conda` for LUNARC.
   - We recommend ``venv`` for HPC2N
   - Otherwise there are some kind of documentation at all sites. 
   - ``venv`` "should" work everywhere but has not been fully tested

Breakout room according to grouping

.. challenge:: Exercise 1: Cover the documentation for venvs or conda

   First try to find it by navigating.

   - Alvis: https://www.c3se.chalmers.se/documentation/first_time_users/
   - NSC: https://www.nsc.liu.se
   - PDC: https://support.pdc.kth.se/doc/
   - LUNARC: https://lunarc-documentation.readthedocs.io/en/latest/
   - UPPMAX: https://docs.uppmax.uu.se/
   - HPC2N: https://docs.hpc2n.umu.se/
   - LUMI: https://docs.lumi-supercomputer.eu/software

   .. solution::

      .. tabs::

         .. tab:: venv

            NSC:

            - `Python <https://www.nsc.liu.se/software/python/>`_

            PDC:

            - `Virtual environment with venv <https://pdc-support.github.io/pdc-intro/#165>`_

            LUNARC

            - `Python <https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/>`_
            
            UPPMAX (only Pelle)

            - `Python venv <https://docs.uppmax.uu.se/software/python_venv/>`_
            - `Video By Richel <https://www.youtube.com/watch?v=lj_Q-5l0BqU>`_
            
            HPC2N

            - `Venv <https://docs.hpc2n.umu.se/software/userinstalls/#venv>`_
            - `Video by Richel <https://www.youtube.com/watch?v=_ev3g5Zvn9g>`_
             
         .. tab:: conda

            NSC:

            - https://www.nsc.liu.se/software/anaconda/

            PDC:

            - https://support.pdc.kth.se/doc/applications/python/

            LUNARC

            - https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/#anaconda-distributions

            UPPMAX

            - https://docs.uppmax.uu.se/software/conda/
            - `Bianca <https://uppmax.github.io/bianca_workshops/extra/conda/>`_ 

            LUMI

            - https://docs.lumi-supercomputer.eu/software/installing/container-wrapper/#examples-of-using-the-lumi-container-wrapper

            HPC2N:

            - Not recommended

.. challenge:: Exercise 2: Prepare the course environment

   There will be a mix of conda and venv att all clusters except for HPC2N where all is ``venv``

   .. tabs::

      .. tab:: NSC

         1. Let's make a Spyder installation in a `conda environment <https://saturncloud.io/blog/how-to-ensure-that-spyder-runs-within-a-conda-environment/#step-2-create-a-conda-environment>`_ 
         
         .. code-block:: 
         
            module load Miniforge/24.7.1-2-hpc1
            export CONDA_PKG_DIRS=/proj/courses-fall-2025/users/$USER
            export CONDA_ENVS_PATH=/proj/courses-fall-2025/users/$USER
            mamba create --prefix=$CONDA_ENVS_PATH/spyder-env python=3.12 spyder 
            mamba activate spyder-env
            # A prompt "(/path-to/spyder-env/)" should show up
            # double-check we are using python from the Conda environment!
            which python  # should point to the conda environment!
            python -V     # should give python version 3.12.X

         - Let's install packages that we need. 

         .. code-block:: 
         
            conda install matplotlib pandas seaborn xarray dask numba        

         - Check that the above packages are there by ``conda list``. 

         We will put requirements files in the course project folder that you can build from in latter lessons.

         - These will cover 

             - TensorFlow
             - PyTorch
             - numba

      .. tab:: PDC 

         1. Let's make a Spyder installation in a `conda environment <https://saturncloud.io/blog/how-to-ensure-that-spyder-runs-within-a-conda-environment/#step-2-create-a-conda-environment>`_ 
        
         .. code-block:: 
         
            ml PDC/24.11
            ml miniconda3/25.3.1-1-cpeGNU-24.11
            export CONDA_ENVS_PATH="/cfs/klemming/projects/supr/courses-fall-2025/$USER/" #only needed once per session
            export CONDA_PKG_DIRS="/cfs/klemming/projects/supr/courses-fall-2025/$USER/" #only needed once per session
            conda create --prefix $CONDA_ENVS_PATH/spyder-env python=3.11.7 spyder 
            source activate spyder-env
            # A prompt "(/path-to/spyder-env/)" should show up
            # double-check we are using python from the Conda environment!
            which python  # should point to the conda environment!
            python -V     # should give python version 3.12.X

         - Let's install packages that we need. 

         .. code-block:: 
         
            conda install matplotlib pandas seaborn xarray dask numba        

         - Check that the above packages are there by ``conda list``. 

         2. Let's make a Jupyter installation based on Python 3.11.7

         .. code-block:: console

            ml PDC/24.11
            ml miniconda3/25.3.1-1-cpeGNU-24.11
            export CONDA_ENVS_PATH="/cfs/klemming/projects/supr/courses-fall-2025/$USER/" #only needed once per session
            export CONDA_PKG_DIRS="/cfs/klemming/projects/supr/courses-fall-2025/$USER/" #only needed once per session
            conda create --prefix $CONDA_ENVS_PATH/jupyter-env python=3.11.7 jupyter
            conda activate jupyter-env
            # A prompt "(/path-to/jupyter-env/)" should show up
            # double-check we are using python from the Conda environment!
            which python  # should point to the conda environment!
            python -V     # should give python version 3.11.7

         - Let's install packages that we need. 

         .. code-block:: 
         
            conda install matplotlib pandas seaborn xarray dask numba

         - Check that the above packages are there by ``conda list``.
     
         We will put requirements files in the course project folder that you can build from in latter lessons

         - These will cover 

             - TensorFlow
             - PyTorch
             - numba
            
      .. tab:: LUNARC 

         - Everything will work by just loading modules.
         - Go down to the other exercises!

      .. tab:: UPPMAX

         Let's make a Spyder installation in a `conda environment <https://saturncloud.io/blog/how-to-ensure-that-spyder-runs-within-a-conda-environment/#step-2-create-a-conda-environment>`_ 
         
         .. tabs::

            .. tab:: Bianca

               .. code-block:: 
         
                  ml conda
                  export CONDA_PKG_DIRS=/proj/<proj-dir>/$USER
                  export CONDA_ENVS_PATH=/proj/<proj-dir>/$USER
                  conda create --prefix $CONDA_ENVS_PATH/spyder-env python=3.12 spyder -c conda-forge
                  source activate spyder-env
                  # A prompt "(/path-to/spyder-env/)" should show up
                  # double-check we are using python from the Conda environment!
                  which python  # should point to the conda environment!
                  python -V     # should give python version 3.12.X

            .. tab:: Pelle

               .. code-block:: 
         
                  ml Miniforge3/24.11.3-0
                  export CONDA_PKG_DIRS=/proj/hpc-python-uppmax/$USER
                  export CONDA_ENVS_PATH=/proj/hpc-python-uppmax/$USER
                  conda create --prefix $CONDA_ENVS_PATH/spyder-env python=3.12 spyder -c conda-forge
                  source activate spyder-env
                  # A prompt "(/path-to/spyder-env/)" should show up
                  # double-check we are using python from the Conda environment!
                  which python  # should point to the conda environment!
                  python -V     # should give python version 3.12.X

         - Let's install packages that we need. 

         .. code-block:: 
         
            conda install matplotlib pandas seaborn xarray dask numba

         - Check that the above packages are there by ``conda list``.

         We will put requirements files in the course project folder that you can build from in latter lessons

         - These will cover 

             - TensorFlow
             - PyTorch

      .. tab:: HPC2N

         We will put requirements files in the course project folder that you can build from in latter lessons

         - These will cover 

             - TensorFlow
             - PyTorch
             - numba

.. challenge:: (Optional) Exercise 3: Install package with venv

   - Choose a track below 
   - Bianca users could follow
       - (attend or cover the `Bianca intermediate course <https://docs.uppmax.uu.se/courses_workshops/bianca_intermediate/>`__ yourself)

   - Confirm package is absent
   - Create environment in your user's folder in the course project
   - Activate environment
   - Confirm package is absent
   - Install package in isolated environment
   - Confirm package is now present
   - Deactivate environment
   - Confirm package is now absent again

   .. tabs::

      .. tab:: NSC 

         - Start in folder ``/proj/courses-fall-2025/$USER``
         - Follow the tutorial at `Python <https://www.nsc.liu.se/software/python/>`_: scroll down to "More on Python virtual environments (venvs)"

      .. tab:: PDC 

         - Start in folder ``/cfs/klemming/projects/snic/courses-fall-2025/$USER``
         - Follow the tutorial at Virtual environment with venv https://pdc-support.github.io/pdc-intro/#165

      .. tab:: UPPMAX: Pelle

         .. code-block:: console

            $ module load Python/3.12.3-GCCcore-13.3.0 
            $ python -m venv --system-site-packages /proj/hpc-python-uppmax/$USER/Example
            $ source /proj/hpc-python-uppmax/$USER/Example/bin/activate

        "Example" is the name of the virtual environment. The directory "Example" is created in the present working directory. The ``-m`` flag makes sure that you use the libraries from the python version you are using.

      .. tab:: HPC2N

         .. code-block:: console

            $ module load GCC/12.3.0 Python/3.11.3
            $ python -m venv /proj/nobackup/fall-courses/$USER/Example
            $ source /proj/nobackup/fall-courses/$USER/Example/bin/activate

         "Example" is the name of the virtual environment. You can name it whatever you want. The directory “Example” is created in the present working directory.

      .. tab:: LUNARC 

         .. code-block:: console

            module load GCC/12.3.0 Python/3.11.3
            python -m venv --system-site-packages /lunarc/nobackup/projects/lu2025-17-52/$USER/Example
            source /lunarc/nobackup/projects/lu2025-17-52/<user-dir>/Example/bin/activate``

         "Example" is the name of the virtual environment. You can name it whatever you want. The directory “Example” is created in the present working directory.

 
   - Note that your prompt is changing to start with (Example) to show that you are within an environment.

   - Install your packages with ``pip``. While not always needed, it is often a good idea to give the correct versions you want, to ensure compatibility with other packages you use. This example assumes your venv is activated: 

   .. code-block:: console

      (Example) $ pip install --no-cache-dir --no-build-isolation numpy matplotlib

   - Deactivate the venv.

   .. code-block:: console

      (Example) $ deactivate

   - Everytime you need the tools available in the virtual environment you activate it as above (after also loading the modules).

   .. prompt:: console

      $ source /proj/<your-project-id>/<your-dir>/Example/bin/activate

.. challenge:: (optional) Exercise 4: like 3, but for Conda

    Let's make an installation with the latest bug fix version of ``Python 3.12`` and compatible ``numpy`` and ``matplotlib`` in a `conda environment <https://saturncloud.io/blog/how-to-ensure-that-spyder-runs-within-a-conda-environment/#step-2-create-a-conda-environment>`_ 

   - Activate environment
   - Confirm package is absent
   - Install package in isolated environment: ``numpy`` + ``matplotlib``
   - Confirm package is now present
   - Deactivate environment
   - Confirm package is now absent again

   .. tabs::

      .. tab:: NSC

         .. code-block:: 
         
            module load Miniforge/24.7.1-2-hpc1
            export CONDA_PKG_DIRS=/proj/courses-fall-2025/users/$USER
            export CONDA_ENVS_PATH=/proj/courses-fall-2025/users/$USER
            mamba create --prefix=$CONDA_ENVS_PATH/examplepython=3.12 example 
            mamba activate example
            # A prompt "(/path-to/example)" should show up
            # double-check we are using python from the Conda environment!
            which python  # should point to the conda environment!
            python -V     # should give python version 3.12.X

      .. tab:: PDC 

         .. code-block:: 
         
            ml PDC/24.11
            ml miniconda3/25.3.1-1-cpeGNU-24.11
            export CONDA_ENVS_PATH="/cfs/klemming/projects/supr/courses-fall-2025/$USER/" #only needed once per session
            export CONDA_PKG_DIRS="/cfs/klemming/projects/supr/courses-fall-2025/$USER/" #only needed once per session
            conda create --prefix $CONDA_ENVS_PATH/example python=3.12  
            source activate example
            # A prompt "(/path-to/example)" should show up
            # double-check we are using python from the Conda environment!
            which python  # should point to the conda environment!
            python -V     # should give python version 3.12.X
    
      .. tab:: LUNARC 

         .. code-block:: console

            ml Miniforge3/24.1.2-0
            export CONDA_ENVS_PATH="/lunarc/nobackup/projects/lu2025-17-52/$USER/" #only needed once per session
            export CONDA_PKG_DIRS="/lunarc/nobackup/projects/lu2025-17-52/$USER/" #only needed once per session
            conda create --prefix $CONDA_ENVS_PATH/example python=3.12 
            conda activate example
            # A prompt "(/path-to/example)" should show up
            # double-check we are using python from the Conda environment!
            which python  # should point to the conda environment!
            python -V     # should give python version 3.12.X
    
      .. tab:: UPPMAX: Bianca

          .. code-block:: 
         
             ml conda
             export CONDA_PKG_DIRS=/proj/hpc-python-uppmax/$USER
             export CONDA_ENVS_PATH=/proj/hpc-python-uppmax/$USER
             conda create --prefix $CONDA_ENVS_PATH/example python=3.12 -c conda-forge
             source activate example
             # A prompt "(/path-to/example/)" should show up
             # double-check we are using python from the Conda environment!
             which python  # should point to the conda environment!
             python -V     # should give python version 3.12.X
   
      .. tab:: UPPMAX: Pelle
   
         .. code-block:: 
            
            ml Miniforge3/24.11.3-0
            export CONDA_PKG_DIRS=/proj/hpc-python-uppmax/$USER  #only needed once per session
            export CONDA_ENVS_PATH=/proj/hpc-python-uppmax/$USER  #only needed once per session
            conda create --prefix $CONDA_ENVS_PATH/example python=3.12 -c conda-forge
            source activate example
            # A prompt "(/path-to/example)" should show up
            # double-check we are using python from the Conda environment!
            which python  # should point to the conda environment!
            python -V     # should give python version 3.12.X
   
      .. tab:: HPC2N

         Skip this exercise
 
   - Let's install packages that we need. 

   .. code-block:: 
         
      conda install matplotlib numpy       

   - Check that the above packages are there by ``conda list``. 






.. challenge:: (optional) 5. Make a test environment and spread (venv)

   Read `here <https://uppmax.github.io/HPC-python/extra/isolated_deeper.html#creator-developer>`_ 

   1. make a virtual environment with the name ``venv1``. Do not include packages from the the loaded module(s)
   2. activate
   3. install ``matplotlib``
   4. make a requirements file of the content
   5. deactivate
   6. make another virtual environment with the name ``venv2``
   7. activate that
   8. install with the aid of the requirements file
   9. check the content
   10. open python shell from command line and try to import `matplotlib`
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

      pip install -r requirements-pip.txt

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

.. challenge:: (optional) Exercise 5b. Make a test environment (conda)

   Principle

   - Start in an environment created above
   - Export the settings: 

   .. code-block:: console

      conda env export > environment.yml

- Look at the resulting file!

   - Create a copy of the environment but called another way to not overwrite the original!
       - Usually someone else use your file or you, yourself, on another computer.

   .. code-block:: console

      conda env create -f environment.yml -p /lunarc/nobackup/projects/lu2025-17-52/bjornc/test



Summary
-------


.. keypoints::

   - With a virtual environment you can tailor an environment with specific versions for Python and packages, not interfering with other installed python versions and packages.
   - Make it for each project you have for reproducibility.
   - There are different tools to create virtual environments.
       - ``venv``, most straight-forward and available at all HPC centers. **Recommended**
            - only supports Python packages
       - ``conda``, only recommended at some clusters
            - supports more and is a bit more reliable
            - do not use together with Python modules
            - install in project folder due to many files.

.. admonition:: Documentation at the centres
   :class: dropdown

   NSC:

   - https://www.nsc.liu.se/software/python/
   - https://www.nsc.liu.se/software/anaconda/

   PDC:

   - https://support.pdc.kth.se/doc/applications/python/
   - https://pdc-support.github.io/pdc-intro/#165

   LUNARC

   - https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/#anaconda-distributions

   UPPMAX

   - https://docs.uppmax.uu.se/software/conda/
   - https://hackmd.io/@pmitev/conda_on_Rackham

   HPC2N

   - https://docs.hpc2n.umu.se/software/userinstalls/#venv

   LUMI

   - https://docs.lumi-supercomputer.eu/software/installing/container-wrapper/#examples-of-using-the-lumi-container-wrapper

Summary
-------

Workflow ``venv``
................

1. Start from a Python version you would like to use (load the module): 
    - This step are different at different clusters since the naming is different

2. Load the Python module you will be using, as well as any site-installed package modules (requires the ``--system-site-packages`` option later)
    - ``module load <python module>``

The next points will be the same for all clusters

3. Create the isolated environment with something like ``python -m venv <name-of-environment>`` 
    - use the ``--system-site-packages`` to include all "non-base" packages
    - include the full path in the name if you want the environment to be stored other than in the "present working directory".

4. Activate the environment with ``source <path to virtual environment>/bin activate``

.. note::
   
   - ``source`` can most often be replaced by ``.``, like in ``. Example/bin/activate``. Note the important <space> after ``.``
   - For clarity we use the ``source`` style here.


5. Install (or update) the environment with the packages you need with the ``pip install`` command

    - Note that ``--user`` must be omitted: else the package will be installed in the global user folder.
    - The ``--no-cache-dir"`` option is required to avoid it from reusing earlier installations from the same user in a different environment. The ``--no-build-isolation`` is to make sure that it uses the loaded modules from the module system when building any Cython libraries.

6. Work in the isolated environment
   - When activated you can always continue to add packages!
7. Deactivate the environment after use with ``deactivate``

.. note::

   To save space, you should load any other Python modules you will need that are system installed before installing your own packages! Remember to choose ones that are compatible with the Python version you picked! 
         ``--system-site-packages`` includes the packages already installed in the loaded python module.

   At HPC2N, NSC and LUNARC, you often have to load SciPy-bundle. This is how you on Tetralith (NSC) could create a venv (Example) with a SciPy-bundle included which is compatible with Python/3.11.5:

   .. code-block:: console

       $ module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 # for NSC
       $ python -m venv --system-site-packages Example

.. warning:: 

   Draw-backs

   - Only works for Python environments
   - Only works with Python versions already installed

Typical workflow Conda
......................

The first 2 steps are cluster dependent and will therefore be slightly different.

1. Make conda available from a software module, like ``ml load conda`` or similar, or use own installation of miniconda or miniforge.
2. First time

   .. admonition:: First time
      :class: dropdown   

      - The variables CONDA_ENVS_PATH and CONDA_PKG_DIRS contains the location of your environments. Set it to your project's environments folder, if you have one, instead of the $HOME folder.
      - Otherwise, the default is ``~/.conda/envs``. 
      - Example:
  
      .. code-block:: console

         $ export CONDA_ENVS_PATH="path/to/your/project/(subdir)"
         $ export CONDA_PKG_DIRS="path/to/your/project/(subdir)"

Next steps are the same for all clusters

3. Create the conda environment ``conda create -n <name-of-env>``
4. Activate the conda environment by: ``source activate <conda-env-name>``

    - You can define the packages to be installed here already.
    - If you want another Python version, you have to define it here, like: ``conda ... python=3.6.8``

5. Install the packages with ``conda install ...`` or ``pip install ...``
6. Now do your work!

    - When activated you can always continue to add packages!

7. Deactivate

 .. prompt:: 
    :language: bash
    :prompts: (python-36-env) $
    
    conda deactivate




.. seealso::

   - want to share your work? :ref:`devel_iso`
   - uploading files
      - `NAISS transfer course <https://uppmax.github.io/naiss_file_transfer_course/sessions/intro/>`_

