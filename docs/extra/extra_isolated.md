# Extra material about isolated environments

## Virtual environment - venv & virtualenv

With this tool you can download and install with ``pip`` from the [PyPI repository](https://pypi.org/)

:::{admonition} ``pip list`` documentation
   :class: dropdown


- ``--local``: If in a virtualenv that has global access, do not list globally-installed packages.
- ``--user``: Only output packages installed in user-site.
- [documentation](https://pip.pypa.io/en/stable/cli/pip_list)
:::

:::{note}

   - You can use "pip list" on the command line (after loading the python module) to see which packages are available and which versions.
   - Some packages may be inherited from the modules you have loaded
   - You can do ``pip list --local`` to see what is installed by you in the environment.
   - Some IDE:s like Spyder may only find those "local" packages
   - To save space, you should load any other Python modules you will need that are system installed before installing your own packages! Remember to choose ones that are compatible with the Python version you picked!
       -   ``--system-site-packages`` includes the packages already installed in the loaded python module.
   - The ``--no-cache-dir"`` option is required to **avoid it from reusing earlier installations from the same user in a different environment**.
   - The ``--no-build-isolation`` is to make sure that it uses the loaded modules from the module system when **building any Cython libraries**.
:::

## Conda

- [Conda](https://anaconda.org/anaconda/conda) is an installer of packages but also bigger toolkits and is useful also for R packages and C/C++ installations.

:::{warning}

   Drawbacks

   - Conda cannot use already install packages from the Python modules and libraries already installed, and hence installs them anyway
   - Conda is therefore known for creating **many** *small* files. Your disk space is not only limited in GB, but also in number of files (typically ``300000`` in $HOME).
   - Check your disk usage and quota limit
       - Do a ``conda clean -a`` once in a while to remove unused and unnecessary files
:::

:::{tip}

   - The conda environments including many small files are by default stored in ``~/.conda`` folder that is in your $HOME directory with limited storage.
   - Move your ``.conda`` directory to your project folder and make a soft link to it from ``$HOME``
   - Do the following (``mkdir -p`` ignores error output and will not recreate another folder if it already exists):
        - (replace what is inside ``<>`` with relevant path)

   - Solution 1

      This works nicely if you have several projects. Then you can change these variables according to what you are currently working with.

      ```bash

         export CONDA_ENVS_PATH="path/to/your/project/(subdir)"
         export CONDA_PKG_DIRS="path/to/your/project/(subdir)"
         mamba create --prefix=$CONDA_ENVS_PATH/<conda env name>
      ```

   - Solution 2

      - This may not be a good idea if you have several projects.

      ```bash

         mkdir -p ~/.conda
         mv ~/.conda /<path-to-project-folder>/<username>/
         ln -s /<path-to-project-folder>/<username>/.conda ~/.conda
      ```
:::

## Workflow ``venv``

1. Start from a Python version you would like to use (load the module):
    - This step are different at different clusters since the naming is different

2. Load the Python module you will be using, as well as any site-installed package modules (requires the ``--system-site-packages`` option later)
    - ``module load <python module>``

The next points will be the same for all clusters

3. Create the isolated environment with something like ``python -m venv <name-of-environment>``
    - use the ``--system-site-packages`` to include all "non-base" packages
    - include the full path in the name if you want the environment to be stored other than in the "present working directory".

4. Activate the environment with ``source <path to virtual environment>/bin activate``

:::{note}

   - ``source`` can most often be replaced by ``.``, like in ``. Example/bin/activate``. Note the important ``<space>`` after ``.``
   - For clarity we use the ``source`` style here.
:::

5. Install (or update) the environment with the packages you need with the ``pip install`` command

    - Note that ``--user`` must be omitted: else the package will be installed in the global user folder.
    - The ``--no-cache-dir"`` option is required to avoid it from reusing earlier installations from the same user in a different environment. The ``--no-build-isolation`` is to make sure that it uses the loaded modules from the module system when building any Cython libraries.

6. Work in the isolated environment
   - When activated you can always continue to add packages!
7. Deactivate the environment after use with ``deactivate``

:::{note}

   To save space, you should load any other Python modules you will need that are system installed before installing your own packages! Remember to choose ones that are compatible with the Python version you picked!
         ``--system-site-packages`` includes the packages already installed in the loaded python module.

   At HPC2N, NSC and LUNARC, you often have to load SciPy-bundle. This is how you on Tetralith (NSC) could create a venv (Example) with a SciPy-bundle included which is compatible with Python/3.11.5:

   ```console

       $ module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 # for NSC
       $ python -m venv --system-site-packages Example
   ```    
:::

:::{warning}

   Draw-backs

   - Only works for Python environments
   - Only works with Python versions already installed
:::


## Typical workflow Conda


The first 2 steps are cluster dependent and will therefore be slightly different.

1. Make conda available from a software module, like ``ml load conda`` or similar, or use own installation of miniconda or miniforge.
2. First time

   :::{admonition} First time
      :class: dropdown

      - The variables CONDA_ENVS_PATH and CONDA_PKG_DIRS contains the location of your environments. Set it to your project's environments folder, if you have one, instead of the $HOME folder.
      - Otherwise, the default is ``~/.conda/envs``.
      - Example:

      ```console

         $ export CONDA_ENVS_PATH="path/to/your/project/(subdir)"
         $ export CONDA_PKG_DIRS="path/to/your/project/(subdir)"
      ```
      
   :::

Next steps are the same for all clusters

3. Create the conda environment ``conda create -n <name-of-env>``
4. Activate the conda environment by: ``source activate <conda-env-name>``

    - You can define the packages to be installed here already.
    - If you want another Python version, you have to define it here, like: ``conda ... python=3.6.8``

5. Install the packages with ``conda install ...`` or ``pip install ...``
6. Now do your work!

    - When activated you can always continue to add packages!

7. Deactivate

:::{prompt}
    :language: bash
    :prompts: (python-36-env) $

    conda deactivate
:::
