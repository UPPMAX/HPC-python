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

- As an example, maybe you have been using TensorFlow 1.x.x for your project and 
    - now you need to install a package that requires TensorFlow 2.x.x 
    - but you will still be needing the old version of TensorFlow for another package, for instance. 
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
    - venv and virtualenv also see other packages if you allowed for that when creating the environment (``--system-site-packages``). 
- You can work in a Python shell or IDE (coming session)
- You can run scripts dependent on packages now instaleld in your environment.

.. warning::

   **About Conda on HPC systems**

   - Conda is good in many ways but can interact negatively when 
      - using the python modules (module load) at the same time
      - having base environment always active
   - Not recommended at HPC2N
   - At the other clusters, handle with care!
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
| LUMI       | venv, conda in container        |
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

   - To use self-installed Python packages in a batch script, you also need to load the above mentioned modules and activate the environment. An example of this will follow later in the course. 
   - To see which Python packages you, yourself, have installed, you can use ``pip list --user`` while the environment you have installed the packages in are active. To see all packages, use ``pip list``. 


.. admonition:: Other tools perhaps covered in the future

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
   - Next steps are identical and involves "activating" and ``pip installs``
   - We recommend ``venv`` in the course. Then we are just needing the Python module itself!


Typical workflow
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
5. Install (or update) the environment with the packages you need with the ``pip install`` command
    - note that ``--user`` must be omitted: else the package
        will be installed in the global user folder.

6. Work in the isolated environment
   - When activated you can always continue to add packages!
7. Deactivate the environment after use with ``deactivate``

.. note::

   To save space, you should load any other Python modules you will need that are system installed before installing your own packages! Remember to choose ones that are compatible with the Python version you picked! 
         ``--system-site-packages`` includes the packages already installed in the loaded python module.

   At HPC2N, NSC and LUNARC, you often have to load SciPy-bundle. This is how you could create a venv (Example3) with a SciPy-bundle included which is compatible with Python/3.11.3:

   .. code-block:: console

       $ module load GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 
       $ module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 # for NSC
       $ python -m venv --system-site-packages Example3

.. admonition:: Draw-backs

   - Only works for Python environments
   - Only works with Python versions already installed

.. admonition:: Example NSC

   .. code-block:: console

      ml buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 
      which python
      python -V
      cd /proj/hpc-python-spring-naiss/users/<username>
      python -m venv env-matplotlib
      activate  env-matplotlib
      pip install matplotlib
      python
      >>> import matplotlib


Conda
-----

- Conda is an installer of packages but also bigger toolkits and is useful also for R packages and C/C++ installations.
- Conda creates isolated environments not clashing with other installations of python and other versions of packages.
- Conda environment requires that you install all packages needed by yourself. 
    - That is,  you cannot load the python module and use the packages therein inside you Conda environment.

.. warning::
 
    - Conda is known to create **many** *small* files. Your diskspace is not only limited in GB, but also in number of files (typically ``300000`` in $HOME). 
    - Check your disk usage and quota limit
    - Do a ``conda clean -a`` once in a while to remove unused and unnecessary files

.. tip::

   - The conda environemnts inclusing many small files are by default stored in ``~/.conda`` folder that is in your $HOME directory with limited storage.
   - Move your ``.conda`` directory to your project folder and make a soft link to it from $HOME
   - Do the following (``mkdir -p`` ignores error output and will not recfreate anothe folder if it already exists):
        - (replace what is inside ``<>`` with relevant path)

   - Solution 1

      This works nicely if you have several projects. Then you can change these varables according to what you are currently working with.

   .. code-block:: bash
   
      export CONDA_ENVS_PATH="path/to/your/project/(subdir)"
      export CONDA_PKG_DIRS="path/to/your/project/(subdir)"

   - Solution 2 

      - This is not good if you have several projects.

   .. code-block:: bash

      $ mkdir -p ~/.conda
      $ mv ~/.conda /<path-to-project-folder>/<username>/
      $ ln -s /<path-to-project-folder>/<username>/.conda ~/.conda

Typical workflow
................

The first 2 steps are cluster dependent and will therefore be slightly different.

1. Make conda available from a software module, like ``ml load conda`` or similar, or use own installation of miniconda or miniforge.
2. First time

   .. admonition:: First time
      :class: dropdown   

      - The variable CONDA_ENVS_PATH contains the location of your environments. Set it to your project's environments folder, if you have one, instead of the $HOME folder.
      - Otherwise, the default is ``~/.conda/envs``. 
      - Example:
  
      .. code-block:: console
 
         $ export CONDA_ENVS_PATH=/proj/<your-project-id>/nobackup/<username>
         $ export CONDA_ENVS_PATH="path/to/your/project/(subdir)"
         $ export CONDA_PKG_DIRS="path/to/your/project/(subdir)"

  
      .. admonition:: By choice
         :class: dropdown
 
      Run ``source conda_init.sh`` to initialise your shell (bash) to be able to run ``conda activate`` and ``conda deactivate`` etcetera instead of ``source activate``. It will modify (append) your ``.bashrc`` file.

Next steps are the same for all clusters

3. Create the conda environment
4. Activate the conda environment by: source activate <conda-env-name>
5. Now do your work!
   - When activated you can always continue to add packages!

6. Deactivate

 .. prompt:: 
    :language: bash
    :prompts: (python-36-env) $
    
    conda deactivate

.. admonition:: Conda base env

   - When conda is loaded you will by default be in the base environment, which works in the same way as other conda environments. It includes a Python installation and some core system libraries and dependencies of Conda. It is a “best practice” to avoid installing additional packages into your base software environment.

.. admonition:: Conda cheat sheet
   :class: dropdown
   
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
   :class: dropdown

   - `what-is-the-difference-with-conda-mamba-poetry-pip <https://pixi.sh/latest/misc/FAQ/#what-is-the-difference-with-conda-mamba-poetry-pip>`_

.. warning::

   - If you experience unexpected problems with the conda provided by the module system on Rackham or anaconda3 on Dardel, you can easily install your own and maintain it yourself.
   - Read more at `Pavlin Mitev's page about conda on Rackham/Dardel <https://hackmd.io/@pmitev/conda_on_Rackham>`_ and change paths to relevant one for your system.
   - Or `Conda - "best practices" - UPPMAX <https://hackmd.io/@pmitev/module_conda_Rackham>`_

Install from file/Set up course environment
-------------------------------------------

- All centers has had different approaches in what is included in the module system and not.
- Therefore the solution to complete the necessary packages needed for the course lessons, different approaches has to be made.
- This is left as exercise for you, see Exercise 3


Exercises
---------

.. challenge:: Exercise 0: Make a decision between ``venv`` or ``conda``.

   - We recommend Conda for LUNARC.
   - We recommend ``venv`` for HPC2N
   - Otherwise there are some kind of documentation at all sites. 
   - ``venv`` "should" work everywhere but has not been fully tested

Breakout room according to grouping

.. challenge:: Exercise 1: Cover the documentation

   First try to find it by navigating.

   - Alvis: https://www.c3se.chalmers.se/documentation/first_time_users/
   - NSC: https://www.nsc.liu.se
   - PDC: https://support.pdc.kth.se/doc/
   - `LUNARC <https://lunarc-documentation.readthedocs.io/en/latest/>`_. 
   - UPPMAX: https://docs.uppmax.uu.se/
   - HPC2N: https://docs.hpc2n.umu.se/
   - LUMI: https://docs.lumi-supercomputer.eu/software

   .. solution::

      .. tabs::

         .. tab:: venv

            NSC:

            - https://www.nsc.liu.se/software/python/

            PDC:

            - `Virtual environment with venv <https://pdc-support.github.io/pdc-intro/#165`>

            LUNARC

            - https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/
            
            UPPMAX

            - https://docs.uppmax.uu.se/software/python_venv/
            - `Video By Richel <https://www.youtube.com/watch?v=lj_Q-5l0BqU>`_
            
            HPC2N

            - https://docs.hpc2n.umu.se/software/userinstalls/#venv
            - Video: https://www.youtube.com/watch?v=_ev3g5Zvn9g
             
            LUMI

            - https://docs.lumi-supercomputer.eu/software/installing/container-wrapper/#examples-of-using-the-lumi-container-wrapper

          .. tab: conda

            NSC:

            - https://www.nsc.liu.se/software/anaconda/

            PDC:

            - https://support.pdc.kth.se/doc/applications/python/

            LUNARC

            - https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/#anaconda-distributions

            UPPMAX

            - https://docs.uppmax.uu.se/software/conda/

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
            mamba create -n spyder-env spyder
            mamba activate spyder-env

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


      .. tab:: PDC 

         1. Let's make a Spyder installation in a `conda environment <https://saturncloud.io/blog/how-to-ensure-that-spyder-runs-within-a-conda-environment/#step-2-create-a-conda-environment>`_ 

         
         .. code-block:: 
         
            ml PDC/23.12
            ml miniconda3/24.7.1-0-cpeGNU-23.12
            conda create --prefix /cfs/klemming/projects/supr/hpc-python-spring-naiss/$USER/spyder-env
            mamba activate spyder-env
            conda install spyder

         **fix** 

         .. code-block:: console

            $ module load PDC/21.11
            $ module load Anaconda3/2021.05
            $ cd /cfs/klemming/home/u/username
            $ python3 -m venv my-venv-dardel

       .. tab:: LUNARC 

         - Everything will work by just loading modules.
         - Go down to optional

      .. tab:: UPPMAX

         1. Let's make a Spyder installation in a `conda environment <https://saturncloud.io/blog/how-to-ensure-that-spyder-runs-within-a-conda-environment/#step-2-create-a-conda-environment>`_ 
         
         .. code-block:: 
         
            ml conda
            export CONDA_PKG_DIRS=/proj/hpc-python-uppmax/bjornc
            export CONDA_ENVS_PATH=/proj/hpc-python-uppmax/bjornc
            conda create -n spyder-env spyder -c conda-forge
            source activate spyder-env

         **Check were to find environments needed for the lessons in the afternoon tomorrow**

         - browse ``/proj/hpc-python-uppmax/`` to see the available environments. 
         - their names are, for instance
             - ``venv-TF``
             - ``venv-torch``

         - Extra exercise can be to reproduce the examples above.

      .. tab:: HPC2N

         **Check where to find possible environments needed for the lessons in the afternoon tomorrow**

         - browse ``/proj/nobackup/hpc-python-spring/`` to see the available environments.
         - It may be empty for now but may show up by tomorrow
         - their names may be, for instance
             - ``venv-TF``
             - ``venv-torch``

.. challenge:: (Optional) Exercise 3a: Install package

   - Choose a package of the ones below

       - mhcnuggets

   - Confirm package is absent
   - Create environment
   - Activate environment
   - Confirm package is absent
   - Install package in isolated environment
   - Confirm package is now present
   - Deactivate environment
   - Confirm package is now absent again

      **NOTE**: since it may take up a bit of space if you are installing many Python packages to your virtual environment, we **strongly** recommend you place it in your project storage! 

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

               Follow the turtorial at `Python <https://www.nsc.liu.se/software/python/>`_: scroll down to "More on Python virtual environments (venvs)"

            .. tab:: PDC 

               Follow the tutorial at Virtual environment with venv https://pdc-support.github.io/pdc-intro/#165


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


.. challenge:: (optional) 4a. Make a test environment and spread (venv)

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


Summary
.......



.. keypoints::

   - With a virtual environment you can tailor an environment with specific versions for Python and packages, not interfering with other installed python versions and packages.
   - Make it for each project you have for reproducibility.
   - There are different tools to create virtual environments.
       - ``venv``, most straight-forward and available at all HPC centers. **Recommended**
       - ``conda``, only recommended for personal use and at some clusters

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


.. seealso::

   - want to share your work? :ref:`devel_iso`
   - uploading files
      - `NAISS transfer course <https://uppmax.github.io/naiss_file_transfer_course/sessions/intro/>`_

