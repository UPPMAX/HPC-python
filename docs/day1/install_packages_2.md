# Install packages #

<dl>
  <dt>   Learners can </dt>
  <dd>
    <p>Learners can</p>
    <p>- work (create, activate, work, deactivate) with virtual environments</p>
    <p>- install a python package</p>
    <p>- export and import a virtual environment</p>
  </dd>
</dl>
## Introduction ##

There are 2-3 ways to install missing python packages at a HPC cluster.

- <dl>
    <dt>Local installation, always available for the version of Python you had active when doing the installation</dt>
    <dd>- ``pip install --user [package name]``</dd>
  </dl>
- <dl>
    <dt>Isolated environment. Use some packages just needed for a specific use case.</dt>
    <dd>
      <p>- ``venv``/``virtualenv`` in combination with ``pip``</p>
    <p>- recommended/working in all HPC centers in Sweden</p>
    <p>- ``conda``</p>
    <p>- just recommended in some HPC centers in Sweden</p>
  </dd>
  </dl>

### Local (general installation) ###

<dl>
  <dt>.. note :  : </dt>
  <dd>
    <p>``pip install --user [package name]``</p>
    <p>- The package end up in ``~/.local``</p>
    <p>- target directory can be changed by ``--prefix=[root_folder of installation]``</p>
  </dd>
</dl>
### Isolated environments ###

As an example, maybe you have been using TensorFlow 1.x.x for your project and now you need to install a package that requires TensorFlow 2.x.x but you will still be needing the old version of TensorFlow for another package, for instance. This is easily solved with isolated environments.

.. note::
#### Isolated/virtual environments solve a couple of problems: ####

 > - You can install specific, also older, versions into them.
 > - You can create one for each project and no problem if the two projects require different versions.
 > - You can remove the environment and create a new one, if not needed or with errors.
 > 

- Isolated environments lets you create separate workspaces for different versions of Python and/or different versions of packages. 
- You can activate and deactivate them one at a time, and work as if the other workspace does not exist.
  

<dl>
  <dt>**The tools**</dt>
  <dd>
    <p>- venv            UPPMAX+HPC2N+LUNARC+NSC</p>
    <p>- virtualenv      UPPMAX+HPC2N+LUNARC+NSC</p>
    <p>- Conda           LUNARC + UPPMAX (recommended only for Bianca cluster)</p>
  </dd>
</dl>
<dl>
  <dt>.. warning :  : </dt>
  <dd>
    <p>**About Conda on HPC systems**</p>
    <p>- Conda is good in many ways but can interact negatively when trying to use the pytrhon modules in the HPC systems.</p>
    <p>- LUNARC seems to have working solutions</p>
    <p>- At UPPMAX Conda is installed but we have many users that get into problems.</p>
    <p>- However, on Bianca this is the most straight-forward way to install packages (no ordinary internet)</p>
  </dd>
</dl>
- `Anaconda at LUNARC <https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/#anaconda-distributions>`_
- `Conda at UPPMAX <https://docs.uppmax.uu.se/software/conda/>`_ 
  - `Conda on Bianca <https://uppmax.github.io/bianca_workshop/intermediate/install/#install-packages-principles>`_
    


## Virtual environment - venv & virtualenv ##

1. You load the Python module you will be using, as well as any site-installed package modules (requires the ``--system-site-packages`` option later)
2. You create the isolated environment with something like venv, virtualenv (use the ``--system-site-packages`` to include all "non-base" packages)
3. You activate the environment
4. You install (or update) the environment with the packages you need
5. You work in the isolated environment
6. You deactivate the environment after use 
   

- These are almost completely interchangeable
- The difference being that **virtualenv supports older python versions** and has a few more minor unique features, while **venv is in the standard library**.
- <dl>
    <dt>Step 1 : </dt>
    <dd>
      <p>- Virtualenv: ``virtualenv --system-site-packages Example``</p>
    <p>- venv: ``python -m venv --system-site-packages Example2``</p>
  </dd>
  </dl>
- Next steps are identical and involves "activating" and ``pip installs``
- We recommend ``venv`` in the course. Then we are just needing the Python module itself!
  

<dl>
  <dt>.. keypoints :  : </dt>
  <dd>
    <p>- With a virtual environment you can tailor an environment with specific versions for Python and packages, not interfering with other installed python versions and packages.</p>
    <p>- Make it for each project you have for reproducibility.</p>
    <p>- There are different tools to create virtual environments.</p>
    <p>- ``conda``, only recommended for personal use and at some clusters</p>
    <p>- ``virtualenv``, may require to load extra python bundle modules.</p>
    <p>- ``venv``, most straight-forward and available at all HPC centers. **Recommended**</p>
    <p>- More details to follow!</p>
  </dd>
</dl>
### Example ###

<dl>
  <dt>.. tip :  : </dt>
  <dd>
    <p></p>
    <p>**Do not type along!**</p>
  </dd>
</dl>
Create a ``venv``. First load the python version you want to base your virtual environment on:

<dl>
  <dt>.. tabs :  : </dt>
  <dd>
    <p>.. tab:: UPPMAX</p>
    <p>.. code-block:: console</p>
    <p>$ module load python/3.11.8</p>
    <p>$ python -m venv --system-site-packages Example2</p>
    <p></p>
    <p>"Example2" is the name of the virtual environment. The directory "Example2" is created in the present working directory. The ``-m`` flag makes sure that you use the libraries from the python version you are using.</p>
    <p>.. tab:: HPC2N</p>
    <p>.. code-block:: console</p>
    <p>$ module load GCC/12.3.0 Python/3.11.3</p>
    <p>$ python -m venv --system-site-packages Example2</p>
    <p>"Example2" is the name of the virtual environment. You can name it whatever you want. The directory “Example2” is created in the present working directory.</p>
    <p>.. tab:: LUNARC</p>
    <p>.. code-block:: console</p>
    <p>$ module load GCC/12.3.0 Python/3.11.3</p>
    <p>$ python -m venv --system-site-packages Example2</p>
    <p>"Example2" is the name of the virtual environment. You can name it whatever you want. The directory “Example2” is created in the present working directory.</p>
    <p>.. tab:: NSC</p>
    <p>.. code-block:: console</p>
    <p>$ ml buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5</p>
    <p>$ python -m venv --system-site-packages Example2</p>
    <p>"Example2" is the name of the virtual environment. You can name it whatever you want. The directory “Example2” is created in the present working directory.</p>
    <p></p>
  </dd>
</dl>
<dl>
  <dt>.. note :  : </dt>
  <dd>
    <p>To save space, you should load any other Python modules you will need that are system installed before installing your own packages! Remember to choose ones that are compatible with the Python version you picked!</p>
    <p>``--system-site-packages`` includes the packages already installed in the loaded python module.</p>
    <p>At HPC2N, NSC and LUNARC, you often have to load SciPy-bundle. This is how you could create a venv (Example3) with a SciPy-bundle included which is compatible with Python/3.11.3:</p>
    <p></p>
    <p>.. code-block:: console</p>
    <p>$ module load GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 # for HPC2N and LUNAR</p>
    <p>$ module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 # for NSC</p>
    <p>$ python -m venv --system-site-packages Example3</p>
  </dd>
</dl>
**NOTE**: since it may take up a bit of space if you are installing many Python packages to your virtual environment, we **strongly** recommend you place it in your project storage! 

<dl>
  <dt>**NOTE** : if you need to for instance working with both Python 2 and 3, then you can of course create more than one virtual environment, just name them so you can easily remember which one has what. </dt>
  <dd></dd>
</dl>
- <dl>
    <dt>Example for course project location and ``$USER`` being you user name. </dt>
    <dd>- If your directory in the project has another name, replace ``$USER`` with that one!</dd>
  </dl>
- <dl>
    <dt>UPPMAX : </dt>
    <dd>
      <p>- Create: ``python -m venv /proj/hpc-python-fall/$USER/Example``</p>
    <p>- Activate: ``source /proj/hpc-python-fall/<user-dir>/Example/bin/activate``</p>
  </dd>
  </dl>
- <dl>
    <dt>HPC2N : </dt>
    <dd>
      <p>- Create: ``python -m venv /proj/nobackup/hpc-python-fall-hpc2n/$USER/Example``</p>
    <p>- Activate: ``source /proj/nobackup/hpc-python-fall-hpc2n/<user-dir>/Example/bin/activate``</p>
  </dd>
  </dl>
- <dl>
    <dt>LUNARC : </dt>
    <dd>
      <p>- Create: ``python -m venv /lunarc/nobackup/projects/lu2024-17-44/$USER/Example``</p>
    <p>- Activate: ``source /lunarc/nobackup/projects/lu2024-17-44/<user-dir>/Example/bin/activate``</p>
  </dd>
  </dl>
- <dl>
    <dt>NSC : </dt>
    <dd>
      <p>- Create: ``python -m venv /proj/hpc-python-fall-nsc/$USER/Example``</p>
    <p>- Activate: ``source /proj/hpc-python-fall-nsc/<user-dir>/Example/bin/activate``</p>
    <p></p>
  </dd>
  </dl>
  Note that your prompt is changing to start with (Example) to show that you are within an environment.
  

<dl>
  <dt>.. note :  : </dt>
  <dd>
    <p>- ``source`` can most often be replaced by ``.``, like in ``. Example/bin/activate``. Note the important <space> after ``.``</p>
    <p>- For clarity we use the ``source`` style here.</p>
  </dd>
</dl>
### Install packages to the virtual environment with pip ###

<dl>
  <dt>.. tip :  : </dt>
  <dd>
    <p>**Do not type along!**</p>
    <p></p>
  </dd>
</dl>
Install your packages with ``pip``. While not always needed, it is often a good idea to give the correct versions you want, to ensure compatibility with other packages you use. This example assumes your venv is activated: 

<dl>
  <dt>      </dt>
  <dd>
    <p></p>
    <p>(Example) $ pip install --no-cache-dir --no-build-isolation numpy matplotlib</p>
  </dd>
</dl>
The ``--no-cache-dir"`` option is required to avoid it from reusing earlier installations from the same user in a different environment. The ``--no-build-isolation`` is to make sure that it uses the loaded modules from the module system when building any Cython libraries.

Deactivate the venv.

<dl>
  <dt>      </dt>
  <dd>
    <p></p>
    <p>(Example) $ deactivate</p>
    <p></p>
  </dd>
</dl>
Everytime you need the tools available in the virtual environment you activate it as above (after also loading the modules).

<dl>
  <dt>   source /proj/<your-project-id>/<your-dir>/Example/bin/activate</dt>
  <dd>
    <p>source /proj/<your-project-id>/<your-dir>/Example/bin/activate</p>
    <p></p>
    <p></p>
  </dd>
</dl>
<dl>
  <dt>.. note :  : </dt>
  <dd>
    <p>- You can use "pip list" on the command line (after loading the python module) to see which packages are available and which versions.</p>
    <p>- Some packaegs may be inhereted from the moduels yopu have loaded</p>
    <p>- You can do ``pip list --local`` to see what is instaleld by you in the environment.</p>
    <p>- Some IDE:s like Spyder may only find those "local" packages</p>
  </dd>
</dl>
## Working with virtual environments defined from files ##

### Creator/developer ###

- First _create_ and _activate_ an environment (see above)
- Install packages with pip
- Create file from present virtual environment:
  

   $ pip freeze > requirements.txt

- That includes also the *system site packages* if you included them with ``--system-site-packages``
- Test that everything works by running use cases scripts within the environment
- You can list packages specific for the virtualenv by ``pip list --local`` 
- So, creating a file from just the local environment:
  

   $ pip freeze --local > requirements.txt

<dl>
  <dt>   ``requirements.txt`` (used by the virtual environment) is a simple text file which looks similar to this :  : </dt>
  <dd>
    <p>``requirements.txt`` (used by the virtual environment) is a simple text file which looks similar to this::</p>
    <p>numpy</p>
    <p>matplotlib</p>
    <p>pandas</p>
    <p>scipy</p>
    <p>``requirements.txt`` with versions that could look like this::</p>
    <p>numpy==1.20.2</p>
    <p>matplotlib==3.2.2</p>
    <p>pandas==1.1.2</p>
    <p>scipy==1.6.2</p>
  </dd>
</dl>
- Deactivate
  

### User ###

- Create an environment based on dependencies given in an environment file
- This can be done in new virtual environment or as a genera installtion locally (not activating any environment
  
   pip install -r requirements.txt
- Check
  

<dl>
  <dt>   pip list</dt>
  <dd>
    <p>pip list</p>
    <p></p>
  </dd>
</dl>
- `Dependency management from course Python for Scientific computing <https://aaltoscicomp.github.io/python-for-scicomp/dependencies/>`_
  

<dl>
  <dt> : class : dropdown</dt>
  <dd>
    <p>:class: dropdown</p>
    <p>It is difficult to give an exhaustive list of useful packages for Python in HPC, but this list contains some of the more popular ones:</p>
    <p>.. list-table:: Popular packages</p>
    <p>:widths: 8 10 10 20</p>
    <p>:header-rows: 1</p>
    <p>* - Package</p>
    <p>- Module to load, UPPMAX</p>
    <p>- Module to load, HPC2N</p>
    <p>- Brief description</p>
    <p>* - Dask</p>
    <p>- ``python``</p>
    <p>- ``dask``</p>
    <p>- An open-source Python library for parallel computing.</p>
    <p>* - Keras</p>
    <p>- ``python_ML_packages``</p>
    <p>- ``Keras``</p>
    <p>- An open-source library that provides a Python interface for artificial neural networks. Keras acts as an interface for both the TensorFlow and the Theano libraries.</p>
    <p>* - Matplotlib</p>
    <p>- ``python`` or ``matplotlib``</p>
    <p>- ``matplotlib``</p>
    <p>- A plotting library for the Python programming language and its numerical mathematics extension NumPy.</p>
    <p>* - Mpi4Py</p>
    <p>- Not installed</p>
    <p>- ``SciPy-bundle``</p>
    <p>- MPI for Python package. The library provides Python bindings for the Message Passing Interface (MPI) standard.</p>
    <p>* - Numba</p>
    <p>- ``python``</p>
    <p>- ``numba``</p>
    <p>- An Open Source NumPy-aware JIT optimizing compiler for Python. It translates a subset of Python and NumPy into fast machine code using LLVM. It offers a range of options for parallelising Python code for CPUs and GPUs.</p>
    <p>* - NumPy</p>
    <p>- ``python``</p>
    <p>- ``SciPy-bundle``</p>
    <p>- A library that adds support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.</p>
    <p>* - Pandas</p>
    <p>- ``python``</p>
    <p>- ``SciPy-bundle``</p>
    <p>- Built on top of NumPy. Responsible for preparing high-level data sets for machine learning and training.</p>
    <p>* - PyTorch/Torch</p>
    <p>- ``PyTorch`` or ``python_ML_packages``</p>
    <p>- ``PyTorch``</p>
    <p>- PyTorch is an ML library based on the C programming language framework, Torch. Mainly used for natural language processing or computer vision.</p>
    <p>* - SciPy</p>
    <p>- ``python``</p>
    <p>- ``SciPy-bundle``</p>
    <p>- Open-source library for data science. Extensively used for scientific and technical computations, because it extends NumPy (data manipulation, visualization, image processing, differential equations solver).</p>
    <p>* - Seaborn</p>
    <p>- ``python``</p>
    <p>- Not installed</p>
    <p>- Based on Matplotlib, but features Pandas’ data structures. Often used in ML because it can generate plots of learning data.</p>
    <p>* - Sklearn/SciKit-Learn</p>
    <p>- ``scikit-learn``</p>
    <p>- ``scikit-learn``</p>
    <p>- Built on NumPy and SciPy. Supports most of the classic supervised and unsupervised learning algorithms, and it can also be used for data mining, modeling, and analysis.</p>
    <p>* - StarPU</p>
    <p>- Not installed</p>
    <p>- ``StarPU``</p>
    <p>- A task programming library for hybrid architectures. C/C++/Fortran/Python API, or OpenMP pragmas.</p>
    <p>* - TensorFlow</p>
    <p>- ``TensorFlow``</p>
    <p>- ``TensorFlow``</p>
    <p>- Used in both DL and ML. Specializes in differentiable programming, meaning it can automatically compute a function’s derivatives within high-level language.</p>
    <p>* - Theano</p>
    <p>- Not installed</p>
    <p>- ``Theano``</p>
    <p>- For numerical computation designed for DL and ML applications. It allows users to define, optimise, and gauge mathematical expressions, which includes multi-dimensional arrays.</p>
    <p>Remember, in order to find out how to load one of the modules, which prerequisites needs to be loaded, as well as which versions are available, use ``module spider <module>`` and ``module spider <module>/<version>``.</p>
    <p>Often, you also need to load a python module, except in the cases where it is included in ``python`` or ``python_ML_packages`` at UPPMAX or with ``SciPy-bundle`` at HPC2N.</p>
    <p>NOTE that not all versions of Python will have all the above packages installed! </p>
  </dd>
</dl>
<dl>
  <dt>   In addition to loading Python, you will also often need to load site-installed modules for Python packages, or use own-installed Python packages. The work-flow would be something like this : </dt>
  <dd>
    <p>In addition to loading Python, you will also often need to load site-installed modules for Python packages, or use own-installed Python packages. The work-flow would be something like this:</p>
    <p></p>
  </dd>
</dl>
#### 1. Load Python and prerequisites: ``module load <pre-reqs> Python/<version>`` ####

1. Load site-installed Python packages (optional): ``module load <pre-reqs> <python-package>/<version>``
   1. Create the virtual environment: ``python -m venv [PATH]/Example``
   2. Activate your virtual environment: ``source <path-to-virt-env>/Example/bin/activate``
   3. Install any extra Python packages: ``pip install --no-cache-dir --no-build-isolation <python-package>``
   4. Start Python or run python script: ``python``
   5. Do your work
   6. Deactivate
   7. Installed Python modules (modules and own-installed) can be accessed within Python with ``import <package>`` as usual. 
   8. The command ``pip list`` given within Python will list the available modules to import. 
   9. More about packages and virtual/isolated environment to follow in later sections of the course! 
      


## Exercises ##

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
   
   :class:     dropdown
13. First load the required Python module(s) if not already done so in earlier lessons. Remember that this steps differ between the HPC centers
14. make the first environment
   
   $ python -m venv venv1
15. Activate it.
   
   $ source venv1/bin/activate
   
  - Note that your prompt is changing to start with ``(venv1)`` to show that you are within an environment.

16. install ``matplotlib``
   
   pip install matplotlib
17. make a requirements file of the content
   
   pip freeze --local > requirements.txt
18. deactivate
   
   deactivate
19. make another virtual environment with the name ``venv2``
   
   python -m venv venv2
20. activate that
   
   source venv2/bin/activate
21. install with the aid of the requirements file
   
   pip install -r requirements.txt
22. check the content
   
   pip list
23. open python shell from command line and try to import
   
   python
   
   import matplotlib
24. exit python
   
   exit()
25. deactivate
   
   deactivate
   
   Prepare fore the course environments

---

<dl>
  <dt>.. note :  : </dt>
  <dd>
    <p>- All centers has had different approaches in what is included in the module system and not.</p>
    <p>- Therefore the solution to complete the necessary packages needed for the course lessons, different approaches has to be made.</p>
    <p>- This is left as exercise for you</p>
  </dd>
</dl>
We will need to install the LightGBM Python package for one of the examples in the ML section. 

<dl>
  <dt>.. tip :  : </dt>
  <dd>
    <p></p>
    <p>**Follow the track where you are working right now**</p>
  </dd>
</dl>
Create a virtual environment called ``vpyenv``. First load the python version you want to base your virtual environment on, as well as the site-installed ML packages. 

<dl>
  <dt>.. tabs :  : </dt>
  <dd>
    <p>.. tab:: NSC</p>
    <p>**If you do not have matplotlib already outside any virtual environment**</p>
    <p>- Install matplotlib in your ``.local`` folder, not in a virtual environment.</p>
    <p>- Do:</p>
    <p>.. code-block:: console</p>
    <p>ml buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5</p>
    <p>pip install --user matplotlib</p>
  </dd>
</dl>
#### - Check that matplotlib is there by ``pip list`` ####

 > **Check were to find environments needed for the lessons in the afternoon tomorrow**
 > 
 > - browse ``/proj/hpc-python-fall-nsc/`` to see the available environments. 
 > - <dl>
 >   <dt>their names are</dt>
 > <dd>
 >   <p>- ``venvNSC-TF``</p>
    <p>- ``venvNSC-torch``</p>
    <p>- ``venvNSC-numba``</p>
    <p>- ``venv-spyder-only``</p>
  </dd>
 > </dl>
 >  .. tab:: LUNARC 
 > - Everything will work by just loading modules, see each last section
 > - Extra exercise can be to reproduce the examples above.
 > 
 >  .. tab:: UPPMAX
 > 

 > **Check were to find environments needed for the lessons in the afternoon tomorrow**
 > 
 > - browse ``/proj/hpc-python-fall/`` to see the available environments. 
 > - <dl>
 >   <dt>their names are, for instance</dt>
 > <dd>
 >   <p>- ``venv-spyder``</p>
    <p>- ``venv-TF``</p>
    <p>- ``venv-torch``</p>
  </dd>
 > </dl>
 > - Extra exercise can be to reproduce the examples above.
 > 
 >  .. tab:: HPC2N
 > 

 > **Check were to find possible environments needed for the lessons in the afternoon tomorrow**
 > 
 > - browse ``/proj/nobackup/hpc-python-fall-hpc2n/`` to see the available environments.
 > - It may be empty for now but may show up by tomorrow
 > - <dl>
 >   <dt>their names may be, for instance</dt>
 > <dd>
 >   <p>- ``venv-TF``</p>
    <p>- ``venv-torch``</p>
  </dd>
 > </dl>
 > - Extra exercise can be to reproduce the examples above.
 > 

<dl>
  <dt>.. note :  : </dt>
  <dd>
    <p>- To use self-installed Python packages in a batch script, you also need to load the above mentioned modules and activate the environment. An example of this will follow later in the course.</p>
    <p>- To see which Python packages you, yourself, have installed, you can use ``pip list --user`` while the environment you have installed the packages in are active. To see all packages, use ``pip list``. </p>
  </dd>
</dl>
<dl>
  <dt>.. seealso :  : </dt>
  <dd>
    <p>- UPPMAX's documentation pages about installing Python packages and virtual environments: http://docs.uppmax.uu.se/software/python/#installing-python-packages</p>
    <p>- HPC2N's documentation pages about installing Python packages and virtual environments: https://www.hpc2n.umu.se/resources/software/user_installed/python</p>
  </dd>
</dl>
<dl>
  <dt>.. keypoints :  : </dt>
  <dd>
    <p>- With a virtual environment you can tailor an environment with specific versions for Python and packages, not interfering with other installed python versions and packages.</p>
    <p>- Make it for each project you have for reproducibility.</p>
    <p>- There are different tools to create virtual environemnts.</p>
    <p></p>
    <p>- UPPMAX has ``conda`` and ``venv`` and ``virtualenv``</p>
    <p>- HPC2N has ``venv`` and ``virtualenv``</p>
  </dd>
</dl>
