# Load and run python and use packages #

<dl>
  <dt>At UPPMAX, HPC2N, LUNARC, and NSC (and most other Swedish HPC centres) we call the applications available via the module system modules. </dt>
  <dd>
    <p>- http://docs.uppmax.uu.se/cluster_guides/modules/</p>
    <p>- https://docs.hpc2n.umu.se/documentation/modules/</p>
    <p>- https://lunarc-documentation.readthedocs.io/en/latest/manual/manual_modules/</p>
    <p>- https://www.nsc.liu.se/software/modules/</p>
  </dd>
</dl>
- Show how to load Python
- Show how to run Python scripts and start the Python command line
  

<dl>
  <dt>    </dt>
  <dd>
    <p></p>
    <p>- See which modules exists: ``module spider`` or ``ml spider``</p>
    <p>- Find module versions for a particular software: ``module spider <software>``</p>
    <p>- Modules depending only on what is currently loaded: ``module avail`` or ``ml av``</p>
    <p>- See which modules are currently loaded: ``module list`` or ``ml``</p>
    <p>- Load a module: ``module load <module>/<version>`` or ``ml <module>/<version>``</p>
    <p>- Unload a module: ``module unload <module>/<version>`` or ``ml -<module>/<version>``</p>
    <p>- More information about a module: ``module show <module>/<version>`` or ``ml show <module>/<version>``</p>
    <p>- Unload all modules except the 'sticky' modules: ``module purge`` or ``ml purge``</p>
    <p></p>
  </dd>
</dl>
<dl>
  <dt>.. warning :  : </dt>
  <dd>
    <p></p>
    <p>- Note that the module systems at UPPMAX, HPC2N, LUNARC, and NSC are slightly different.</p>
    <p>- While all modules at</p>
    <p>- UPPMAX not directly related to bio-informatics are shown by ``ml avail``</p>
    <p>- NSC are show by ``ml avail``</p>
    <p>- HPC2N and LUNARC are hidden until one has loaded a prerequisite like the compiler ``GCC``.</p>
  </dd>
</dl>
- For reproducibility reasons, you should always load a specific version of a module instead of just the default version
- Many modules have prerequisite modules which needs to be loaded first (at HPC2N/LUNARC/NSC this is also the case for the Python modules). When doing ``module spider <module>/<version>`` you will get a list of which other modules needs to be loaded first
  

## Check for Python versions ##

<dl>
  <dt>.. tip :  : </dt>
  <dd>
    <p></p>
    <p>**Type along!**</p>
  </dd>
</dl>
<dl>
  <dt>.. tabs :  : </dt>
  <dd>
    <p>.. tab:: UPPMAX</p>
    <p>Check all available Python versions with:</p>
    <p>.. code-block:: console</p>
    <p>$ module avail python</p>
    <p>NOTE that python is written in lower case!</p>
    <p>.. tab:: HPC2N</p>
    <p></p>
    <p>Check all available version Python versions with:</p>
    <p>.. code-block:: console</p>
  </dd>
</dl>
### $ module spider Python ###

 > To see how to load a specific version of Python, including the prerequisites, do 
 > 
 > <dl>
 >   <dt>   </dt>
 > <dd>
 >   <p></p>
    <p>$ module spider Python/<version></p>
  </dd>
 > </dl>
 > Example for Python 3.11.3 
 > 
 > <dl>
 >   <dt>   $ module spider Python/3.11.3</dt>
 > <dd>
 >   <p>$ module spider Python/3.11.3</p>
    <p>.. tab:: LUNARC </p>
  </dd>
 > </dl>
 > Check all available Python versions with: 
 > 
 >    $ module spider Python 
 > 
 > To see how to load a specific version of Python, including the prerequisites, do 
 > 
 >    $ module spider Python/<version>
 > 
 > Example for Python 3.11.5 
 > 
 > <dl>
 >   <dt>   $ module spider Python/3.11.5</dt>
 > <dd>
 >   <p>$ module spider Python/3.11.5</p>
    <p>.. tab:: NSC</p>
  </dd>
 > </dl>
 > Check all available Python versions with: 
 > 
 >    $ module spider Python
 > 
 > To see how to load a specific version of Python, including the prerequisites, do 
 > 
 >    $ module spider Python/<version>
 > 
 > Example for Python 3.10.4
 > 
 >    $ module spider Python/3.10.4
 > 
<dl>
  <dt> : class : dropdown</dt>
  <dd>
    <p>:class: dropdown</p>
    <p></p>
    <p>.. code-block::  console</p>
    <p></p>
    <p>----------------------------------- /sw/mf/rackham/applications -----------------------------------</p>
    <p>python_GIS_packages/3.10.8      python_ML_packages/3.9.5-gpu         wrf-python/1.3.1</p>
    <p>python_ML_packages/3.9.5-cpu    python_ML_packages/3.11.8-cpu (D)</p>
    <p></p>
    <p>------------------------------------ /sw/mf/rackham/compilers -------------------------------------</p>
    <p>python/2.7.6     python/3.4.3    python/3.9.5         python3/3.6.8     python3/3.11.8</p>
    <p>python/2.7.9     python/3.5.0    python/3.10.8        python3/3.7.2     python3/3.12.1 (D)</p>
    <p>python/2.7.11    python/3.6.0    python/3.11.4        python3/3.8.7</p>
    <p>python/2.7.15    python/3.6.8    python/3.11.8        python3/3.9.5</p>
    <p>python/3.3       python/3.7.2    python/3.12.1 (D)    python3/3.10.8</p>
    <p>python/3.3.1     python/3.8.7    python3/3.6.0        python3/3.11.4</p>
    <p>Where:</p>
    <p>D:  Default Module</p>
    <p>Use module spider" to find all possible modules and extensions.</p>
    <p>Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".</p>
  </dd>
</dl>
<dl>
  <dt> : class : dropdown</dt>
  <dd>
    <p>:class: dropdown</p>
    <p>.. code-block:: console</p>
    <p>b-an01 [~]$ module spider Python</p>
    <p>----------------------------------------------------------------------------</p>
    <p>Python:</p>
    <p>----------------------------------------------------------------------------</p>
    <p>Description:</p>
    <p>Python is a programming language that lets you work more quickly and</p>
    <p>integrate your systems more effectively.</p>
    <p></p>
    <p>Versions:</p>
    <p>Python/2.7.15</p>
    <p>Python/2.7.16</p>
    <p>Python/2.7.18-bare</p>
    <p>Python/2.7.18</p>
    <p>Python/3.7.2</p>
    <p>Python/3.7.4</p>
    <p>Python/3.8.2</p>
    <p>Python/3.8.6</p>
    <p>Python/3.9.5-bare</p>
    <p>Python/3.9.5</p>
    <p>Python/3.9.6-bare</p>
    <p>Python/3.9.6</p>
    <p>Python/3.10.4-bare</p>
    <p>Python/3.10.4</p>
    <p>Python/3.10.8-bare</p>
    <p>Python/3.10.8</p>
    <p>Python/3.11.3</p>
    <p>Python/3.11.5</p>
    <p>Other possible modules matches:</p>
    <p>Biopython  Boost.Python  GitPython  IPython  flatbuffers-python  ...</p>
    <p>----------------------------------------------------------------------------</p>
    <p>To find other possible module matches execute:</p>
    <p>$ module -r spider '.*Python.*'</p>
    <p>----------------------------------------------------------------------------</p>
    <p>For detailed information about a specific "Python" package (including how to load the modules) use the module's full name.</p>
    <p>Note that names that have a trailing (E) are extensions provided by other modules.</p>
    <p></p>
    <p>For example:</p>
    <p>$ module spider Python/3.9.5</p>
    <p>----------------------------------------------------------------------------</p>
  </dd>
</dl>
<dl>
  <dt> : class : dropdown</dt>
  <dd>
    <p>:class: dropdown</p>
    <p>.. code-block:: console</p>
    <p>$ module spider Python</p>
    <p>--------------------------------------------------------------------------------------------------------</p>
    <p>Python:</p>
    <p>--------------------------------------------------------------------------------------------------------</p>
    <p>Description:</p>
    <p>Python is a programming language that lets you work more quickly and integrate your systems more effectively.</p>
    <p>Versions:</p>
    <p>Python/2.7.18-bare</p>
    <p>Python/2.7.18</p>
    <p>Python/3.8.6</p>
    <p>Python/3.9.5-bare</p>
    <p>Python/3.9.5</p>
    <p>Python/3.9.6-bare</p>
    <p>Python/3.9.6</p>
    <p>Python/3.10.4-bare</p>
    <p>Python/3.10.4</p>
    <p>Python/3.10.8-bare</p>
    <p>Python/3.10.8</p>
    <p>Python/3.11.3</p>
    <p>Python/3.11.5</p>
    <p>Python/3.12.3</p>
    <p>Other possible modules matches:</p>
    <p>Biopython  GitPython  IPython  Python-bundle  Python-bundle-PyPI  bx-python  flatbuffers-python  ...</p>
    <p>--------------------------------------------------------------------------------------------------------</p>
    <p>To find other possible module matches execute:</p>
    <p>$ module -r spider '.*Python.*'</p>
    <p>--------------------------------------------------------------------------------------------------------</p>
    <p>For detailed information about a specific "Python" package (including how to load the modules) use the module's full name.</p>
    <p>Note that names that have a trailing (E) are extensions provided by other modules.</p>
    <p>For example:</p>
    <p>$ module spider Python/3.12.3</p>
    <p>--------------------------------------------------------------------------------------------------------</p>
  </dd>
</dl>
<dl>
  <dt> : class : dropdown</dt>
  <dd>
    <p>:class: dropdown</p>
    <p>.. code-block:: console</p>
    <p>$ module spider Python</p>
    <p>####################################################################################################################################</p>
    <p># NOTE: At NSC the output of 'module spider' is generally not helpful as all relevant software modules are shown by 'module avail' #</p>
    <p># Some HPC centers hide software until the necessary dependencies have been loaded. NSC does not do that.                          #</p>
    <p>####################################################################################################################################</p>
    <p>----------------------------------------------------------------------------</p>
    <p>Python:</p>
    <p>----------------------------------------------------------------------------</p>
    <p>Versions:</p>
    <p>Python/recommendation</p>
    <p>Python/2.7.18-bare-hpc1-gcc-2022a-eb</p>
    <p>Python/2.7.18-bare</p>
    <p>Python/3.10.4-bare-hpc1-gcc-2022a-eb</p>
    <p>Python/3.10.4-bare</p>
    <p>Python/3.10.4-env-hpc1-gcc-2022a-eb</p>
    <p>Python/3.10.4-env-hpc2-gcc-2022a-eb</p>
    <p>Python/3.10.4</p>
    <p>Python/3.10.8-bare</p>
    <p>Python/3.10.8</p>
    <p>Python/3.11.3</p>
    <p>Python/3.11.5</p>
    <p>Other possible modules matches:</p>
    <p>IPython  netcdf4-python</p>
    <p>----------------------------------------------------------------------------</p>
    <p>To find other possible module matches execute:</p>
    <p>$ module -r spider '.*Python.*'</p>
    <p>----------------------------------------------------------------------------</p>
    <p>For detailed information about a specific "Python" package (including how to load the modules) use the module's full name.</p>
    <p>Note that names that have a trailing (E) are extensions provided by other modules.</p>
    <p>For example:</p>
    <p>$ module spider Python/3.11.5</p>
    <p>----------------------------------------------------------------------------</p>
  </dd>
</dl>
   Unless otherwise said, we recommend using Python 3.11.x in this course at HPC2N, UPPMAX, LUNARC, and NSC. We will us Python 3.10.4 at NSC for a small number of examples, since more packages are installed for that. 

## Load a Python module ##

For reproducibility, we recommend ALWAYS loading a specific module instad of using the default version! 

<dl>
  <dt>.. tip :  : </dt>
  <dd>
    <p></p>
    <p>**Type along!**</p>
  </dd>
</dl>
<dl>
  <dt>.. tabs :  : </dt>
  <dd>
    <p>.. tab:: UPPMAX</p>
    <p></p>
    <p>Go back and check which Python modules were available. To load version 3.11.8, do:</p>
    <p>.. code-block:: console</p>
    <p>$ module load python/3.11.8</p>
    <p></p>
    <p>Note: Lowercase ``p``.</p>
    <p>For short, you can also use:</p>
    <p>.. code-block:: console</p>
    <p>$ ml python/3.11.8</p>
  </dd>
</dl>
### .. tab:: HPC2N ###

 > To load Python version 3.11.3, do:         
### .. code-block:: console ###

 > <dl>
 >   <dt>$ module load GCC/12.3.0 Python/3.11.3</dt>
 > <dd>
 >   <p>Note: Uppercase ``P``.</p>
    <p>For short, you can also use:</p>
    <p>.. code-block:: console</p>
  </dd>
 > </dl>
 > <dl>
 >   <dt>$ ml GCC/12.3.0 Python/3.11.3</dt>
 > <dd>
 >   <p>.. tab:: LUNARC</p>
    <p>To load Python version 3.11.5, do:</p>
    <p>.. code-block:: console</p>
  </dd>
 > </dl>
 > <dl>
 >   <dt>$ module load GCC/13.2.0 Python/3.11.5</dt>
 > <dd>
 >   <p>Note: Uppercase ``P``.</p>
    <p>For short, you can also use:</p>
    <p>.. code-block:: console</p>
  </dd>
 > </dl>
 > <dl>
 >   <dt>$ ml GCC/13.2.0 Python/3.11.5</dt>
 > <dd>
 >   <p>.. tab:: NSC (Tetralith)</p>
    <p>To load Python version 3.11.5, do:</p>
    <p>.. code-block:: console</p>
  </dd>
 > </dl>
 > <dl>
 >   <dt>$ ml buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5</dt>
 > <dd>
 >   <p>To load Python version 3.10.4, do:</p>
    <p>.. code-block:: console</p>
  </dd>
 > </dl>
 > <dl>
 >   <dt>$ module load buildtool-easybuild/4.8.0-hpce082752a2  GCC/11.3.0 Python/3.10.4</dt>
 > <dd>
 >   <p>GCC/11.3.0 Python/3.10.4</p>
    <p>Note: Uppercase ``P``.</p>
    <p>For short, you can also use (Python/3.10.4):</p>
    <p>.. code-block:: console</p>
  </dd>
 > </dl>
 > $ ml buildtool-easybuild/4.8.0-hpce082752a2  GCC/11.3.0 Python/3.10.4 
 > 
<dl>
  <dt>.. warning :  : </dt>
  <dd>
    <p>+ UPPMAX: Don’t use system-installed python (2.7.5)</p>
    <p>+ UPPMAX: Don't use system installed python3 (3.6.8)</p>
    <p>+ HPC2N: Don’t use system-installed python (2.7.18)</p>
    <p>+ HPC2N: Don’t use system-installed python3  (3.8.10)</p>
    <p>+ LUNARC: Don’t use system-installed python/python3 (3.9.18)</p>
    <p>+ NSC: Don't use system-installed python/python3 (3.9.18)</p>
    <p>+ ALWAYS use python module</p>
  </dd>
</dl>
- Some existing software might use `Python2` and some will use `Python3`. 
- Some of the Python packages have both `Python2` and `Python3` versions. 
- Check what your software as well as the installed modules need when you pick!   
  
- Sometimes existing software might use `python2` and there's nothing you can do about that.
- In pipelines and other toolchains the different tools may together require both `python2` and `python3`.
- Here's how you handle that situation:
  
  - You can run two python modules at the same time if ONE of the module is ``python/2.X.Y`` and the other module is ``python3/3.X.Y`` (not ``python/3.X.Y``).
    


   The answer depends on which module is loaded. If Python/3.X.Y is loaded, then ``python`` is just an alias for ``python3`` and it will start the same command line. However, if Python/2.7.X is loaded, then ``python`` will start the Python/2.7.X command line while ``python3`` will start the system version (3.9.18). If you load Python/2.7.X and then try to load Python/3.X.Y as well, or vice-versa, the most recently loaded Python version will replace anything loaded prior, and all dependencies will be upgraded or downgraded to match. Only the system’s Python/3.X.Y version can be run at the same time as a version of Python/2.7.X.

## Run ##

#### Run Python script ####

<dl>
  <dt>.. hint :  : </dt>
  <dd>
    <p>- There are many ways to edit your scripts.</p>
    <p>- If you are rather new.</p>
    <p>- Graphical: ``$ gedit <script> &``</p>
    <p></p>
    <p>- (``&`` is for letting you use the terminal while editor window is open)</p>
    <p>- Requires ThinLinc or ``ssh -X``</p>
    <p>- Terminal: ``$ nano <script>``</p>
    <p>- Otherwise you would know what to do!</p>
    <p>- |:warning:| The teachers may use their common editor, like ``vi``/``vim``</p>
    <p>- If you get stuck in ``vim``, press: ``<esc>`` and then ``:q`` !</p>
  </dd>
</dl>
 

<dl>
  <dt>.. type-along :  : </dt>
  <dd>
    <p>- Let's make a script with the name ``example.py``</p>
    <p>.. code-block:: console</p>
    <p>$ nano example.py</p>
    <p>- Insert the following text</p>
    <p>.. code-block:: python</p>
    <p># This program prints Hello, world!</p>
    <p>print('Hello, world!')</p>
    <p>- Save and exit. In nano: ``<ctrl>+O``, ``<ctrl>+X``</p>
    <p>You can run a python script in the shell like this:</p>
    <p>.. code-block:: bash</p>
    <p>$ python example.py</p>
    <p># or</p>
    <p>$ python3 example.py</p>
  </dd>
</dl>
<dl>
  <dt>.. warning :  : </dt>
  <dd>
    <p>- *ONLY* run jobs that are short and/or do not use a lot of resources from the command line.</p>
    <p>- Otherwise use the batch system (see the `batch session <https://uppmax.github.io/HPC-python/day1/batch.html>`_)</p>
    <p></p>
  </dd>
</dl>
#### Run an interactive Python shell ####

- You can start a simple python terminal by:
  

<dl>
  <dt>   $ python </dt>
  <dd>
    <p>$ python</p>
    <p></p>
  </dd>
</dl>
**Example**

<dl>
  <dt>   >>> a=3</dt>
  <dd>
    <p>>>> a=3</p>
    <p>>>> b=7</p>
    <p>>>> c=a+b</p>
    <p>>>> c</p>
    <p>10</p>
  </dd>
</dl>
- Exit Python with <Ctrl-D>, ``quit()`` or ``exit()`` in the python prompt
  

<dl>
  <dt>    >>> <Ctrl-D></dt>
  <dd>
    <p>>>> <Ctrl-D></p>
    <p>>>> quit()</p>
    <p>>>> exit()</p>
  </dd>
</dl>
For more interactiveness you can run Ipython.

<dl>
  <dt>.. tip :  : </dt>
  <dd>
    <p></p>
    <p>**Type along!**</p>
  </dd>
</dl>
<dl>
  <dt>.. tabs :  : </dt>
  <dd>
    <p>.. tab:: UPPMAX</p>
    <p>NOTE: remember to load a python module first. Then start IPython from the terminal</p>
    <p></p>
    <p>.. code-block:: console</p>
    <p>$ ipython</p>
    <p></p>
    <p>or</p>
    <p>.. code-block:: console</p>
    <p>$ ipython3</p>
    <p></p>
    <p>UPPMAX has also ``jupyter-notebook`` installed and available from the loaded Python module. Start with</p>
    <p></p>
    <p>.. code-block:: console</p>
    <p>$ jupyter-notebook</p>
    <p></p>
    <p>You can decide on your own favorite browser and add ``--no-browser`` and open the given URL from the output given.</p>
    <p>From python/3.10.8 and forward, also jupyterlab is available.</p>
    <p></p>
    <p></p>
    <p>.. tab:: HPC2N</p>
    <p></p>
    <p>NOTE: remember to load an IPython module first. You can see possible modules with</p>
    <p>.. code-block:: console</p>
    <p>$ module spider IPython</p>
    <p>And load one of them (here 8.14.0) with</p>
    <p>.. code-block:: console</p>
    <p></p>
    <p>$ ml GCC/12.3.0 IPython/8.14.0</p>
    <p></p>
    <p>Then start Ipython with (lowercase):</p>
    <p></p>
    <p>.. code-block:: console</p>
    <p>$ ipython</p>
    <p>HPC2N also has ``JupyterLab`` installed. It is available as a module, but the process of using it is somewhat involved. We will cover it more under the session on <a href="https://uppmax.github.io/HPC-python/day1/interactive.html">Interactive work on the compute nodes</a>. Otherwise, see this tutorial:</p>
    <p>- https://docs.hpc2n.umu.se/tutorials/jupyter/</p>
    <p>.. tab:: LUNARC</p>
    <p>NOTE: remember to load an IPython module first. You can see possible modules with</p>
    <p>.. code-block:: console</p>
    <p>$ module spider IPython</p>
    <p>And load one of them (here 8.14.0) with</p>
    <p>.. code-block:: console</p>
    <p></p>
    <p>$ ml GCC/12.3.0 IPython/8.14.0</p>
    <p></p>
    <p>Then start Ipython with (lowercase):</p>
    <p></p>
    <p>.. code-block:: console</p>
    <p>$ ipython</p>
    <p>LUNARC also has ``JupyterLab``, ``JupyterNotebook``, and ``JupyterHub`` installed.</p>
    <p>.. tab:: NSC (Tetralith)</p>
    <p>NOTE: remember to load an IPython module first. You can see possible modules with</p>
    <p>.. code-block:: console</p>
    <p>$ module spider IPython</p>
    <p>And load one of them (here 8.5.0) with</p>
    <p>.. code-block:: console</p>
    <p></p>
    <p>$ ml buildtool-easybuild/4.8.0-hpce082752a2  GCC/11.3.0 IPython/8.5.0</p>
    <p></p>
    <p>Then start Ipython with (lowercase):</p>
    <p></p>
    <p>.. code-block:: console</p>
    <p>$ ipython </p>
  </dd>
</dl>
- Exit IPython with <Ctrl-D>, ``quit()`` or ``exit()`` in the python prompt
  

iPython

<dl>
  <dt>    In [2] : <Ctrl-D></dt>
  <dd>
    <p>In [2]: <Ctrl-D></p>
    <p>In [12]: quit()</p>
    <p>In [17]: exit()</p>
  </dd>
</dl>
## Packages/Python modules ##

- Python **packages broaden the use of python** to almost infinity! 
- Instead of writing code yourself there may be others that have done the same!
- Many **scientific tools** are distributed as **python packages**, making it possible to run a script in the prompt and there define files to be analysed and arguments defining exactly what to do.
- 1. nice **introduction to packages** can be found here: `Python for scientific computing <https://aaltoscicomp.github.io/python-for-scicomp/dependencies/>`_
      


<dl>
  <dt>.. questions :  : </dt>
  <dd>
    <p>- How do I find which packages and versions are available?</p>
    <p>- What to do if I need other packages?</p>
    <p>- Are there differences between HPC2N, LUNARC, UPPMAX, and NSC?</p>
    <p></p>
  </dd>
</dl>
- Show how to check for Python packages
- show how to install own packages on the different clusters
  

## Check current available packages ##

#### General for all four centers ####

Some python packages are working as stand-alone tools, for instance in bioinformatics. The tool may be already installed as a module. Check if it is there by:

<dl>
  <dt>   $ module spider <tool-name or tool-name part> </dt>
  <dd>
    <p>$ module spider <tool-name or tool-name part></p>
    <p></p>
  </dd>
</dl>
Using ``module spider`` lets you search regardless of upper- or lowercase characters and regardless of already loaded modules (like ``GCC`` on HPC2N/LUNARC/NSC and ``bioinfo-tools`` on UPPMAX).

<dl>
  <dt>.. tabs :  : </dt>
  <dd>
    <p>.. tab:: UPPMAX</p>
    <p>Check the pre-installed packages of a specific python module:</p>
    <p>.. code-block:: console</p>
    <p>$ module help python/<version> </p>
  </dd>
</dl>
###   ###

 >  At HPC2N, a way to find Python packages that you are unsure how are names, would be to do
 > 
 > <dl>
 >   <dt> .. code-block :  : console</dt>
 > <dd>$ module -r spider ’.*Python.*’</dd>
 > </dl>
 >  or
 > 
 > <dl>
 >   <dt> .. code-block :  : console</dt>
 > <dd>$ module -r spider ’.*python.*’</dd>
 > </dl>
 >  Do be aware that the output of this will not just be Python packages, some will just be programs that are compiled with Python, so you need to check the list carefully.   
 > 
 > <dl>
 >   <dt>   At LUNARC, a way to find Python packages that you are unsure how are names, would be to do</dt>
 > <dd>
 >   <p>At LUNARC, a way to find Python packages that you are unsure how are names, would be to do</p>
    <p>.. code-block:: console</p>
    <p>$ module -r spider ’.*Python.*’</p>
    <p>or</p>
    <p>.. code-block:: console</p>
    <p>$ module -r spider ’.*python.*’</p>
    <p>Do be aware that the output of this will not just be Python packages, some will just be programs that are compiled with Python, so you need to check the list carefully.   </p>
  </dd>
 > </dl>
 > <dl>
 >   <dt>   At NSC, a way to find Python packages that you are unsure how are names, would be to do</dt>
 > <dd>
 >   <p>At NSC, a way to find Python packages that you are unsure how are names, would be to do</p>
    <p>.. code-block:: console</p>
    <p>$ module -r spider ’.*Python.*’</p>
    <p>or</p>
    <p>.. code-block:: console</p>
    <p>$ module -r spider ’.*python.*’</p>
    <p>Do be aware that the output of this will not just be Python packages, some will just be programs that are compiled with Python, so you need to check the list carefully.</p>
    <p></p>
  </dd>
 > </dl>
Check the pre-installed packages of a loaded python module, in shell:

   $ pip list

To see which Python packages you, yourself, has installed, you can use ``pip list --user`` while the environment you have installed the packages in are active.

You can also test from within python to make sure that the package is not already installed:

<dl>
  <dt>    >>> import <package></dt>
  <dd>
    <p>>>> import <package></p>
    <p></p>
  </dd>
</dl>
Does it work? Then it is there!

Otherwise, you can either use ``pip`` or ``conda``.

- In a python session, type:
  
   import [a_module]
   print([a_module].__file__)
- The print-out tells you the path to the `.pyc` file, but should give you a hint where it belongs.
  

- See if the following packages are installed. Use python version ``3.11.8`` on Rackham, ``3.11.3`` on Kebnekaise, ``3.11.5`` on Cosmos, and ``3.10.4`` on Tetralith (remember: the Python module on kebnekaise/cosmos/tetralith has prerequisite(s)). 
  
  - ``numpy``
  - ``mpi4py``
  - ``distributed``
  - ``multiprocessing``
  - ``time``
  - ``dask``
    
    .. solution::

- Rackham has for ordinary python/3.11.8 module already installed: 
  - ``numpy`` |:white_check_mark:|
  - ``pandas`` |:white_check_mark:|
  - ``mpi4py`` |:x:|
  - ``distributed`` |:x:|
  - ``multiprocessing`` |:white_check_mark:|  (standard library)
  - ``time`` |:white_check_mark:|  (standard library)
  - ``dask`` |:white_check_mark:|

- Kebnekaise has for ordinary Python/3.11.3 module already installed:
  - ``numpy`` |:x:|
  - ``pandas`` |:x:| 
  - ``mpi4py`` |:x:|
  - ``distributed`` |:x:|
  - ``multiprocessing`` |:white_check_mark:|  (standard library)
  - ``time`` |:white_check_mark:|  (standard library)
  - ``dask``  |:x:|

- Cosmos has for ordinary Python/3.11.5 module already installed: 
  - ``numpy`` |:x:|
  - ``pandas`` |:x:| 
  - ``mpi4py`` |:x:|
  - ``distributed`` |:x:|
  - ``multiprocessing`` |:white_check_mark:|  (standard library)
  - ``time`` |:white_check_mark:|  (standard library)
  - ``dask``  |:x:|

- Tetralith has for ordinary Python/3.10.4 module already installed: 
  - ``numpy`` |:x:|
  - ``pandas`` |:x:| 
  - ``mpi4py`` |:x:|
  - ``distributed`` |:x:|
  - ``multiprocessing`` |:white_check_mark:|  (standard library)
  - ``time`` |:white_check_mark:|  (standard library)
  - ``dask``  |:x:|

- You could check for another Python version, say 3.11.5 on Tetralith!
- See next session how to find more pre-installed packages!
  

**NOTE**: at HPC2N, LUNARC, and NSC, the available Python packages needs to be loaded as modules/module-bundles before using! See a list of some of them below, under the HPC2N/LUNARC/NSC tab or find more as mentioned above, using ``module spider -r ...``

1. selection of the Python packages and libraries installed on UPPMAX, HPC2N, LUNARC, and NSC are given in extra reading: `UPPMAX clusters <https://uppmax.github.io/HPC-python/uppmax.html>`_ and `Kebnekaise cluster <https://uppmax.github.io/HPC-python/kebnekaise.html>`_ and eventually LUNARC cluster and NSC cluster
   

<dl>
  <dt>.. tabs :  : </dt>
  <dd>
    <p>.. tab:: UPPMAX</p>
    <p>- The python application at UPPMAX comes with several preinstalled packages.</p>
    <p>- You can check them here: `UPPMAX packages <https://uppmax.github.io/HPC-python/uppmax.html#uppmax-packages>`_.</p>
    <p>- In addition there are packages available from the module system as `python tools/packages <https://uppmax.github.io/HPC-python/uppmax.html#uppmax-packages>`_</p>
    <p>- Note that bioinformatics-related tools can be reached only after loading ``bioinfo-tools``.</p>
    <p>- Two modules contains topic specific packages. These are:</p>
    <p></p>
    <p>- Machine learning: ``python_ML_packages`` (cpu and gpu versions and based on python/3.9.5 and python/3.11.8)</p>
    <p>- GIS: ``python_GIS_packages`` (cpu version based on python/3.10.8)</p>
    <p>.. tab:: HPC2N</p>
    <p>- The python application at HPC2N comes with several preinstalled packages - check first before installing yourself!</p>
    <p>- HPC2N has both Python 2.7.x and Python 3.x installed.</p>
    <p>- We will be using Python 3.x in this course.  For this course, the recommended version of Python to use on Kebnekaise is 3.11.3.</p>
    <p>NOTE:  HPC2N do NOT recommend (and do not support) using Anaconda/Conda on our systems. You can read more about this here: `Anaconda <https://docs.hpc2n.umu.se/tutorials/anaconda/>`_.</p>
    <p>- This is a selection of the packages and libraries installed at HPC2N. These are all installed as **modules** and need to be loaded before use.</p>
    <p></p>
    <p>- ``ASE``</p>
    <p>- ``Keras``</p>
    <p>- ``PyTorch``</p>
    <p>- ``SciPy-bundle`` (Bottleneck, deap, mpi4py, mpmath, numexpr, numpy, pandas, scipy - some of the versions have more)</p>
    <p>- ``TensorFlow``</p>
    <p>- ``Theano``</p>
    <p>- ``matplotlib``</p>
    <p>- ``scikit-learn``</p>
    <p>- ``scikit-image``</p>
    <p>- ``iPython``</p>
    <p>- ``Cython``</p>
    <p>- ``Flask``</p>
    <p>- ``JupyterLab``</p>
    <p>- ``Python-bundle-PyPI`` (Bundle of Python packages from PyPi)</p>
    <p>.. tab:: LUNARC</p>
    <p>- The python application at LUNARC comes with several preinstalled packages - check first before installing yourself!</p>
    <p>- LUNARC has both Python 2.7.x and Python 3.x installed.</p>
    <p>- We will be using Python 3.x in this course.  For this course, the recommended version of Python to use on Cosmos is 3.11.5.</p>
    <p>- This is a selection of the packages and libraries installed at LUNARC. These are all installed as **modules** and need to be loaded before use.</p>
    <p>- ``PyTorch``</p>
    <p>- ``SciPy-bundle`` (Bottleneck, deap, mpi4py, mpmath, numexpr, numpy, pandas, scipy - some of the versions have more)</p>
    <p>- ``TensorFlow``</p>
    <p>- ``matplotlib``</p>
    <p>- ``scikit-learn``</p>
    <p>- ``scikit-image``</p>
    <p>- ``iPython``</p>
    <p>- ``Cython``</p>
    <p>- ``Biopython``</p>
    <p>- ``JupyterLab``</p>
    <p>- ``Python-bundle`` (NumPy, SciPy, Matplotlib, JupyterLab, MPI4PY, ...)  </p>
  </dd>
</dl>
### .. tab:: NSC (Tetralith)  ###

 > - The python application at NSC (Tetralith) comes with few preinstalled packages, but many can be found in extra modules - check first before installing yourself! 
 > - NSC has both Python 2.7.x and Python 3.x installed. 
 > - We will be using Python 3.x in this course.  For this course, the recommended version of Python to use on Tetralith is 3.11.5 in most cases, but 3.10.4 will be used for some examples. 
 > - <dl>
 >   <dt>This is a selection of the packages and libraries installed at NSC (Tetralith). These are all installed as **modules** and need to be loaded before use. </dt>
 > <dd>
 >   <p>- ``SciPy-bundle`` (Bottleneck, deap, mpi4py, mpmath, numexpr, numpy, pandas, scipy - some of the versions have more)</p>
    <p>- ``matplotlib``</p>
    <p>- ``iPython``</p>
    <p>- ``JupyterLab`` </p>
  </dd>
 > </dl>
 >   
 > 

## Demo/Type-along  ##

This is an exercise that combines loading, running, and using site-installed packages. Later, during the batch session, we will look at running the same exercise, but as a batch job. There is also a follow-up exercise of an extended version of the script, if you want to try run that as well (see further down on the page). 

We will **use** the pandas and matplotlib packages in this very simple example, but not explain anything about them. That comes later in the course! 

<dl>
  <dt>    You need the data-file ``scottish_hills.csv`` which can be found in the directory ``Exercises/examples/programs``. If you have cloned the git-repo for the course, or copied the tar-ball, you should have this directory. The easiest thing to do is just change to that directory and run the exercise there. </dt>
  <dd>
    <p>You need the data-file ``scottish_hills.csv`` which can be found in the directory ``Exercises/examples/programs``. If you have cloned the git-repo for the course, or copied the tar-ball, you should have this directory. The easiest thing to do is just change to that directory and run the exercise there.</p>
    <p>Since the exercise opens a plot, you need to login with ThinLinc (or otherwise have an x11 server running on your system and login with ``ssh -X ...``). </p>
  </dd>
</dl>
The exercise is modified from an example found on https://ourcodingclub.github.io/tutorials/pandas-python-intro/. 

<dl>
  <dt>.. warning :  : </dt>
  <dd>
    <p>**Not relevant if using UPPMAX. Only if you are using HPC2N, LUNARC, or NSC!**</p>
    <p>You need to also load Tkinter.</p>
    <p>**For HPC2N:**</p>
    <p>.. code-block:: console</p>
    <p>ml GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 Tkinter/3.11.3</p>
    <p>**For LUNARC**</p>
    <p>.. code-block:: console</p>
    <p>ml GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 matplotlib/3.8.2 Tkinter/3.11.5</p>
    <p>**For NSC (Tetralith)**</p>
    <p>.. code-block:: console</p>
    <p>ml buildtool-easybuild/4.8.0-hpce082752a2 GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4 SciPy-bundle/2022.05 matplotlib/3.5.2 Tkinter/3.10.4</p>
    <p>In addition, you need to add the following two lines to the top of your python script/run them first in Python, for HPC2N, LUNARC, and NSC:</p>
    <p>.. code-block:: python</p>
    <p>import matplotlib</p>
    <p>matplotlib.use('TkAgg')</p>
  </dd>
</dl>
<dl>
  <dt>   **NOTE** if you have loaded a different Python version than what we use here, do ``ml purge`` first to get a clean work area. </dt>
  <dd>
    <p>**NOTE** if you have loaded a different Python version than what we use here, do ``ml purge`` first to get a clean work area.</p>
    <p>We are using Python version ``3.11.x`` except on Tetralith where we use Python/3.10.4. To access the packages ``pandas`` and ``matplotlib``, you may need to load other modules, depending on the site where you are working.</p>
    <p></p>
    <p>.. tabs::</p>
    <p>.. tab:: UPPMAX</p>
    <p>Here you only need to load the ``python`` module, as the relevant packages are included (as long as you are not using GPUs, but that is talked about later in the course). Thus, you just do:</p>
    <p>.. code-block:: console</p>
    <p>$ ml python/3.11.8</p>
    <p>.. tab:: HPC2N</p>
    <p>On Kebnekaise you also need to load ``SciPy-bundle`` and ``matplotlib`` (and their prerequisites). These versions will work well together (and with the Tkinter/3.11.3):</p>
    <p>.. code-block:: console</p>
    <p>$ ml GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 Tkinter/3.11.3</p>
    <p></p>
    <p>.. tab:: LUNARC</p>
    <p>On Cosmos you also need to load ``SciPy-bundle`` and ``matplotlib`` (and their prerequisites). These versions will work well together (and with the Tkinter/3.11.5):</p>
    <p>.. code-block:: console</p>
    <p>$ ml GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 matplotlib/3.8.2 Tkinter/3.11.5</p>
    <p>.. tab:: NSC (Tetralith)</p>
    <p>On Tetralith you also need to load ``SciPy-bundle`` and ``matplotlib`` (and their prerequisites). In this example we will use Python 3.10.4 as that is the one that has compatible versions and has a compatible TKinter 3.10.4):</p>
    <p>.. code-block:: console</p>
    <p>$ ml buildtool-easybuild/4.8.0-hpce082752a2  GCC/11.3.0  OpenMPI/4.1.4 matplotlib/3.5.2 SciPy-bundle/2022.05 Tkinter/3.10.4</p>
    <p></p>
    <p>1. From inside Python/interactive (if you are on Kebnekaise/Cosmos/Tetralith, mind the warning above about loading a compatible Tkinter and adding the two lines importing matplotlib and setting TkAgg at the top):</p>
    <p>**Not on UPPMAX, but on HPC2N, LUNARC, NSC**: Start Python and run these lines:</p>
    <p>.. code-block:: python</p>
    <p>import matplotlib</p>
    <p>matplotlib.use('TkAgg')</p>
    <p>**On all systems**: Start python (if you have not already) and run these lines:</p>
    <p>.. code-block:: python</p>
    <p>import pandas as pd</p>
    <p>.. code-block:: python</p>
    <p>import matplotlib.pyplot as plt</p>
    <p>.. code-block:: python</p>
    <p>dataframe = pd.read_csv("scottish_hills.csv")</p>
    <p>.. code-block:: python</p>
    <p>x = dataframe.Height</p>
    <p>.. code-block:: python</p>
    <p>y = dataframe.Latitude</p>
    <p>.. code-block:: python</p>
    <p>plt.scatter(x, y)</p>
    <p>.. code-block:: python</p>
    <p>plt.show()</p>
    <p>If you change the last line to ``plt.savefig("myplot.png")`` then you will instead get a file ``myplot.png`` containing the plot. This is what you would do if you were running a python script in a batch job.</p>
    <p>- On UPPMAX, LUNARC, and NSC you can view png files with the program ``eog``</p>
    <p>- Test: ``eog myplot.png &``</p>
    <p>- On HPC2N you can view png files with the program ``eom``</p>
    <p>- Test: ``eom myplot.png &``</p>
    <p>2. As a Python script (if you are on Kebnekaise/Cosmos/Tetralith, mind the warning above about Tkinter):</p>
    <p>Copy and save this script as a file (or just run the file ``pandas_matplotlib-<system>.py`` that is located in the ``<path-to>/Exercises/examples/programs`` directory you got from the repo or copied. Where <system> is either ``rackham``, ``kebnekaise``, ``cosmos``, or ``tetralith``.</p>
    <p>.. tabs::</p>
    <p>.. tab:: rackham</p>
    <p>.. code-block:: python</p>
  </dd>
</dl>
### import pandas as pd ###

<dl>
  <dt>import matplotlib.pyplot as plt</dt>
  <dd>
    <p>dataframe = pd.read_csv("scottish_hills.csv")</p>
    <p>x = dataframe.Height</p>
    <p>y = dataframe.Latitude</p>
    <p>plt.scatter(x, y)</p>
    <p>plt.show()</p>
    <p>.. tab:: kebnekaise</p>
    <p>.. code-block:: python</p>
    <p>import pandas as pd</p>
    <p>import matplotlib</p>
    <p>import matplotlib.pyplot as plt</p>
    <p></p>
    <p>matplotlib.use('TkAgg')</p>
    <p>dataframe = pd.read_csv("scottish_hills.csv")</p>
    <p>x = dataframe.Height</p>
    <p>y = dataframe.Latitude</p>
    <p>plt.scatter(x, y)</p>
    <p>plt.show()</p>
    <p></p>
    <p>.. tab:: Cosmos</p>
    <p>.. code-block:: python</p>
    <p>import pandas as pd</p>
    <p>import matplotlib</p>
    <p>import matplotlib.pyplot as plt</p>
    <p></p>
    <p>matplotlib.use('TkAgg')</p>
    <p>dataframe = pd.read_csv("scottish_hills.csv")</p>
    <p>x = dataframe.Height</p>
    <p>y = dataframe.Latitude</p>
    <p>plt.scatter(x, y)</p>
    <p>plt.show()</p>
    <p></p>
    <p>.. tab:: Tetralith</p>
    <p>.. code-block:: python</p>
    <p>import pandas as pd</p>
    <p>import matplotlib</p>
    <p>import matplotlib.pyplot as plt</p>
    <p></p>
    <p>matplotlib.use('TkAgg')</p>
    <p>dataframe = pd.read_csv("scottish_hills.csv")</p>
    <p>x = dataframe.Height</p>
    <p>y = dataframe.Latitude</p>
    <p>plt.scatter(x, y)</p>
    <p>plt.show()</p>
    <p></p>
    <p></p>
  </dd>
</dl>
If you have time, you can also try and run these extended versions, which also requires the ``scipy`` packages (included with python at UPPMAX and with the same modules loaded as for ``pandas`` for HPC2N/LUNARC/NSC):

## Exercises  (C. 10 min) ##

<dl>
  <dt>   You can either save the scripts or run them line by line inside Python. The scripts are also available in the directory ``<path-to>/Exercises/examples/programs``, as ``pandas_matplotlib-linreg.py`` and ``pandas_matplotlib-linreg-pretty.py``.</dt>
  <dd>
    <p>You can either save the scripts or run them line by line inside Python. The scripts are also available in the directory ``<path-to>/Exercises/examples/programs``, as ``pandas_matplotlib-linreg.py`` and ``pandas_matplotlib-linreg-pretty.py``.</p>
    <p>**NOTE** that there are separate versions for rackham, kebnekaise, cosmos, and tetralith and that you for kebnekaise, cosmos, and tetralith need to again add the same lines regarding TkAgg as mentioned under the warning before the previous exercise. The example below shows how it looks for rackham.</p>
    <p>Remember that you also need the data file ``scottish_hills.csv`` located in the above directory.</p>
    <p>Examples are from https://ourcodingclub.github.io/tutorials/pandas-python-intro/</p>
    <p>``pandas_matplotlib-linreg.py``</p>
    <p>.. code-block:: python</p>
    <p>import pandas as pd</p>
    <p>import matplotlib.pyplot as plt</p>
    <p>from scipy.stats import linregress</p>
    <p>dataframe = pd.read_csv("scottish_hills.csv")</p>
    <p>x = dataframe.Height</p>
    <p>y = dataframe.Latitude</p>
    <p>stats = linregress(x, y)</p>
    <p>m = stats.slope</p>
    <p>b = stats.intercept</p>
    <p>plt.scatter(x, y)</p>
    <p>plt.plot(x, m * x + b, color="red")   # I've added a color argument here</p>
    <p>plt.show()</p>
    <p>``pandas_matplotlib-linreg-pretty.py``</p>
    <p>.. code-block:: python</p>
    <p>import pandas as pd</p>
    <p>import matplotlib.pyplot as plt</p>
    <p>from scipy.stats import linregress</p>
    <p>dataframe = pd.read_csv("scottish_hills.csv")</p>
    <p>x = dataframe.Height</p>
    <p>y = dataframe.Latitude</p>
    <p>stats = linregress(x, y)</p>
    <p>m = stats.slope</p>
    <p>b = stats.intercept</p>
    <p># Change the default figure size</p>
    <p>plt.figure(figsize=(10,10))</p>
    <p># Change the default marker for the scatter from circles to x's</p>
    <p>plt.scatter(x, y, marker='x')</p>
    <p># Set the linewidth on the regression line to 3px</p>
    <p>plt.plot(x, m * x + b, color="red", linewidth=3)</p>
    <p># Add x and y lables, and set their font size</p>
    <p>plt.xlabel("Height (m)", fontsize=20)</p>
    <p>plt.ylabel("Latitude", fontsize=20)</p>
    <p># Set the font size of the number lables on the axes</p>
    <p>plt.xticks(fontsize=18)</p>
    <p>plt.yticks(fontsize=18)</p>
    <p>plt.show()</p>
  </dd>
</dl>
<dl>
  <dt>.. keypoints :  : </dt>
  <dd>
    <p>- Before you can run Python scripts or work in a Python shell, first load a python module and probable prerequisites</p>
    <p>- Start a Python shell session either with ``python`` or ``ipython``</p>
    <p>- Run scripts with ``python3 <script.py>``</p>
    <p>- You can check for packages</p>
    <p></p>
    <p>- from the Python shell with the ``import`` command</p>
    <p>- from BASH shell with the</p>
    <p></p>
    <p>- ``pip list`` command at all three centers</p>
    <p>- ``ml help python/<version>`` at UPPMAX</p>
    <p></p>
    <p>- Installation of Python packages can be done either with **PYPI** or **Conda**</p>
    <p>- You install own packages with the ``pip install`` command (This is the recommended way on HPC2N)</p>
    <p>- At UPPMAX, LUNARC, and NSC Conda is also available (See Conda section)</p>
  </dd>
</dl>
