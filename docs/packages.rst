Packages
========

- Python **packages broaden the use of python** to almost infinity! 

- Instead of writing code yourself there may be others that have done the same!

- Many **scientific tools** are distributed as **python package**'s making it possible to run a script in the prompt and there define files to be analysed and arguments defining exactly what to do.

- A nice **introduction to packages** can be found here: https://aaltoscicomp.github.io/python-for-scicomp/dependencies/ 

.. admonition:: There are two package installation systems

    - **PyPI** (``pip``) is traditionally for Python-only packages but it is no problem to also distribute packages written in other languages as long as they provide a Python interface.

    - **Conda** (``conda``) is more general and while it contains many Python packages and packages with a Python interface, it is often used to also distribute packages which do not contain any Python (e.g. C or C++ packages).
    	- Creates its own environment that does not interact with other python installations
	- At HPC2N, Conda is not recommended, and we do not support it there

    - Many libraries and tools are distributed in both ecosystems.


Check current available packages
--------------------------------

Some python packages are working as stand-alone tools, for instance in bioinformatics. The tool may be already installed as a module. Check if it is there by:

.. code-block:: sh 

   $ module spider <tool-name or tool-name part> 
    
Using ``module spider`` lets you search regardless of upper- or lowercase characters.

Another way to find Python packages that you are unsure how are names, would be to do

.. code-block:: sh 

   $ module -r spider ’.*Python.*’
   
or

.. code-block:: sh 

   $ module -r spider ’.*python.*’
   
Do be aware that the output of this will not just be Python packages, some will just be programs that are compiled with Python, so you need to check the list carefully.   

**UPPMAX only!**

Check the pre-installed packages of a specific python module:

.. code-block:: sh 

   $ module help python/<version> 
  
or with python module loaded (more certain), in shell:

.. code-block:: sh 

   $ pip list

You can also test from within python to make sure that the package is not already installed:

.. code-block:: python 

    >>> import <package>
    
Does it work? Then it is there!
Otherwise, you can either use ``pip`` or ``conda``.

**NOTE**: at HPC2N, the available Python packages needs to be loaded as modules before using! See a list of some of them here: https://uppmax.github.io/HPC-python/intro.html#python-at-hpc2n or find more as mentioned above, using ``module spider -r ....```

Install with pip
----------------

You use ``pip`` this way, in a Linux shell OR a python shell: 

.. code-block:: sh 

    $ pip install –-user <package>
    
Use ``pip3`` if you loaded python3.

Then the package ends up in ~/.local/lib/python<version>/site-packages/ .

At HPC2N we HIGHLY recommend using a virtual environment during installation, since this makes it easier to install for different versions of Python. more information will follow later in this course (https://uppmax.github.io/HPC-python/isolated.html). 

