Summary
==============

.. keypoints::

   - Load and run and pre-installed packages
      - Use python from module system
      - Start a Python shell session either with ``python`` or ``ipython``
      - run scripts with ``python3 <script.py>``
      - Check for preinstalled packages
         - from the Python shell with the ``import`` command
         - from BASH shell with the
            - ``pip list`` command at both centers
            - ``ml help python/3.11.8`` at UPPMAX
            - ``module -r spider '.*Python.*'`` at otherwise
     
   - Install packages and use isolated environments 
      - With a virtual environment you can tailor an environment with specific versions for Python and packages, not interfering with other installed python versions and packages.
      - Make it for each project you have for reproducibility.
      - There are different tools to create virtual environemnts.
         - ``virtualenv`` and ``venv``
            - install packages with ``pip``.
            - the flag ``--system-site-packages`` includes preinstalled packages as well
      - Conda (available everywhere but not HPC2N)
         - Conda is an installer of packages but also bigger toolkits
             - Conda creates isolated environments as well
             - requires that you install all packages needed. 
         - Rackham: Pip or secondary conda
         - Bianca: conda and secondary wharf + (pip or conda)

   - Batch mode
      - The SLURM scheduler handles allocations to the calculation nodes
      - Batch jobs runs without interaction with user
      - A batch script consists of a part with *SLURM parameters* describing the allocation and a second part describing the actual work within the job, for instance one or several Python scripts.
      - Remember to include possible input arguments to the Python script in the batch script.
   
   - Interactive work on calculation nodes
      - Start an interactive session on a calculation node by a SLURM allocation (similar flags)
         - At HPC2N: ``salloc`` ...
         - At UPPMAX/NSC: ``interactive`` ...
         - At LUNARC: Desktop on demand
      - Follow the same procedure as usual by loading the Python module and possible prerequisites.

   - IDEs
       - Jupyter-lab/notebook
           - Available in all clusters
       - Spyder
           - Best available at LUNARC
           - Possible at the others through virtual environments (pip) or Conda (not HPC2N)
       - VScode: perhaps shown tomorrow

   - Matplotlib
       - Matplotlib is the essential Python data visualization package, with nearly 40 different plot types to choose from depending on the shape of your data and which qualities you want to highlight.
       - Almost every plot will start by instantiating the figure, ``fig`` (the blank canvas), and 1 or more ``axes`` objects, ``ax``, with ``fig, ax = plt.subplots(*args, **kwargs)``.
       - Most of the plotting and formatting commands you will use are methods of Axes objects, but a few, like colorbar are methods of the Figure, and some commands are methods both.

   - GPUs
       - You deploy GPU nodes via SLURM, either in interactive mode or batch
       - In Python the numba package is handy

Reference `Summary of second day <./summary2.rst_>`_
`Summary of second day <./summary2.html>`_


.. seealso::

    - `Python documentation <https://www.python.org/doc/>`_. 
    - `Python forum <https://python-forum.io/>`_.
    - `Introduction to packages <https://aaltoscicomp.github.io/python-for-scicomp/dependencies/>`_
    - `CodeRefinery lessons <https://coderefinery.org/lessons/>`_
    - `A workshop more devoted to packages and Conda on UPPMAX <https://uppmax.github.io/R-python-julia-matlab-HPC/>`_

.. note::
    
    - Julia language becomes increasingly popular.
    - `Julia at UPPMAX <https://docs.uppmax.uu.se/software/julia/>`_
    - `Julia at HPC2N <https://www.hpc2n.umu.se/resources/software/julia>`_





    
