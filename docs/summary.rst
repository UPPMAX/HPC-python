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
            - ``ml help python/3.9.5`` at UPPMAX
            - ``module -r spider '.*Python.*'`` at HPC2N
     
   - Install packages and use isolated environments 
      - With a virtual environment you can tailor an environment with specific versions for Python and packages, not interfering with other installed python versions and packages.
      - Make it for each project you have for reproducibility.
      - There are different tools to create virtual environemnts.
         - ``virtualenv`` and ``venv``
            - install packages with ``pip``.
            - the flag ``--system-site-packages`` includes preinstalled packages as well
      - (At UPPMAX Conda is also available)
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
         - At UPPMAX: ``interactive`` ...
      - Follow the same procedure as usual by loading the Python module and possible prerequisites.

   - Parallel
      - You deploy cores and nodes via SLURM, either in interactive mode or batch
      - In Python, threads, distributed and MPI parallelization and DASK can be used.

   - GPUs
      -  You deploy GPU nodes via SLURM, either in interactive mode or batch
      -  In Python the numba package is handy

   - Machine Learning
      - At all clusters you will find PyTorch, TensorFlow, Scikit-learn
      - The loading are slightly different at the clusters
         - UPPMAX: All tools are available from the module ``python_ML_packages``
         - HPC2N: ``module load GCC/10.3.0  OpenMPI/4.1.1 TensorFlow/2.6.0-CUDA-11.3.1``


.. seealso::

    - `Python documentation <https://www.python.org/doc/>`_. 
    - `Python forum <https://python-forum.io/>`_.
    - `Introduction to packages <https://aaltoscicomp.github.io/python-for-scicomp/dependencies/>`_
    - `CodeRefinery lessons <https://coderefinery.org/lessons/>`_
    - `A workshop more devoted to packages and Conda on UPPMAX <https://uppmax.github.io/R-python-julia-HPC/>`_

.. note::
    
    - Julia language becomes increasingly popular.
    - `Julia at UPPMAX <https://uppmax.uu.se/support/user-guides/julia-user-guide/>`_
    - `Julia at HPC2N <https://www.hpc2n.umu.se/resources/software/julia>`_





    
