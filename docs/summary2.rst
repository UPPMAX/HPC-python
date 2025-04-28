Summary day 2
=============


`Summary of first day <./summary3.html>`_
`Summary of first day <./summary4.html>`_


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





    

.. keypoints::

   - Intro to Pandas

       - Lets you construct list- or table-like data structures with mixed data types, the contents of which can be indexed by arbitrary row and column labels
       - The main data structures are Series (1D) and DataFrames (2D). Each column of a DataFrame is a Series

   - Seaborn
       - Seaborn makes statistical plots easy and good-looking!

       - Seaborn plotting functions take in a Pandas DataFrame, sometimes the names of variables in the DataFrame to extract as x and y, and often a hue that makes different subsets of the data appear in different colors depending on the value of the given categorical variable.

   - Parallel
      - You deploy cores and nodes via SLURM, either in interactive mode or batch
      - In Python, threads, distributed and MPI parallelization and DASK can be used.

   - Big data

       - allocate resources sufficient to data size
       - decide on useful file formats
       - use data-chunking as technique

   - Machine Learning

      - General overview of ML/DL with Python.
      - General overview of installed ML/DL tools at HPC2N, UPPMAX, and LUNARC.
      - Get started with ML/DL in Python.
      - At all clusters you will find PyTorch, TensorFlow, Scikit-learn
      - The loading are slightly different at the clusters
         - UPPMAX: All tools are available from the module ``python_ML_packages/3.11.8``
         - HPC2N: 
            - For TensorFlow ``ml GCC/12.3.0  OpenMPI/4.1.5 TensorFlow/2.15.1-CUDA-12.1.1 scikit-learn/1.4.2 Tkinter/3.11.3 matplotlib/3.7.2``
            - For the Pytorch: ``ml GCC/12.3.0  OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1 scikit-learn/1.4.2 Tkinter/3.11.3 matplotlib/3.7.2``
         - LUNARC:
            - For TensorFlow ``module load GCC/11.3.0 Python/3.10.4 SciPy-bundle/2022.05 TensorFlow/2.11.0-CUDA-11.7.0 scikit-learn/1.1.2``
            - For Pytorch ``module load GCC/11.3.0 Python/3.10.4 SciPy-bundle/2022.05 PyTorch/1.12.1-CUDA-11.7.0 scikit-learn/1.1.2``
         - NSC: For Tetralith, use virtual environment. Pytorch and TensorFlow might coming soon to the cluster!

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





    
