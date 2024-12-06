Summary day2
==============

`Summary of first day <docs/summary1.html>`_

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





    
