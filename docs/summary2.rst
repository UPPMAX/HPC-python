Summary day2
==============

.. keypoints::

   - Parallel
      - You deploy cores and nodes via SLURM, either in interactive mode or batch
      - In Python, threads, distributed and MPI parallelization and DASK can be used.

   - Machine Learning
      - At all clusters you will find PyTorch, TensorFlow, Scikit-learn
      - The loading are slightly different at the clusters
         - UPPMAX: All tools are available from the module ``python_ML_packages/3.11.8``
         - HPC2N: 
            - For TensorFlow ``ml GCC/12.3.0  OpenMPI/4.1.5 TensorFlow/2.15.1-CUDA-12.1.1 scikit-learn/1.4.2 Tkinter/3.11.3 matplotlib/3.7.2``
            - For the rest: ``ml GCC/12.3.0  OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1 scikit-learn/1.4.2 Tkinter/3.11.3 matplotlib/3.7.2``
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





    
