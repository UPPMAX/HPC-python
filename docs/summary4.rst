Summary day 4
=============

`Summary of first day <./summary1.html>`_
`Summary of first day <./summary2.html>`_
`Summary of first day <./summary3.html>`_

.. keypoints::

   - Parallel
      - You deploy cores and nodes via SLURM, either in interactive mode or batch
      - In Python, threads, distributed and MPI parallelization and DASK can be used.

   - GPUs
       - You deploy GPU nodes via SLURM, either in interactive mode or batch
       - In Python the numba package is handy

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






    
