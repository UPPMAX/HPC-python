Using GPUs with Python
======================

Not every Python program is suitable for GPU acceleration. GPUs processes simple functions very fast, and are best suited for repetitive and highly-parallel computing tasks. 

GPUs are originally designed to render high-resolution images and video concurrently and fast, but since they can perform parallel operations on multiple sets of data, they are also often used for other, non-graphical tasks. Common uses are machine learning and scientific computation were the GPUs can take advantage of massive parallelism. 

Many Python packages are not CUDA aware, but some have been written specifically with GPUs in mind. 

If you are usually working with for instance NumPy and SciPy, you could optimize your code for GPU computing by using CuPy which mimics most of the NumPy functions. Another option is using Numba, which has bindings to CUDA and lets you write CUDA kernels in Python yourself. This means you can use custom algorithms. 

One of the most common use of GPUs with Python is for machine learning or deep learning. For these cases you would use something like Tensorflow or PyTorch - libraries which can handle CPU and GPU processing internally without the programmer needing to do so. 

Numba example
-------------

Numba is installed as a module at HPC2N, but not in a version compatible with the Python we are using in this course (3.9.5), so we will have to install it ourselves. The process is the same as in the examples given for the isolated/virtual environment, and we will be using the virtual environment created earlier here. We also need numpy, so we are loading SciPy-bundle as we have done before: 

.. admonition:: Load Python 3.9.5 and its prerequisites + SciPy-bundle + CUDA, then activate the virtual environment before installing numba 
    :class: dropdown
   
        .. code-block:: sh
      
             b-an01 [/proj/nobackup/support-hpc2n/bbrydsoe]$ module load GCC/10.3.0 OpenMPI/4.1.1 Python/3.9.5 SciPy-bundle/2021.05 CUDA/11.3.1
             b-an01 [/proj/nobackup/support-hpc2n/bbrydsoe]$ source /proj/nobackup/support-hpc2n/bbrydsoe/vpyenv/bin/activate 
             (vpyenv) b-an01 [/proj/nobackup/support-hpc2n/bbrydsoe]$ pip install --no-cache-dir --no-build-isolation numba
             Collecting numba
               Downloading numba-0.56.0-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.5 MB)
                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.5/3.5 MB 38.7 MB/s eta 0:00:00
             Requirement already satisfied: setuptools in /pfs/proj/nobackup/fs/projnb10/support-hpc2n/bbrydsoe/vpyenv/lib/python3.9/site-packages (from numba) (63.1.0)
             Requirement already satisfied: numpy<1.23,>=1.18 in /cvmfs/ebsw.hpc2n.umu.se/amd64_ubuntu2004_bdw/software/SciPy-bundle/2021.05-foss-2021a/lib/python3.9/site-packages (from numba) (1.20.3)
             Collecting llvmlite<0.40,>=0.39.0dev0
               Downloading llvmlite-0.39.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (34.6 MB)
                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 34.6/34.6 MB 230.0 MB/s eta 0:00:00
             Installing collected packages: llvmlite, numba
             Successfully installed llvmlite-0.39.0 numba-0.56.0
           
             [notice] A new release of pip available: 22.1.2 -> 22.2.2
             [notice] To update, run: pip install --upgrade pip

We can ignore the comment about pip. The package was successfully installed. now let us try using it. We are going to use the following program for testing (it was taken from https://linuxhint.com/gpu-programming-python/ but there are also many great examples at https://numba.readthedocs.io/en/stable/cuda/examples.html): 

.. admonition:: Python example using Numba 
    :class: dropdown
   
        .. code-block:: python
        
             import numpy as np
             from timeit import default_timer as timer
             from numba import vectorize
             
             # This should be a substantially high value.
             NUM_ELEMENTS = 100000000
             
             # This is the CPU version.
             def vector_add_cpu(a, b):
               c = np.zeros(NUM_ELEMENTS, dtype=np.float32)
               for i in range(NUM_ELEMENTS):
                   c[i] = a[i] + b[i]
               return c
               
             # This is the GPU version. Note the @vectorize decorator. This tells
             # numba to turn this into a GPU vectorized function.
             @vectorize(["float32(float32, float32)"], target='cuda')
             def vector_add_gpu(a, b):
               return a + b;
 
             def main():
               a_source = np.ones(NUM_ELEMENTS, dtype=np.float32)
               b_source = np.ones(NUM_ELEMENTS, dtype=np.float32)
               
               # Time the CPU function
               start = timer()
               vector_add_cpu(a_source, b_source)
               vector_add_cpu_time = timer() - start
 
               # Time the GPU function
               start = timer()
               vector_add_gpu(a_source, b_source)
               vector_add_gpu_time = timer() - start
 
                # Report times
                print("CPU function took %f seconds." % vector_add_cpu_time)
                print("GPU function took %f seconds." % vector_add_gpu_time)
              
                return 0
 
             if __name__ == "__main__":
               main()
                 
As before, we need a batch script to run the code. There are no GPUs on the login node. 

.. admonition:: Batch script to run the numba code (add-list.py) at Kebnekaise 
    :class: dropdown
   
        .. code-block:: sh

            #!/bin/bash
            # Remember to change this to your own project ID after the course!
            #SBATCH -A SNIC2022-22-641
            # We are asking for 5 minutes
            #SBATCH --time=00:05:00
            # Asking for one K80
            #SBATCH --gres=gpu:k80:1

            # Remove any loaded modules and load the ones we need
            module purge  > /dev/null 2>&1
            module load GCC/10.3.0  OpenMPI/4.1.1 Python/3.9.5 SciPy-bundle/2021.05 CUDA/11.3.1

            # Activate the virtual environment we installed to
            source /proj/nobackup/support-hpc2n/bbrydsoe/vpyenv/bin/activate

            # Run your Python script
            python add-list.py


As before, submit with ``sbatch add-list.sh`` (assuming you called the batch script thus - change to fit your own naming style). 
