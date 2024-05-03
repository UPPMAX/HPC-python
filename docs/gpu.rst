Using GPUs with Python
======================

.. questions::

   - What is GPU acceleration?
   - How to enable GPUs (for instance with CUDA) in Python code?
   - How to deploy GPUs at HPC2N and UPPMAX?
   
   

.. objectives::

   - Learn common schemes for GPU code acceleration
   - Learn about the GPU nodes at HPC2N and UPPMAX


In order to understand the capabilities of a GPU, it is instructive to compare a pure CPU
architecture with a GPU based architecture. Here, there is a schemematics of the former:

.. figure:: img/cpus.png
   :align: center

   Pure CPU architecture (single node). In the present case there are 28 cores, each with 
   its own cache memory (LX). There is a shared memory (64 GB/NUMA node) for all these cores.
   The base frequency for each core is 2.6 GHz.

As for the GPU architecture, a K80 engine looks like this:

.. figure:: img/gpu.png
   :align: center

   A single GPU engine of a K80 card. Each green dot represents a core (single precision) which
   runs at a frequency of 562 MHz. The cores are arranged in slots called streaming multiprocessors (SMX)
   in the figure. Cores in the same SMX share some local and fast cache memory.

In a typical cluster, some GPUs are attached to a single node resulting in a CPU-GPU
hybrid architecture. The CPU component is called the host and the GPU part the device.
One possible layout (Kebnekaise) is as follows:


.. figure:: img/cpu-gpu.png
   :align: center

   Schematics of a hybrid CPU-GPU architecture. A GPU K80 card consisting of two engines is attached
   to a NUMA island which in turn contains 14 cores. The NUMA island and the GPUs are
   connected through a PCI-E interconnect which makes the data transfer between both components rather
   slow.

We can characterize the CPU and GPU performance with two quantities: the **latency** and the **througput**.
**Latency** refers to the time spent in a sole computation. **Throughput** denotes the number of 
computations that can be performed in parallel. Then, we can say that a CPU has low latency
(able to do fast computations) but low throughput (only a few computations simultaneously).
In the case of GPUs, the latency is high and the throughput is also high. We can visualize the behavior
of the CPUs and GPUs with cars as in the figure below. A CPU would be compact road where only a few 
racing cars can drive whereas a GPU would be a broader road where plenty of slow cars can drive.


.. figure:: img/cpu-gpu-highway.png
   :align: center

   Cars and roads analogy for the CPU and GPU behavior. The compact road is analogous to the CPU
   (low latency, low throughput) and the broader road is analogous to the GPU (high latency, high throughput).




Not every Python program is suitable for GPU acceleration. GPUs process simple functions rapidly, 
and are best suited for repetitive and highly-parallel computing tasks. GPUs were originally 
designed to render high-resolution images and video concurrently and fast, but since they can 
perform parallel operations on multiple sets of data, they are also often used for other, 
non-graphical tasks. Common uses are machine learning and scientific computation were the GPUs can 
take advantage of massive parallelism. 

Many Python packages are not CUDA aware, but some have been written specifically with GPUs in mind. 
If you are usually working with for instance NumPy and SciPy, you could optimize your code for GPU 
computing by using CuPy which mimics most of the NumPy functions. Another option is using Numba, which 
has bindings to CUDA and lets you write CUDA kernels in Python yourself. This means you can
use custom algorithms. 

One of the most common use of GPUs with Python is for machine learning or deep learning. For 
these cases you would use something like Tensorflow or PyTorch libraries which can handle CPU
and GPU processing internally without the programmer needing to do so. 

Numba example
-------------

Numba is installed as a module at HPC2N, but not in a version compatible with the Python we 
are using in this course (3.9.6->HPC2N, 3.9.5->UPPMAX), so we will have to install it ourselves. The process is the same
as in the examples given for the isolated/virtual environment, and we will be using the virtual 
environment created earlier here. We also need numpy, so we are loading SciPy-bundle as we have done before: 

.. admonition::  Python 3.9.6 as the basis
    :class: dropdown
   
        Load Python 3.9.6 and its prerequisites + CUDA, then activate the virtual environment before installing numba
        Then remember to change the path to your own in the example below!
   
        .. code-block:: console
      
             $ module load GCCcore/11.2.0 Python/3.9.6 GCC/11.2.0 OpenMPI/4.1.1 CUDA/11.4.1
             $ virtualenv --system-site-packages /proj/nobackup/<your-project-directory>/vpyenv-gpu
             $ source /proj/nobackup/<your-project-directory>/vpyenv-gpu/bin/activate 
             (vpyenv) $ pip install --no-cache-dir --no-build-isolation numba

             Collecting numba
               Downloading numba-0.58.1-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.7 kB)
             Collecting llvmlite<0.42,>=0.41.0dev0 (from numba)
               Downloading llvmlite-0.41.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.8 kB)
             Collecting numpy<1.27,>=1.22 (from numba)
               Downloading numpy-1.26.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
                  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.2/61.2 kB 7.9 MB/s eta 0:00:00
             Downloading numba-0.58.1-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.6 MB)
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.6/3.6 MB 81.0 MB/s eta 0:00:00
             Downloading llvmlite-0.41.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (43.6 MB)
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 43.6/43.6 MB 254.1 MB/s eta 0:00:00
             Downloading numpy-1.26.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 215.6 MB/s eta 0:00:00
             Installing collected packages: numpy, llvmlite, numba
             Successfully installed llvmlite-0.41.1 numba-0.58.1 numpy-1.26.2

We can ignore the comment about pip. The package was successfully installed. now let us try using it. 
We are going to use the following program for testing (it was taken from 
https://linuxhint.com/gpu-programming-python/ but there are also many great examples at 
https://numba.readthedocs.io/en/stable/cuda/examples.html): 

.. admonition:: Python example ``add-list.py`` using Numba 
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

.. tabs::

   .. tab:: UPPMAX
      
      .. code-block:: console
      
         [bjornc@rackham3 ~]$ interactive -A naiss2024-22-415 -n 1 -M snowy --gres=gpu:1  -t 1:00:01 --mail-type=BEGIN --mail-user=bjorn.claremar@uppmax.uu.se
         You receive the high interactive priority.

         Please, use no more than 8 GB of RAM.

         Waiting for job 6907137 to start...
         Starting job now -- you waited for 90 seconds.

         [bjornc@s160 ~]$  ml python/3.9.5
         [bjornc@s160 ~]$ python add-list.py
         CPU function took 36.849201 seconds.
         GPU function took 1.574953 seconds.


   .. tab:: HPC2N
   
      Running a GPU Python code interactively. When you code-along, remember to change the activation path for the virtual environment to your own! 

      .. code-block:: console

         $ salloc -A hpc2nXXXX-YYY --time=00:30:00 -n 1 --gres=gpu:v100:1 
         salloc: Pending job allocation 20346979
         salloc: job 20346979 queued and waiting for resources
         salloc: job 20346979 has been allocated resources
         salloc: Granted job allocation 20346979
         salloc: Waiting for resource configuration
         salloc: Nodes b-cn1101 are ready for job
         $
         $ module load GCCcore/11.2.0 Python/3.9.6 GCC/11.2.0 OpenMPI/4.1.1 CUDA/11.4.1
         $ source /proj/nobackup/<your-project-directory>/vpyenv-gpu/bin/activate
         (vpyenv) b-an01$ srun python add-list.py
         CPU function took 31.905025 seconds.
         GPU function took 0.684060 seconds.

      Because this is a short job, you can also use this shortcut as we did in the ``Parallel`` session: 

      .. code-block:: console

         $ module load GCCcore/11.2.0 Python/3.9.6 GCC/11.2.0 OpenMPI/4.1.1 CUDA/11.4.1
         $ source /proj/nobackup/<your-project-directory>/vpyenv-gpu/bin/activate
         $ srun -A hpc2nXXXX-YYY -n 1 --gres=gpu:v100:1 -t 00:10:00 python add-list.py


   .. tab:: Batch script for HPC2N

      Batch script, ``add-list.sh``, to run the same GPU Python script (the numba code, ``add-list.py``) at Kebnekaise. 
      As before, submit with ``sbatch add-list.sh`` (assuming you called the batch script thus - change to fit your own naming style). 
      
      .. code-block:: console

          #!/bin/bash
          # Remember to change this to your own project ID after the course!
          #SBATCH -A hpc2nXXXX-YYY     # HPC2N ID - change to naiss2024-22-415 for UPPMAX
          # We are asking for 5 minutes
          #SBATCH --time=00:05:00
          # Asking for one v100
          #SBATCH --gres=gpu:v100:1     # For HPC2N. Remove if on UPPMAX
          ##SBATCH -M snowy            # For UPPMAX. Remove leading # to use
          ##SBATCH --gres=gpu:1        # For UPPMAX. Remove leading # to use

          # Remove any loaded modules and load the ones we need
          module purge  > /dev/null 2>&1
          module load GCCcore/11.2.0 Python/3.9.6 GCC/11.2.0 OpenMPI/4.1.1 CUDA/11.4.1

          # Activate the virtual environment we installed to
          source /proj/nobackup/support-hpc2n/bbrydsoe/vpyenv/bin/activate

          # Run your Python script
          python add-list.py


Exercises
---------

.. challenge:: Integration 2D with Numba

   An initial implementation of the 2D integration problem with the CUDA support for 
   Numba could be as follows:

   .. admonition:: ``integration2d_gpu.py``
      :class: dropdown

      .. code-block:: python

         from __future__ import division
         from numba import cuda, float32
         import numpy
         import math
         from time import perf_counter
         
         # grid size
         n = 100*1024
         threadsPerBlock = 16
         blocksPerGrid = int((n+threadsPerBlock-1)/threadsPerBlock)
         
         # interval size (same for X and Y)
         h = math.pi / float(n)
         
         @cuda.jit
         def dotprod(C):
             tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x 
         
             if tid >= n:
                 return
         
             #cummulative variable
             mysum = 0.0
             # fine-grain integration in the X axis
             x = h * (tid + 0.5)
             # regular integration in the Y axis
             for j in range(n):
                 y = h * (j + 0.5)
                 mysum += math.sin(x + y)
         
             C[tid] = mysum
         
         
         # array for collecting partial sums on the device
         C_global_mem = cuda.device_array((n),dtype=numpy.float32)
         
         starttime = perf_counter()
         dotprod[blocksPerGrid,threadsPerBlock](C_global_mem)
         res = C_global_mem.copy_to_host()
         integral = h**2 * sum(res)
         endtime = perf_counter()
         
         print("Integral value is %e, Error is %e" % (integral, abs(integral - 0.0)))
         print("Time spent: %.2f sec" % (endtime-starttime))



   Notice the larger size of the grid in the present case (100*1024) compared
   to the serial case's size we used previously (10000). Large computations are 
   necessary on the GPUs to get the benefits of this architecture. 

   One can take advantage of the shared memory in a thread block to write faster 
   code. Here, we wrote the 2D integration example from the previous section where 
   threads in a block write on a `shared[]` array. Then, this array is reduced 
   (values added) and the output is collected in the array ``C``. The entire code 
   is here:


   .. admonition:: ``integration2d_gpu_shared.py``
      :class: dropdown

      .. code-block:: python

         from __future__ import division
         from numba import cuda, float32
         import numpy
         import math
         from time import perf_counter
         
         # grid size
         n = 100*1024
         threadsPerBlock = 16
         blocksPerGrid = int((n+threadsPerBlock-1)/threadsPerBlock)
         
         # interval size (same for X and Y)
         h = math.pi / float(n)
         
         @cuda.jit
         def dotprod(C):
             # using the shared memory in the thread block
             shared = cuda.shared.array(shape=(threadsPerBlock), dtype=float32) 
         
             tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x 
             shrIndx = cuda.threadIdx.x
         
             if tid >= n:
                 return
         
             #cummulative variable
             mysum = 0.0
             # fine-grain integration in the X axis
             x = h * (tid + 0.5)
             # regular integration in the Y axis
             for j in range(n):
                 y = h * (j + 0.5)
                 mysum += math.sin(x + y)
         
             shared[shrIndx] = mysum
         
             cuda.syncthreads()
         
             # reduction for the whole thread block
             s = 1
             while s < cuda.blockDim.x:
                 if shrIndx % (2*s) == 0:
                     shared[shrIndx] += shared[shrIndx + s]
                 s *= 2
                 cuda.syncthreads()
             # collecting the reduced value in the C array
             if shrIndx == 0:
                 C[cuda.blockIdx.x] = shared[0]
         
         # array for collecting partial sums on the device
         C_global_mem = cuda.device_array((blocksPerGrid),dtype=numpy.float32)
         
         starttime = perf_counter()
         dotprod[blocksPerGrid,threadsPerBlock](C_global_mem)
         res = C_global_mem.copy_to_host()
         integral = h**2 * sum(res)
         endtime = perf_counter()
         
         print("Integral value is %e, Error is %e" % (integral, abs(integral - 0.0)))
         print("Time spent: %.2f sec" % (endtime-starttime))

   Prepare a batch script to run these two versions of the integration 2D with Numba support
   and monitor the timings for both cases.

.. solution:: Solution for HPC2N
    :class: dropdown

     A template for running the python codes at HPC2N is here:

     .. admonition:: ``job-gpu.sh``
        :class: dropdown
      
         .. code-block:: console 

            #!/bin/bash
            # Remember to change this to your own project ID after the course!
            #SBATCH -A hpc2n20XX-XYZ
            #SBATCH -t 00:08:00
            #SBATCH -N 1
            #SBATCH -n 28
            #SBATCH -o output_%j.out   # output file
            #SBATCH -e error_%j.err    # error messages
            #SBATCH --gres=gpu:v100:2
            #SBATCH --exclusive
     
            ml purge > /dev/null 2>&1
            ml GCCcore/11.2.0 Python/3.9.6
            ml GCC/11.2.0 OpenMPI/4.1.1
            ml CUDA/11.4.1
    
            # CHANGE TO YOUR OWN PATH! 
            source /proj/nobackup/<your-project-storage>/vpyenv-python-course/bin/activate
       
            python integration2d_gpu.py
            python integration2d_gpu_shared.py

     For the ``integration2d_gpu.py`` implementation, the time for executing the kernel 
     and doing some postprocessing to the outputs (copying the C array and doing a reduction)  
     was 4.35 sec. which is a much smaller value than the time for the serial numba code of 152 sec
     obtained previously. 

     The simulation time for the ``integration2d_shared.py`` implementation was 1.87 sec. 
     by using the shared memory trick. 

.. keypoints::

   -  You deploy GPU nodes via SLURM, either in interactive mode or batch
   -  In Python the numba package is handy


Additional information
----------------------
         
* `Numba documentation examples <http://numba.pydata.org/numba-doc/latest/cuda/examples.html>`_
* `New York University CUDA/Numba lesson  <https://nyu-cds.github.io/python-numba/05-cuda/>`_
* Hands-On GPU Programming with Python and CUDA : Explore High-Performance Parallel Computing with CUDA, Brian Tuomanen. Packt publishing.
* Parallel and High Performance Computing, Robert Robey and Yuliana Zamora. Manning publishing.
