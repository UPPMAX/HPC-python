Parallel computing with Python
==============================

.. questions::

   - What is parallel computing?
   - What are the different parallelization mechanisms for Python?
   - How to implement parallel in Python code?
   - How to deploy threads and workers at HPC2N and UPPMAX?
  

.. objectives::

   - Learn common schemes for the parallelization of codes
   - Learn general concepts for parallel computing



**Prerequisites**

- Code along!

.. tabs::

   .. tab:: HPC2N
      
      ml GCCcore/11.2.0 Python/3.9.6

      virtualenv --system-site-packages /proj/nobackup/<your-project-storage>/vpyenv-python-course

      source /proj/nobackup/<your-project-storage>/vpyenv-python-course/bin/activate

      - For the ``numba`` example install the corresponding module:

        python -m pip install numba

      - For the ``mpi4py`` example add the following modules:

        ml GCC/11.2.0 OpenMPI/4.1.1

        python -m pip install mpi4py

      - For the ``f2py`` example, ``f2py3.9`` should be available on the terminal when ``numpy`` is installed:

        python -m pip install numpy

      - For the Julia example we will need PyJulia:
        
        ml Julia/1.8.5-linux-x86_64

        python -m pip install julia

        Start Python on the command line and type:

        .. code-block:: python

           >>> import julia
           >>> julia.install()

        Quit Python, you should be ready to go!

   .. tab:: UPPMAX

      If not already done so:
      
      .. code-block:: sh

         $ module load python/3.9.5
         $ python -m venv --system-site-packages /proj/naiss202X-XY-XYZ/nobackup/<user>/venv-python-course
    
      Activate it if needed (is the name shown in the prompt)

      .. code-block:: sh

         $ source /proj/naiss202X-XY-XYZ/nobackup/<user>/venv-python-course/bin/activate

      - For the ``numba`` example install the corresponding module:

        python -m pip install numba

      - For the ``mpi4py`` example add the following modules:

      .. code-block:: sh

         $ ml gcc/9.3.0 openmpi/3.1.5
         $ python -m pip install mpi4py

      - For the Julia example we will need PyJulia:
        
      .. code-block:: sh
       
         $ ml julia/1.7.2
         $ python -m pip install julia

      Start Python on the command line and type:

      .. code-block:: python
       
         >>> import julia
         >>> julia.install()
      Quit Python, you should be ready to go!

Parallelization mechanisms in Python
------------------------------------

In Python there are different schemes that can be used to parallelize your code. 
We will only take a look at some of these schemes that illustrate the general concepts of
parallel computing. The aim of this lecture is to learn how to run parallel codes
in Python rather than learning to write those codes.

The workhorse for this section will be a 2D integration example:

   .. math:: 
       \int^{\pi}_{0}\int^{\pi}_{0}\sin(x+y)dxdy = 0

One way to perform the integration is by creating a grid in the ``x`` and ``y`` directions.
More specifically, one divides the integration range in both directions into ``n`` bins. A
serial code (without optimization) can be seen in the following code block.

   .. admonition:: ``integration2d_serial.py``
      :class: dropdown

      .. code-block:: python

         import math
         import sys
         from time import perf_counter
         
         # grid size
         n = 10000
         
         def integration2d_serial(n):
             global integral;
             # interval size (same for X and Y)
             h = math.pi / float(n)
             # cummulative variable 
             mysum = 0.0
             
             # regular integration in the X axis
             for i in range(n):
                 x = h * (i + 0.5)
                 # regular integration in the Y axis
                 for j in range(n):
                     y = h * (j + 0.5)
                     mysum += math.sin(x + y)
             
             integral = h**2 * mysum
         
         
         if __name__ == "__main__":
         
             starttime = perf_counter()
             integration2d_serial(n)
             endtime = perf_counter()
         
         print("Integral value is %e, Error is %e" % (integral, abs(integral - 0.0)))
         print("Time spent: %.2f sec" % (endtime-starttime))

We can run this code on the terminal as follows (similarly at both HPC2N and UPPMAX): 


.. code-block:: sh 

    $ python integration2d_serial.py
    Integral value is -7.117752e-17, Error is 7.117752e-17
    Time spent: 20.39 sec

Note that outputs can be different, when timing a code a more realistic approach
would be to run it several times to get statistics.

One of the crucial steps upon parallelizing a code is identifying its bottlenecks. In
the present case, we notice that the most expensive part in this code is the double `for loop`. 
Just before we jump into a parallelization project, Python offers some options to make
serial code faster. For instance, the ``Numba`` module can assist you to obtain a 
compiled-quality function with minimal efforts. This can be achieved with the ``njit()`` 
decorator: 

   .. admonition:: ``integration2d_serial_numba.py``
      :class: dropdown

      .. code-block:: python

         from numba import njit
         import math
         import sys
         from time import perf_counter
         
         # grid size
         n = 10000
         
         def integration2d_serial(n):
             # interval size (same for X and Y)
             h = math.pi / float(n)
             # cummulative variable 
             mysum = 0.0
             
             # regular integration in the X axis
             for i in range(n):
                 x = h * (i + 0.5)
                 # regular integration in the Y axis
                 for j in range(n):
                     y = h * (j + 0.5)
                     mysum += math.sin(x + y)
             
             integral = h**2 * mysum
             return integral
         
         
         if __name__ == "__main__":
         
             starttime = perf_counter()
             integral = njit(integration2d_serial)(n)
             endtime = perf_counter()
         
         print("Integral value is %e, Error is %e" % (integral, abs(integral - 0.0)))
         print("Time spent: %.2f sec" % (endtime-starttime))

The execution time is now:

.. code-block:: sh 

    $ python integration2d_serial_numba.py
    Integral value is -7.117752e-17, Error is 7.117752e-17
    Time spent: 1.90 sec

Another option for making serial codes faster, and specially in the case of arithmetic 
intensive codes, is to write the most expensive parts of them in a compiled language such 
as Fortran or C/C++. In the next paragraphs we will show you how Fortran code for the 
2D integration case can be called in Python.

We start by writing the expensive part of our Python code in a Fortran function in a file
called ``fortran_function.f90``:


   .. admonition:: ``fortran_function.f90``
      :class: dropdown

      .. code-block:: fortran

         function integration2d_fortran(n) result(integral)
             implicit none
             integer, parameter :: dp=selected_real_kind(15,9)
             real(kind=dp), parameter   :: pi=3.14159265358979323_dp
             integer, intent(in)        :: n
             real(kind=dp)              :: integral
         
             integer                    :: i,j
         !   interval size
             real(kind=dp)              :: h
         !   x and y variables
             real(kind=dp)              :: x,y
         !   cummulative variable
             real(kind=dp)              :: mysum
         
             h = pi/(1.0_dp * n)
             mysum = 0.0_dp
         !   regular integration in the X axis
             do i = 0, n-1
                x = h * (i + 0.5_dp)
         !      regular integration in the Y axis
                do j = 0, n-1
                    y = h * (j + 0.5_dp)
                    mysum = mysum + sin(x + y)
                enddo
             enddo
         
             integral = h*h*mysum
                     
         end function integration2d_fortran

Then, we need to compile this code and generate the Python module
(``myfunction``):

.. code-block:: sh 

    $ f2py3.9 -c -m myfunction fortran_function.f90  
    running build
    running config_cc
    ...

this will produce the Python/C API ``myfunction.cpython-39-x86_64-linux-gnu.so``, which 
can be called in Python as a module:


   .. admonition:: ``call_fortran_code.py``
      :class: dropdown

      .. code-block:: python

         from time import perf_counter
         import myfunction
         import numpy
         
         # grid size
         n = 10000
         
         if __name__ == "__main__":
         
             starttime = perf_counter()
             integral = myfunction.integration2d_fortran(n)
             endtime = perf_counter()
         
         print("Integral value is %e, Error is %e" % (integral, abs(integral - 0.0)))
         print("Time spent: %.2f sec" % (endtime-starttime))

The execution time is considerably reduced: 

.. code-block:: sh 

    $ python call_fortran_code.py
    Integral value is -7.117752e-17, Error is 7.117752e-17
    Time spent: 1.30 sec

Compilation of code can be tedious specially if you are in a developing phase of your code. As 
an alternative to improve the performance of expensive parts of your code (without using a 
compiled language) you can write these parts in Julia (which doesn't require compilation) and 
then calling Julia code in Python. For the workhorse integration case that we are using, 
the Julia code can look like this:

   .. admonition:: ``julia_function.jl``
      :class: dropdown

      .. code-block:: julia

         function integration2d_julia(n::Int)
         # interval size
           h = π/n
         # cummulative variable
           mysum = 0.0
         # regular integration in the X axis
           for i in 0:n-1
             x = h*(i+0.5)
         #   regular integration in the Y axis
             for j in 0:n-1
                y = h*(j + 0.5)
                mysum = mysum + sin(x+y)
             end
           end
           return mysum*h*h
         end


A caller script for Julia would be,


   .. admonition:: ``call_julia_code.py``
      :class: dropdown

      .. code-block:: python

         from time import perf_counter
         import julia
         from julia import Main
         
         Main.include('julia_function.jl')
         
         # grid size
         n = 10000
         
         if __name__ == "__main__":
         
             starttime = perf_counter()
             integral = Main.integration2d_julia(n)
             endtime = perf_counter()
         
         print("Integral value is %e, Error is %e" % (integral, abs(integral - 0.0)))
         print("Time spent: %.2f sec" % (endtime-starttime))

Timing in this case is similar to the Fortran serial case,

.. code-block:: sh 

    $ python call_julia_code.py
    Integral value is -7.117752e-17, Error is 7.117752e-17
    Time spent: 1.29 sec

If even with the previous (and possibly others from your own) serial optimizations your code
doesn't achieve the expected performance, you may start looking for some parallelization 
scheme. Here, we describe the most common schemes.  

Threads
-------

In a threaded parallelization scheme the workers (threads) share a global memory address space.
The `threading <https://docs.python.org/3/library/threading.html>`_ 
module is built into Python so you don't have to installed it. By using this
module, one can create several threads to do some work in parallel (in principle).
For jobs dealing with files I/O one can observe some speedup by using the `threading` module.
However, for CPU intensive jobs one would see a decrease in performance w.r.t. the serial code.
This is because Python uses the Global Interpreter Lock 
(`GIL <https://docs.python.org/3/c-api/init.html>`_) which serializes the code when 
several threads are used.

In the following code we used the `threading` module to parallelize the 2D integration example.
Threads are created with the construct ``threading.Thread(target=function, args=())``, where 
`target` is the function that will be executed by each thread and `args` is a tuple containing the
arguments of that function. Threads are started with the ``start()`` method and when they finish
their job they are joined with the ``join()`` method,

   .. admonition:: ``integration2d_threading.py``
      :class: dropdown

      .. code-block:: python

         import threading
         import math
         import sys
         from time import perf_counter
         
         # grid size
         n = 10000
         # number of threads
         numthreads = 4
         # partial sum for each thread
         partial_integrals = [None]*numthreads
         
         def integration2d_threading(n,numthreads,threadindex):
             global partial_integrals;
             # interval size (same for X and Y)
             h = math.pi / float(n)
             # cummulative variable 
             mysum = 0.0
             # workload for each thread
             workload = n/numthreads
             # lower and upper integration limits for each thread 
             begin = int(workload*threadindex)
             end = int(workload*(threadindex+1))
             # regular integration in the X axis
             for i in range(begin,end):
                 x = h * (i + 0.5)
                 # regular integration in the Y axis
                 for j in range(n):
                     y = h * (j + 0.5)
                     mysum += math.sin(x + y)
             
             partial_integrals[threadindex] = h**2 * mysum
         
         
         if __name__ == "__main__":
         
             starttime = perf_counter()
             # start the threads 
             threads = []
             for i in range(numthreads):
                 t = threading.Thread(target=integration2d_threading, args=(n,numthreads,i))
                 threads.append(t)
                 t.start()
         
             # waiting for the threads
             for t in threads:
                 t.join()
         
             integral = sum(partial_integrals)
             endtime = perf_counter()
         
         print("Integral value is %e, Error is %e" % (integral, abs(integral - 0.0)))
         print("Time spent: %.2f sec" % (endtime-starttime))


Notice the output of running this code on the terminal:

.. code-block:: sh 

    $ python integration2d_threading.py
    Integral value is 4.492851e-12, Error is 4.492851e-12
    Time spent: 21.29 sec

Although we are distributing the work on 4 threads, the execution time is longer than in the 
serial code. This is due to the GIL mentioned above.

Implicit Threaded 
-----------------

Some libraries like OpenBLAS, LAPACK, and MKL provide an implicit threading mechanism. They
are used, for instance, by ``numpy`` module for computing linear algebra operations. You can obtain information
about the libraries that are available in ``numpy`` with ``numpy.show_config()``.
This can be useful at the moment of setting the number of threads as these libraries could
use different mechanisms for it, for the following example we will use the OpenMP
environment variables.

Consider the following code that computes the dot product of a matrix with itself:

   .. admonition:: ``dot.py``
      :class: dropdown

      .. code-block:: python

         from time import perf_counter
         import numpy as np
         
         A = np.random.rand(3000,3000)
         starttime = perf_counter()
         B = np.dot(A,A)
         endtime = perf_counter()
         
         print("Time spent: %.2f sec" % (endtime-starttime))

the timing for running this code with 1 thread is:


.. code-block:: sh 

    $ export OMP_NUM_THREADS=1
    $ python dot.py
    Time spent: 1.14 sec

while running with 2 threads is:


.. code-block:: sh 

    $ export OMP_NUM_THREADS=2
    $ python dot.py
    Time spent: 0.60 sec

It is also possible to use efficient threads if you have blocks of code written
in a compiled language. Here, we will see the case of the Fortran code written above
where OpenMP threads are used. The parallelized code looks as follows:

   .. admonition:: ``fortran_function_openmp.f90``
      :class: dropdown

      .. code-block:: fortran

         function integration2d_fortran_openmp(n) result(integral)
             !$ use omp_lib
             implicit none
             integer, parameter :: dp=selected_real_kind(15,9)
             real(kind=dp), parameter   :: pi=3.14159265358979323
             integer, intent(in)        :: n
             real(kind=dp)              :: integral
         
             integer                    :: i,j
         !   interval size
             real(kind=dp)              :: h
         !   x and y variables
             real(kind=dp)              :: x,y
         !   cummulative variable
             real(kind=dp)              :: mysum
         
             h = pi/(1.0_dp * n)
             mysum = 0.0_dp
         !   regular integration in the X axis
         !$omp parallel do reduction(+:mysum) private(x,y,j)
             do i = 0, n-1
                x = h * (i + 0.5_dp)
         !      regular integration in the Y axis
                do j = 0, n-1
                    y = h * (j + 0.5_dp)
                    mysum = mysum + sin(x + y)
                enddo
             enddo
         !$omp end parallel do
         
             integral = h*h*mysum
                     
         end function integration2d_fortran_openmp

The way to compile this code differs to the one we saw before, now we will need the flags
for OpenMP:


.. code-block:: sh 

    $ f2py3.9 -c --f90flags='-fopenmp' -lgomp -m myfunction_openmp fortran_function_openmp.f90


the generated module can be then loaded,

   .. admonition:: ``call_fortran_code_openmp.py``
      :class: dropdown

      .. code-block:: python

         from time import perf_counter
         import myfunction_openmp
         import numpy
         
         # grid size
         n = 10000
         
         if __name__ == "__main__":
         
             starttime = perf_counter()
             integral = myfunction_openmp.integration2d_fortran_openmp(n)
             endtime = perf_counter()
         
         print("Integral value is %e, Error is %e" % (integral, abs(integral - 0.0)))
         print("Time spent: %.2f sec" % (endtime-starttime))

the execution time by using 4 threads is:

.. code-block:: sh 

    $ export OMP_NUM_THREADS=4
    $ python call_fortran_code_openmp.py
    Integral value is 4.492945e-12, Error is 4.492945e-12
    Time spent: 0.37 sec

More information about how OpenMP works can be found in the material of a previous
`OpenMP course <https://github.com/hpc2n/OpenMP-Collaboration>`_ offered by some of us.

Distributed
-----------

In the distributed parallelization scheme the workers (processes) can share some common
memory but they can also exchange information by sending and receiving messages for
instance.

   .. admonition:: ``integration2d_multiprocessing.py``
      :class: dropdown

      .. code-block:: python

         import multiprocessing
         from multiprocessing import Array
         import math
         import sys
         from time import perf_counter
         
         # grid size
         n = 10000
         # number of processes
         numprocesses = 4
         # partial sum for each thread
         partial_integrals = Array('d',[0]*numprocesses, lock=False)
         
         def integration2d_multiprocessing(n,numprocesses,processindex):
             global partial_integrals;
             # interval size (same for X and Y)
             h = math.pi / float(n)
             # cummulative variable 
             mysum = 0.0
             # workload for each process
             workload = n/numprocesses
         
             begin = int(workload*processindex)
             end = int(workload*(processindex+1))
             # regular integration in the X axis
             for i in range(begin,end):
                 x = h * (i + 0.5)
                 # regular integration in the Y axis
                 for j in range(n):
                     y = h * (j + 0.5)
                     mysum += math.sin(x + y)
             
             partial_integrals[processindex] = h**2 * mysum
         
         
         if __name__ == "__main__":
         
             starttime = perf_counter()
             
             processes = []
             for i in range(numprocesses):
                 p = multiprocessing.Process(target=integration2d_multiprocessing, args=(n,numprocesses,i))
                 processes.append(p)
                 p.start()
         
             # waiting for the processes
             for p in processes:
                 p.join()
         
             integral = sum(partial_integrals)
             endtime = perf_counter()
         
         print("Integral value is %e, Error is %e" % (integral, abs(integral - 0.0)))
         print("Time spent: %.2f sec" % (endtime-starttime))

In this case, the execution time is reduced:

.. code-block:: sh 

    $ python integration2d_multiprocessing.py
    Integral value is 4.492851e-12, Error is 4.492851e-12
    Time spent: 6.06 sec

MPI
---

More details for the MPI parallelization scheme in Python can be found in a previous
`MPI course <https://github.com/SNIC-MPI-course/MPI-course>`_ offered by some of us.

   .. admonition:: ``integration2d_mpi.py``
      :class: dropdown

      .. code-block:: python

         from mpi4py import MPI
         import math
         import sys
         from time import perf_counter
         
         # MPI communicator
         comm = MPI.COMM_WORLD
         # MPI size of communicator
         numprocs = comm.Get_size()
         # MPI rank of each process
         myrank = comm.Get_rank()
         
         # grid size
         n = 10000
         
         def integration2d_mpi(n,numprocs,myrank):
             # interval size (same for X and Y)
             h = math.pi / float(n)
             # cummulative variable 
             mysum = 0.0
             # workload for each process
             workload = n/numprocs
         
             begin = int(workload*myrank)
             end = int(workload*(myrank+1))
             # regular integration in the X axis
             for i in range(begin,end):
                 x = h * (i + 0.5)
                 # regular integration in the Y axis
                 for j in range(n):
                     y = h * (j + 0.5)
                     mysum += math.sin(x + y)
             
             partial_integrals = h**2 * mysum
             return partial_integrals
         
         
         if __name__ == "__main__":
         
             starttime = perf_counter()
             
             p = integration2d_mpi(n,numprocs,myrank)
         
             # MPI reduction
             integral = comm.reduce(p, op=MPI.SUM, root=0)
         
             endtime = perf_counter()
         
         if myrank == 0:
             print("Integral value is %e, Error is %e" % (integral, abs(integral - 0.0)))
             print("Time spent: %.2f sec" % (endtime-starttime))


Execution of this code gives the following output:

.. code-block:: sh 

    $ mpirun -np 4 python integration2d_mpi.py
    Integral value is 4.492851e-12, Error is 4.492851e-12
    Time spent: 5.76 sec

For long jobs, one will need to run in batch mode. Here is an example of a batch script for this MPI
example,

.. tabs::

   .. tab:: HPC2N

      .. code-block:: sh 

         #!/bin/bash
         #SBATCH -A hpc2n20XX-XYZ
         #SBATCH -t 00:05:00
         #SBATCH -n 4
         #SBATCH -o output_%j.out   # output file
         #SBATCH -e error_%j.err    # error messages
     
         ml purge > /dev/null 2>&1
         ml GCCcore/11.2.0 Python/3.9.6
         ml GCC/11.2.0 OpenMPI/4.1.1
         #ml Julia/1.7.1-linux-x86_64  # if Julia is needed
      
         source /proj/nobackup/<your-project-storage>/vpyenv-python-course/bin/activate
       
         mpirun -np 4 python integration2d_mpi.py

   .. tab:: UPPMAX

      .. code-block:: sh 

         #!/bin/bash
         #SBATCH -A naiss202X-XY-XYZ
         #SBATCH -t 00:05:00
         #SBATCH -n 4
         #SBATCH -o output_%j.out   # output file
         #SBATCH -e error_%j.err    # error messages
     
         ml python/3.9.5
         ml gcc/9.3.0 openmpi/3.1.5
         #ml julia/1.7.2  # if Julia is needed
      
         source /proj/naiss202X-XY-XYZ/nobackup/<user>/venv-python-course/bin/activate
       
         mpirun -np 4 python integration2d_mpi.py

Monitoring resources' usage
---------------------------

Monitoring the resources that a certain job uses is important specially when this
job is expected to run on many CPUs and/or GPUs. It could happen, for instance, that 
an incorrect module is loaded or the command for running on many CPUs is not 
the proper one and our job runs in serial mode while we allocated possibly many 
CPUs/GPUs. For this reason, there are several tools available in our centers to 
monitor the performance of running jobs.

HPC2N
~~~~~

On a Kebnekaise terminal, you can type the command: 

.. code-block:: sh 

    $ job-usage job_ID

where ``job_ID`` is the number obtained when you submit your job with the ``sbatch``
command. This will give you a URL that you can copy and then paste in your local
browser. The results can be seen in a graphical manner a couple of minutes after the
job starts running, here there is one example of how this looks like:

.. figure:: img/monitoring-jobs.png
   :align: center

   The resources used by a job can be monitored in your local browser.   
   For this job, we can notice that 100% of the requested CPU 
   and 60% of the GPU resources are being used.

Additional information
----------------------
         
* `On parallel software engineering education using python <https://link.springer.com/article/10.1007/s10639-017-9607-0>`_
* `High Performance Data Analytics in Python @ENCCS  <https://enccs.github.io/HPDA-Python/parallel-computing/>`_
* `List of parallel libraries for Python <https://wiki.python.org/moin/ParallelProcessing>`_


.. keypoints::

   - You deploy cores and nodes via SLURM, either in interactive mode or batch
   - In Python, threads, distributed and MPI parallelization can be used.
  
