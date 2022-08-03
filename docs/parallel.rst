Parallel computing with Python
==============================

.. objectives::

   - Learn common schemes for the parallelization of codes
   - Learn general concepts for parallel computing

.. important::
   :class: dropdown

    **Prerequisites**

    - For Kebnekaise:
    
      ml GCCcore/11.2.0 Python/3.9.6

      virtualenv --system-site-packages /proj/nobackup/<your-project-storage>/vpyenv-python-course

      source /proj/nobackup/<your-project-storage>/vpyenv-python-course/bin/activate

      For the ``mpi4py`` modules add the following modules:

      ml GCC/11.2.0 OpenMPI/4.1.1

      python -m pip install mpi4py


In Python there are different schemes that can be used to parallelize Python codes. 
We will only take a look at some of these schemes that illustrate the general concepts of
parallel computing.

The workhorse for this section will be a 2D integration example:

   :math:`\int^{\pi}_{0}\int^{\pi}_{0}\sin(x+y)dxdy = 0`

   .. math::
      :nowrap:

      \begin{center}
         \int^{\pi}_{0}\int^{\pi}_{0}\sin(x+y)dxdy = 0
      \end{center}


One way to perform the integration is by creating a grid in the ``x`` and ``y`` directions.
More specifically, one divides the integration range in both directions into ``n`` bins. A
serial code can be seen in the following code block.

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

The most expensive part in this code is the double `for loop` and this will be the target
for parallelization. We can run this code on the terminal as follows: 


.. code-block:: sh 

    $ python integration2d_serial.py
    Integral value is -7.117752e-17, Error is 7.117752e-17
    Time spent: 21.01 sec


Threads
-------

In a threaded parallelization scheme the workers (threads) share a global memory address space.
The `threading <https://docs.python.org/3/library/threading.html>`_ 
module is built into Python so you don't have to installed it. By using this
modules, one can create several threads that can do some work (in principle) in parallel.
For jobs dealing with files I/O one can observe some speedup by using the `threading` module.
However, for CPU intensive jobs one will see a decrease in performance w.r.t. the serial code.
This is because Python uses the Global Interpreter Lock (`GIL <https://docs.python.org/3/c-api/init.html>`_)
which serializes the code when several threads are used.

In the following code we used the `threading` module to parallelize the 2D integration example.
Threads are created with the construct ``threading.Thread(target=function, args=())``, where 
`target` is the function that will be executed by each thread and `args` is a tuple containing the
arguments of that function. Threads are started with the ``start()`` method and when they finish
their job they are joined with the ``join()`` method.

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

Distributed
-----------


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
`MPI course <https://github.com/SNIC-MPI-course/MPI-course>`_ offered by SNIC.

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

    $ mpirun -np 4 python mpi-pi-calculation.py
    Integral value is 4.492851e-12, Error is 4.492851e-12
    Time spent: 5.76 sec


Additional information
----------------------
         
* `On parallel software engineering education using python <https://link.springer.com/article/10.1007/s10639-017-9607-0>`_
* `High Performance Data Analytics in Python @ENCCS  <https://enccs.github.io/HPDA-Python/parallel-computing/>`_
* `List of parallel libraries for Python <https://wiki.python.org/moin/ParallelProcessing>`_
