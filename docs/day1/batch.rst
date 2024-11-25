Running Python in batch mode
============================

.. questions::

   - What are the UPPMAX, HPC2N, LUNARC, and NSC clusters?
   - What is a batch job?
   - How to make a batch job?
 
.. objectives:: 

   - Short overview of the HPC systems
   - Short introduction to SLURM scheduler
   - Show structure of a batch script
   - Try example

Briefly about the cluster hardware and system at UPPMAX, HPC2N, LUNARC, and NSC
-------------------------------------------------------------------------------

**What is a cluster?**

- Login nodes and calculations/compute nodes

- A network of computers, each computer working as a **node**.
     
- Each node contains several processor cores and RAM and a local disk called scratch.

.. figure:: img/node.png
   :align: center

- The user logs in to **login nodes**  via Internet through ssh or Thinlinc.

  - Here the file management and lighter data analysis can be performed.

.. figure:: img/nodes.png
   :align: center

- The **calculation nodes** have to be used for intense computing. 

- Beginner's guide to clusters: https://www.hpc2n.umu.se/documentation/guides/beginner-guide

Common features
###############

- Intel CPUs
- Linux kernel
- Bash shell

.. role:: raw-html(raw)
    :format: html

.. list-table:: Hardware
   :widths: 25 25 25 25 25
   :header-rows: 1

   * - Technology
     - Kebnekaise
     - Rackham
     - Snowy
     - Bianca
     - Cosmos
     - Tetralith  
   * - Cores per calculation node
     - 28 (72 for largemem part + 8 nodes with 128)
     - 20
     - 16
     - 16
     - 48 (AMD) and 32 (Intel) 
     - 32   
   * - Memory per calculation node
     - 128-3072 GB 
     - 128-1024 GB
     - 128-4096 GB
     - 128-512 GB
     - 256-512 GB 
     - 96-384 GB  
   * - GPU
     - NVidia V100 + NVidia A100, :raw-html:`<br/>` AMD MI100, NVidia H100, :raw-html:`<br />` Nvidia A600, and 10 NVidia L40S
     - None
     - Nvidia T4 
     - 2 NVIDIA A100
     - NVidia A100
     - NVidia T4 


Running your programs and scripts on UPPMAX, HPC2N, LUNARC, and NSC
--------------------------------------------------------------------

Any longer, resource-intensive, or parallel jobs must be run through a **batch script**.

The batch system used at UPPMAX, HPC2N, LUNARC, and NSC is called SLURM. 

SLURM is an Open Source job scheduler, which provides three key functions

- Keeps track of available system resources
- Enforces local system resource usage and job scheduling policies
- Manages a job queue, distributing work across resources according to policies

In order to run a batch job, you need to create and submit a SLURM submit file (also called a batch submit file, a batch script, or a job script).

Guides and documentation at: 

- HPC2N: http://www.hpc2n.umu.se/support 
- UPPMAX: http://docs.uppmax.uu.se/cluster_guides/slurm/
- LUNARC: https://lunarc-documentation.readthedocs.io/en/latest/manual/manual_intro/
- NSC: https://www.nsc.liu.se/support/batch-jobs/   

**Workflow**

- Write a batch script

  - Inside the batch script you need to load the modules you need (Python, Python packages, any prerequisites, ... )
  - Possibly activate an isolated/virtual environment to access own-installed packages
  - Ask for resources depending on if it is a parallel job or a serial job, if you need GPUs or not, etc.
  - Give the command(s) to your Python script

- Submit batch script with ``sbatch <my-python-script.sh>`` 

Common file extensions for batch scripts are ``.sh`` or ``.batch``, but they are not necessary. You can choose any name that makes sense to you. 

Useful commands to the batch system
-----------------------------------

- Submit job: ``sbatch <jobscript.sh>``
- Get list of your jobs: ``squeue -u <username>``
- Check on a specific job: ``scontrol show job <job-id>``
- Delete a specific job: ``scancel <job-id>``
- Useful info about a job: ``sacct -l -j <job-id> | less -S``
- Url to a page with info about the job (Kebnekaise only): ``job-usage <job-id>``
         
Example Python batch scripts
---------------------------- 

**Serial code**

.. hint:: 

   Type along!

This first example shows how to run a short, serial script. The batch script (named ``run_mmmult.sh``) can be found in the directory /HPC-Python/Exercises/examples/<center>, where <center> is hpc2n, uppmax, lunarc, or nsc. The Python script is in /HPC-Python/Exercises/examples/programs and is named ``mmmult.py``. 

1. The batch script is run with ``sbatch run_mmmult.sh``. 
2. Try type ``squeue -u <username>`` to see if it is pending or running. 
3. When it has run, look at the output with ``nano slurm-<jobid>.out``. 

.. tabs::

   .. tab:: UPPMAX

        Short serial example script for Rackham. Loading Python 3.11.8. Numpy is preinstalled and does not need to be loaded. 

        .. code-block:: bash

            #!/bin/bash -l 
            #SBATCH -A naiss2024-22-1442 # Change to your own after the course
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here Python 3.11.8. 
            module load python/3.11.8 
            
            # Run your Python script 
            python mmmult.py   
            

   .. tab:: HPC2N

        Short serial example for running on Kebnekaise. Loading SciPy-bundle/2023.07 and Python/3.11.3  
       
        .. code-block:: bash

            #!/bin/bash
            #SBATCH -A hpc2n2024-142 # Change to your own
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python/3.11.3 and compatible SciPy-bundle
            module load GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07
            
            # Run your Python script 
            python mmmult.py    
            
   .. tab:: LUNARC

        Short serial example for running on Cosmos. Loading SciPy-bundle/2023.11 and Python/3.11.5  
       
        .. code-block:: bash

            #!/bin/bash
            #SBATCH -A lu2024-2-88 # Change to your own
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python/3.11.5 and compatible SciPy-bundle
            module load GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11
            
            # Run your Python script 
            python mmmult.py    
            
   .. tab:: NSC

        Short serial example for running on Tetralith. Loading SciPy-bundle/2022.05 and Python/3.10.4 
       
        .. code-block:: bash

            #!/bin/bash
            #SBATCH -A naiss2024-22-1493 # Change to your own
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python/3.10.4 and compatible SciPy-bundle
            module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4 SciPy-bundle/2022.05
            
            # Run your Python script 
            python mmmult.py                
            
   .. tab:: mmmult.py 
   
        Python example code
   
        .. code-block:: python
        
            import timeit
            import numpy as np
            
            starttime = timeit.default_timer()
            
            np.random.seed(1701)
            
            A = np.random.randint(-1000, 1000, size=(8,4))
            B = np.random.randint(-1000, 1000, size =(4,4))
            
            print("This is matrix A:\n", A)
            print("The shape of matrix A is ", A.shape)
            print()
            print("This is matrix B:\n", B)
            print("The shape of matrix B is ", B.shape)
            print()
            print("Doing matrix-matrix multiplication...")
            print()
            
            C = np.matmul(A, B)
            
            print("The product of matrices A and B is:\n", C)
            print("The shape of the resulting matrix is ", C.shape)
            print()
            print("Time elapsed for generating matrices and multiplying them is ", timeit.default_timer() - starttime)

            
        
**Serial code + self-installed package in virt. env.**

.. hint::

   Don't type along! We will go through an example like this with your self-installed virtual environment under the ML section. 

.. tabs::

   .. tab:: UPPMAX

        Short serial example for running on Rackham. Loading python/3.11.8 + using any Python packages you have installed yourself with venv.  

        .. code-block:: bash
        
            #!/bin/bash -l 
            #SBATCH -A naiss2024-22-1442 # Change to your own after the course
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for python 3.11.8 
            module load python/3.11.8
            
            # Activate your virtual environment. 
            source /proj/hpc-python-fall/<user-dir>/<path-to-virtenv>/<virtenv>/bin/activate  
            
            # Run your Python script (remember to add the path to it 
            # or change to the directory with it first)
            python <my_program.py>


   .. tab:: HPC2N

        Short serial example for running on Kebnekaise. Loading SciPy-bundle/2023.07, Python/3.11.3, matplotlib/3.7.2 + using any Python packages you have installed yourself with virtual environment.  
       
        .. code-block:: bash

            #!/bin/bash
            #SBATCH -A hpc2n2024-142 # Change to your own 
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python/3.11.3 and compatible SciPy-bundle
            module load GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2
            
            # Activate your virtual environment. 
            source /proj/nobackup/hpc-python-fall-hpc2n/<user-dir>/<path-to-virt-env>/bin/activate
            
            # Run your Python script  (remember to add the path to it 
            # or change to the directory with it first)
            python <my_program.py>

   .. tab:: LUNARC

        Short serial example for running on Cosmos. Loading SciPy-bundle/2023.11, Python/3.11.5, matplotlib/3.8.2 + using any Python packages you have installed yourself with virtual environment.  
       
        .. code-block:: bash

            #!/bin/bash
            #SBATCH -A lu2024-2-88 # Change to your own 
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python/3.11.5 and compatible SciPy-bundle
            module load GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 matplotlib/3.8.2
            
            # Activate your virtual environment. 
            source <path-to-virt-env>/bin/activate
            
            # Run your Python script  (remember to add the path to it 
            # or change to the directory with it first)
            python <my_program.py>

   .. tab:: NSC

        Short serial example for running on Tetralith. Loading SciPy-bundle/2022.05, Python/3.10.4, matplotlib/3.5.2 + using any Python packages you have installed yourself with virtual environment.  
       
        .. code-block:: bash

            #!/bin/bash
            #SBATCH -A naiss2024-22-1493 # Change to your own 
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python/3.10.4 and compatible SciPy-bundle
            module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4 SciPy-bundle/2022.05 matplotlib/3.5.2
            
            # Activate your virtual environment. 
            source /proj/hpc-python-fall-nsc/<user-dir>/<path-to-virt-env>/bin/activate
            
            # Run your Python script  (remember to add the path to it 
            # or change to the directory with it first)
            python <my_program.py>

            

**Job arrays** 

This is a very simple example of how to run a Python script with a job array. 

.. hint::

   Do not type along! You can try it later during exercise time if you want! 
   
.. tabs:: 

   .. tab:: hello-world-array.py   
      
      .. code-block:: python 

         # import sys library (we need this for the command line args)
         import sys

         # print task number
         print('Hello world! from task number: ', sys.argv[1])

   .. tab:: UPPMAX

      .. code-block:: bash 

         #!/bin/bash -l
         # This is a very simple example of how to run a Python script with a job array
         #SBATCH -A naiss2024-22-415 # Change to your own after the course
         #SBATCH --time=00:05:00 # Asking for 5 minutes
         #SBATCH --array=1-10   # how many tasks in the array 
         #SBATCH -c 1 # Asking for 1 core    # one core per task 
         #SBATCH -o hello-world-%j-%a.out

         # Set a path where the example programs are installed. 
         # Change the below to your own path to where you placed the example programs
         MYPATH=/proj/hpc-python/<userdir>/HPC-python/Exercises/examples/programs/

         # Load any modules you need, here for Python 3.11.8
         ml uppmax
         ml python/3.11.8

         # Run your Python script
         srun python $MYPATH/hello-world-array.py $SLURM_ARRAY_TASK_ID


   .. tab:: HPC2N 

      .. code-block:: bash 

         #!/bin/bash
         # This is a very simple example of how to run a Python script with a job array
         #SBATCH -A hpc2n2024-052 # Change to your own!
         #SBATCH --time=00:05:00 # Asking for 5 minutes
         #SBATCH --array=1-10   # how many tasks in the array 
         #SBATCH -c 1 # Asking for 1 core    # one core per task 
         #SBATCH -o hello-world-%j-%a.out

         # Set a path where the example programs are installed. 
         # Change the below to your own path to where you placed the example programs
         MYPATH=/proj/nobackup/python-hpc/<your-dir>/HPC-python/Exercises/examples/programs/

         # Load any modules you need, here for Python 3.11.3 
         ml GCC/12.3.0 Python/3.11.3

         # Run your Python script
         srun python $MYPATH/hello-world-array.py $SLURM_ARRAY_TASK_ID


**GPU code**

.. hint:: 

   Type along! 

.. tabs::

   .. tab:: UPPMAX

        Short GPU example for running ``compute.py`` on Snowy.         
       
        .. code-block:: bash

            #!/bin/bash -l
            #SBATCH -A naiss2024-22-415
            #SBATCH -t 00:10:00
            #SBATCH --exclusive
            #SBATCH -n 1
            #SBATCH -M snowy
            #SBATCH --gres=gpu=1
            
            # Load any modules you need, here loading python 3.11.8 and the ML packages 
            module load uppmax
            module load python/3.11.8
            module load python_ML_packages/3.11.8-gpu 
            
            # Run your code
            python compute.py 
            

   .. tab:: HPC2N

        Example with running ``compute.py`` on Kebnekaise.        
       
        .. code-block:: bash

            #!/bin/bash
            #SBATCH -A hpc2n2024-052 # Change to your own
            #SBATCH --time=00:10:00  # Asking for 10 minutes
            # Asking for one V100 card
            #SBATCH --gres=gpu:v100:1
            
            # Remove any loaded modules and load the ones we need
            module purge  > /dev/null 2>&1
            module load GCC/12.3.0 OpenMPI/4.1.5 Python/3.11.3 SciPy-bundle/2023.07 numba/0.58.1    
            
            # Run your Python script
            python compute.py
           

   .. tab:: compute.py

        This Python script can (just like the batch scripts for UPPMAX and HPC2N), be found in the ``/HPC-Python/Exercises/examples`` directory, under the subdirectory ``programs`` - if you have cloned the repo or copied the tarball with the exercises.

        .. code-block:: python 

           from numba import jit, cuda
           import numpy as np
           # to measure exec time
           from timeit import default_timer as timer

           # normal function to run on cpu
           def func a):
               for i in range(10000000):
                   a[i]+= 1

           # function optimized to run on gpu
           @jit(target_backend='cuda')
           def func2(a):
               for i in range(10000000):
                   a[i]+= 1
           if __name__=="__main__":
               n = 10000000
               a = np.ones(n, dtype = np.float64)

               start = timer()
               func(a)
               print("without GPU:", timer()-start)

               start = timer()
               func2(a)
               print("with GPU:", timer()-start)


Exercises
---------

.. challenge:: Run the first serial example script from further up on the page for this short Python code (sum-2args.py)
    
    .. code-block:: python
    
        import sys
            
        x = int(sys.argv[1])
        y = int(sys.argv[2])
            
        sum = x + y
            
        print("The sum of the two numbers is: {0}".format(sum))
        
    Remember to give the two arguments to the program in the batch script.

.. solution:: Solution for HPC2N
    :class: dropdown
    
          This batch script is for Kebnekaise. Adding the numbers 2 and 3. 
          
          .. code-block:: bash
 
            #!/bin/bash
            #SBATCH -A hpc2n2024-052 # Change to your own
            #SBATCH --time=00:05:00 # Asking for 5 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python 3.11.3
            module load GCC/12.3.0  Python/3.11.3
            
            # Run your Python script 
            python sum-2args.py 2 3 

.. solution:: Solution for UPPMAX
    :class: dropdown
    
          This batch script is for UPPMAX. Adding the numbers 2 and 3. 
          
          .. code-block:: bash
 
            #!/bin/bash -l
            #SBATCH -A naiss2024-22-415 # Change to your own after the course
            #SBATCH --time=00:05:00 # Asking for 5 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for python 3.11.8
            module load python/3.11.8
            
            # Run your Python script 
            python sum-2args.py 2 3 

.. keypoints::

   - The SLURM scheduler handles allocations to the calculation nodes
   - Interactive sessions was presented in last slide
   - Batch jobs runs without interaction with user
   - A batch script consists of a part with SLURM parameters describing the allocation and a second part describing the actual work within the job, for instance one or several Python scripts.
   
      - Remember to include possible input arguments to the Python script in the batch script.

**Example from previously in th ML section - redo** 

Pandas and matplotlib
---------------------

This is the same example that was shown in the section about loading and running Python, but now changed slightly to run as a batch job. The main difference is that here we cannot open the plot directly, but have to save to a file instead. You can see the change inside the Python script.

.. tabs::

   .. tab:: Directly

      Remove the # if running on Kebnekaise

      .. code-block:: python

         import pandas as pd
         #import matplotlib
         import matplotlib.pyplot as plt

         #matplotlib.use('TkAgg')

         dataframe = pd.read_csv("scottish_hills.csv")
         x = dataframe.Height
         y = dataframe.Latitude
         plt.scatter(x, y)
         plt.show()

   .. tab:: From a Batch-job

      Remove the # if running on Kebnekaise. The script below can be found as ``pandas_matplotlib-batch-rackham.py`` or ``pandas_matplotlib-batch-kebnekaise.py`` in the ``Exercises/examples/programs`` directory.

      .. code-block:: python

         import pandas as pd
         #import matplotlib
         import matplotlib.pyplot as plt

         #matplotlib.use('TkAgg')

         dataframe = pd.read_csv("scottish_hills.csv")
         x = dataframe.Height
         y = dataframe.Latitude
         plt.scatter(x, y)
         plt.show()

   .. tab:: From a Batch-job 

      Remove the # if running on Kebnekaise. The script below can be found as ``pandas_matplotlib-batch-rackham.py`` or ``pandas_matplotlib-batch-kebnekaise.py`` in the ``Exercises/examples/programs`` directory. 

      .. code-block:: python

         import pandas as pd
         #import matplotlib
         import matplotlib.pyplot as plt
         
         #matplotlib.use('TkAgg')

         dataframe = pd.read_csv("scottish_hills.csv")
         x = dataframe.Height
         y = dataframe.Latitude
         plt.scatter(x, y)
         plt.savefig("myplot.png")

.. hint::

   Type along!
   
Batch scripts for running on Rackham and Kebnekaise.

.. tabs:: 

   .. tab:: Rackham 

      .. code-block:: bash

         #!/bin/bash -l
         #SBATCH -A naiss2024-22-415
         #SBATCH --time=00:05:00 # Asking for 5 minutes
         #SBATCH -n 1 # Asking for 1 core

         # Load any modules you need, here for Python 3.11.8
         ml python/3.11.8

         # Run your Python script
         python pandas_matplotlib-batch-rackham.py 

   .. tab:: Kebnekaise 

      .. code-block:: bash

         #!/bin/bash
         #SBATCH -A hpc2n2024-052
         #SBATCH --time=00:05:00 # Asking for 5 minutes
         #SBATCH -n 1 # Asking for 1 core

         # Load any modules you need, here for Python 3.11.3
         ml GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2

         # Run your Python script
         python pandas_matplotlib-batch-kebnekaise.py

Submit with ``sbatch <batch-script.sh>``.

The batch scripts can be found in the directories for hpc2n and uppmax, under ``Exercises/examples/``, and is named ``pandas_matplotlib-batch.sh`` .



