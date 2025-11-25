Basic batch and Slurm
=====================

.. questions::

   - What is a batch job?
   - What are some important commands regarding batch jobs? 
   - How to make a batch job?
 
.. objectives:: 

   - Short introduction to SLURM scheduler commands 
   - Show structure of a batch script
   - Try example

.. admonition:: Compute allocations in this workshop 

   - Pelle: ``uppmax2025-2-393``
   - Kebnekaise: ``hpc2n2025-151``
   - Cosmos: ``lu2025-7-106``
   - Tetralith: ``naiss2025-22-934``  
   - Dardel: ``naiss2025-22-934``
   - Alvis: ``naiss2025-22-934``  

.. admonition:: Storage space for this workshop 

   - Pelle: ``/proj/hpc-python-uppmax``
   - Kebnekaise: ``/proj/nobackup/fall-courses``
   - Cosmos: ``/lunarc/nobackup/projects/lu2025-17-52``
   - Tetralith: ``/proj/courses-fall-courses``
   - Dardel: ``/cfs/klemming/projects/supr/courses-fall-courses``
   - Alvis: ``/mimer/NOBACKUP/groups/courses-fall-2025``

.. admonition:: Reservation

   Include with ``#SBATCH --reservation==<reservation-name>``. On UPPMAX it is "magnetic" and so follows the project ID without you having to add the reservation name. 

   **NOTE** as there is only one/a few nodes reserved, you should NOT use the reservations for long jobs as this will block their use for everyone else. Using them for short test jobs is what they are for. 

   - UPPMAX 
       -   
   - HPC2N
       - ``hpc-python-fri`` for one AMD Zen4 cpu on Friday
       - ``hpc-python-mon`` for one AMD Zen4 cpu on Monday
       - ``hpc-python-tue`` for two L40s gpus on Tuesday
       - it is magnetic, so will be used automatically 

   - LUNARC 
       - ``hpc-python-dayN`` for up to 2 CPU nodes per day, where N=1 for Thursday, 2 for Friday, 3 for Monday, and 4 for Tuesday
       - ``hpc-python-day4-gpu`` for the GPU and ML sessions on Tuesday afternoon


What is a batch job?
--------------------

Batch systems keeps track of available system resources and takes care of scheduling jobs of multiple users running their tasks simultaneously. It typically organizes submitted jobs into some sort of prioritized queue. The batch system is also used to enforce local system resource usage and job scheduling policies.

Most Swedish HPC clusters are running Slurm. It is an Open Source job scheduler, which provides three key functions.

- First, it allocates to users, exclusive or non-exclusive access to resources for some period of time.
- Second, it provides a framework for starting, executing, and monitoring work on a set of allocated nodes (the cluster).
- Third, it manages a queue of pending jobs, in order to distribute work across resources according to policies.

Slurm is designed to handle thousands of nodes in a single cluster, and can sustain throughput of 120,000 jobs per hour.

What is a cluster? 
------------------ 

You can find information about the cluster hardware in the https://uppmax.github.io/HPC-python/common/understanding_clusters.html common section. This is just a brief overview.

- Login nodes and calculations/compute nodes

- A network of computers, each computer working as a **node**.

- Each node contains several processor cores and RAM and a local disk called scratch.

.. figure:: ../img/node.png
   :align: center

- The user logs in to **login nodes**  via Internet through ssh or Thinlinc.
    - Here the file management and lighter data analysis can be performed.
- The **calculation nodes** have to be used for intense computing. Any longer, resource-intensive, or parallel jobs must be run through a **batch script** or an **interactive session**.

- Beginner's guide to clusters: https://docs.hpc2n.umu.se/tutorials/clusterguide/
- NAISS course about Running jobs on HPC systems: https://uppmax.github.io/NAISS_Slurm/

What are some important commands regarding batch jobs? 
------------------------------------------------------
------------------------------------------------------------------------

This is a brief summary of some of the most important Slurm commands: 

- **Submit job**: ``sbatch JOBSCRIPT``
    - When you submit a job, the system also returns a Job-ID.   
- **Get list of your jobs**: ``squeue -u USERNAME`` or ``squeue --me``
    You can also find your Job-ID from this command   
- **Give Slurm commands on command line**: ``srun <commands-for-your-job> program``
- **Check on a specific job**: ``scontrol show job JOBID``
- **Delete a specific job**: ``scancel JOBID``
- **Delete all your own jobs**: ``scancel -u USERNAME``
- **Submit job**: ``sbatch JOBSCRIPT``
- **Get info on partitions and nodes**: ``sinfo`` 

Running your programs and scripts on UPPMAX, HPC2N, LUNARC, NSC, and PDC
------------------------------------------------------------------------

As mentioned under interactive jobs, any longer, resource-intensive, or parallel jobs must be run through a **batch script** or in an interactive session on allocated compute nodes.

A batch job is **not** interactive, so you cannot make changes to the job while it is running. 

In order to run a batch job, you need to create and submit a SLURM submit file (also called a batch submit file, a batch script, or a job script).

Guides and documentation at: 

- HPC2N: http://www.hpc2n.umu.se/support 
- UPPMAX: http://docs.uppmax.uu.se/cluster_guides/slurm/
- LUNARC: https://lunarc-documentation.readthedocs.io/en/latest/manual/manual_intro/
- NSC: https://www.nsc.liu.se/support/batch-jobs/   
- PDC: https://support.pdc.kth.se/doc/run_jobs/job_scheduling/
- C3SE: https://www.c3se.chalmers.se/documentation/submitting_jobs/running_jobs/ 

Workflow
########

- Write a batch script

  - Inside the batch script you need to load the modules you need (Python, Python packages, any prerequisites, ... )
  - Possibly activate an isolated/virtual environment to access own-installed packages
  - Ask for resources depending on if it is a parallel job or a serial job, if you need GPUs or not, etc.
  - Give the command(s) to your Python script

- Submit batch script with ``sbatch <my-python-script.sh>`` 

Common file extensions for batch scripts are ``.sh`` or ``.batch``, but they are not necessary. You can choose any name that makes sense to you. 

Simple example batch script
########################### 

.. hint:: 

   Type along!

This first example shows how to run a short, serial script. The batch script (named ``run_mmmult.sh``) can be found in the directory: 

- If you did ``git clone https://github.com/UPPMAX/HPC-python.git``
    - ``HPC-Python/Exercises/day2/<center>``, where <center> is hpc2n, uppmax, lunarc, nsc, pdc, or c3se. 
    - The Python script is in ``HPC-Python/Exercises/day2/programs`` and is named ``mmmult.py``. 
- If you did ``wget https://github.com/UPPMAX/HPC-python/raw/refs/heads/main/exercises.tar.gz`` and then ``tar -xvzf exercises.tar.gz`` 
    - ``exercises/day2/<center>``, where <center> is hpc2n, uppmax, lunarc, nsc, or pdc.
    - The Python script is in ``exercises/day2/programs`` and is named ``mmmult.py``.  

1. The batch script is run with ``sbatch run_mmmult.sh``. 
2. Try type ``squeue -u <username>`` to see if it is pending or running. 
3. When it has run, look at the output with ``nano slurm-<jobid>.out``. 

.. tabs::

   .. tab:: UPPMAX

      Short serial example script for Pelle. Loading Python 3.12.3 and a compatible SciPy-bundle for Numpy.  

      .. code-block:: bash

         #!/bin/bash -l 
         #SBATCH -A uppmax2025-2-393 # Change to your own after the course
         #SBATCH --time=00:20:00 # Asking for 20 minutes
         #SBATCH -n 1 # Asking for 1 core
            
         # Load any modules you need, here Python 3.12.3 
         # and a compatible SciPy-bundle for numpy  
         module load Python/3.12.3-GCCcore-13.3.0
         module load SciPy-bundle/2024.05-gfbf-2024a
         
         # Run your Python script 
         python mmmult.py   
            

   .. tab:: HPC2N

        Short serial example for running on Kebnekaise. Loading SciPy-bundle/2023.07 and Python/3.11.3  
       
        .. code-block:: bash

            #!/bin/bash
            #SBATCH -A hpc2n2025-151 # Change to your own
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
            #SBATCH -A lu2025-7-106 # Change to your own
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
            #SBATCH -A naiss2025-22-934 # Change to your own
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for 3.11.5 and compatible SciPy-bundle
            module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 JupyterLab/4.2.0
            
            # Run your Python script 
            python mmmult.py                

   .. tab:: PDC

        Short serial example for running on Dardel. Loading cray-python/3.11.7
       
        .. code-block:: bash

            #!/bin/bash
            #SBATCH -A naiss2025-22-934 # Change to your own
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for cray-python/3.11.7.
            module load cray-python/3.11.7
            
            # Run your Python script 
            python mmmult.py                

   .. tab:: C3SE 
    
        Short serial example for running on Alvis. 

        NOTE: You need to ask for a GPU to run on alvis. This is a CPU job. Do not do things like this normally! Only use alvis for GPU jobs!

        .. code-block:: bash 

           #!/bin/bash
           # Change to your own project ID! 
           #SBATCH -A naiss2025-22-934
           #SBATCH --time=00:30:00 # Asking for 30 minutes
           # You need to ask for a GPU to run on alvis.
           # This is a CPU job. Do not do things like this normally!
           # Only use alvis for GPU jobs!
           #SBATCH --gpus-per-node=T4:1
           #SBATCH -n 1 -c 1 # Asking for 1 core    # one core per task

           # Load any modules you need, here for Python 3.11.3 and compatible SciPy-bundle
           module purge  > /dev/null 2>&1
           module load Python/3.11.3-GCCcore-12.3.0 OpenMPI/4.1.5-GCC-12.3.0 SciPy-bundle/2023.07-gfbf-2023a

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

            
        
Exercises
---------

.. challenge:: Run the first serial example script (the one that was used to run mmmult.py) from further up on the page for this short Python code (sum-2args.py) instead 
    
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
            #SBATCH -A hpc2n2025-151 # Change to your own
            #SBATCH --time=00:05:00 # Asking for 5 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python 3.11.3
            module load GCC/12.3.0  Python/3.11.3 SciPy-bundle/2023.07
            
            # Run your Python script 
            python sum-2args.py 2 3 

.. solution:: Solution for UPPMAX
    :class: dropdown
    
          This batch script is for UPPMAX. Adding the numbers 2 and 3. 
          
          .. code-block:: bash
 
            #!/bin/bash -l
            #SBATCH -A uppmax2025-2-393 # Change to your own after the course
            #SBATCH --time=00:05:00 # Asking for 5 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python 3.12.3
            module load Python/3.12.3-GCCcore-13.3.0
            module load SciPy-bundle/2024.05-gfbf-2024a
            
            # Run your Python script 
            python sum-2args.py 2 3 

.. solution:: Solution for LUNARC
    :class: dropdown
    
          This batch script is for Cosmos. Adding the numbers 2 and 3. 
          
          .. code-block:: bash
 
            #!/bin/bash
            #SBATCH -A lu2025-7-106 # Change to your own
            #SBATCH --time=00:05:00 # Asking for 5 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python 3.11.5
            module load GCC/13.2.0  Python/3.11.5
            
            # Run your Python script 
            python sum-2args.py 2 3 

.. solution:: Solution for NSC
    :class: dropdown

          This batch script is for Tetralith. Adding the numbers 2 and 3.

          .. code-block:: bash

            #!/bin/bash
            #SBATCH -A naiss2025-22-934 # Change to your own
            #SBATCH --time=00:05:00 # Asking for 5 minutes
            #SBATCH -n 1 # Asking for 1 core

            # Load any modules you need, here for Python 3.11.5
            module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 JupyterLab/4.2.0

            # Run your Python script
            python sum-2args.py 2 3

.. solution:: Solution for PDC
    :class: dropdown

          This batch script is for Dardel. Adding the numbers 2 and 3.

          .. code-block:: bash

            #!/bin/bash
            #SBATCH -A naiss2025-22-934 # Change to your own
            #SBATCH --time=00:05:00 # Asking for 5 minutes
            #SBATCH -n 1 # Asking for 1 core

            # Load any modules you need, here for Python 3.11.x
            module load cray-python/3.11.7

            # Run your Python script
            python sum-2args.py 2 3
            
.. solution:: Solution for C3SE 

              Alvis should only be used for GPU jobs 

.. keypoints::

   - The SLURM scheduler handles allocations to the calculation nodes
   - Batch jobs runs without interaction with user
   - A batch script consists of a part with SLURM parameters describing the allocation and a second part describing the actual work within the job, for instance one or several Python scripts.
   
      - Remember to include possible input arguments to the Python script in the batch script.

