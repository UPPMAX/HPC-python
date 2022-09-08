Running Python in batch mode
============================

.. questions::

   - What is a batch job?
   - How to make a batch job?

   
   
.. objectives:: 

   - Short introduction to SLURM scheduler
   - Show structure of a batch script
   - Try example


Any longer, resource-intensive, or parallel jobs must be run through a **batch script**.

The batch system used at both UPPMAX and HPC2N is called SLURM. 

SLURM is an Open Source job scheduler, which provides three key functions

- Keeps track of available system resources
- Enforces local system resource usage and job scheduling policies
- Manages a job queue, distributing work across resources according to policies

In order to run a batch job, you need to create and submit a SLURM submit file (also called a batch submit file, a batch script, or a job script).

Guides and documentation at: http://www.hpc2n.umu.se/support and https://www.uppmax.uu.se/support/user-guides/slurm-user-guide/ 

**Workflow**

- Write a batch script

  - Inside the batch script you need to load the modules you need (Python, Python packages ... )
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

Serial code
'''''''''''
            
.. tabs::

   .. tab:: UPPMAX

        Short serial example script for Rackham. Loading Python 3.9.5. Numpy is preinstalled and does not need to be loaded. 

        .. code-block:: sh

            #!/bin/bash
            #SBATCH -A SNIC2022-22-641 # Change to your own after the course
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here Python 3.9.5. 
            module load python/3.9.5 
            
            # Run your Python script 
            python mmmult.py   
            

   .. tab:: HPC2N

        Short serial example for running on Kebnekaise. Loading SciPy-bundle/2021.05 and Python/3.9.5  
       
        .. code-block:: sh

            #!/bin/bash
            #SBATCH -A SNIC2022-22-641 # Change to your own after the course
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python 3.9.5 and compatible SciPy-bundle
            module load GCC/10.3.0  OpenMPI/4.1.1 Python/3.9.5 SciPy-bundle/2021.05
            
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


        
Serial code + self-installed package in virt. env. 
''''''''''''''''''''''''''''''''''''''''''''''''''

.. tabs::

   .. tab:: UPPMAX

        Short serial example for running on Rackham. Loading Python/3.9.5 + using any Python packages you have installed yourself with venv. More information will follow under the separate session for UPPMAX. 

        .. code-block:: sh
        
            #!/bin/bash
            #SBATCH -A SNIC2022-22-641 # Change to your own after the course
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python 3.9.5 
            module load python/3.9.5
            
            # Activate your virtual environment. 
            # CHANGE <path-to-virt-env> to the full path where you installed your virtual environment
            # Example: /proj/snic2022-22-641/nobackup/mrspock/pythonUPPMAX 
            source <path-to-virt-env>/bin/activate
            
            # Run your Python script
            python <my_program.py>


   .. tab:: HPC2N

        Short serial example for running on Kebnekaise. Loading SciPy-bundle/2021.05, Python/3.9.5 + using any Python packages you have installed yourself with virtual environment. During the separate session for HPC2N there will more about how to install something yourself this way. 
       
        .. code-block:: sh

            #!/bin/bash
            #SBATCH -A SNIC2022-22-641 # Change to your own after the course
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python 3.9.5 and compatible SciPy-bundle
            module load GCC/10.3.0  OpenMPI/4.1.1 Python/3.9.5 SciPy-bundle/2021.05
            
            # Activate your virtual environment. 
            # CHANGE <path-to-virt-env> to the full path where you installed your virtual environment
            # Example: /proj/nobackup/snic2022-22-641/bbrydsoe/pythonHPC2N 
            source <path-to-virt-env>/bin/activate
            
            # Run your Python script 
            python <my_program.py>


GPU code
''''''''

.. tabs::

   .. tab:: UPPMAX



   .. tab:: HPC2N

        Short serial example for running on Kebnekaise. Loading SciPy-bundle/2021.05, Python/3.9.5 + Python package you have installed yourself with virtual environment. The full example can be found under "Using Python for Machine Learning jobs.       
       
        .. code-block:: sh

            #!/bin/bash
            #SBATCH -A SNIC2022-22-641 # Change to your own after the course
            #SBATCH --reservation=hpc-python
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            # Asking for one K80 card
            #SBATCH --gres=gpu:k80:1
            
            # Remove any loaded modules and load the ones we need
            module purge  > /dev/null 2>&1
            module load GCC/10.3.0  OpenMPI/4.1.1 TensorFlow/2.6.0-CUDA-11.3.1
          
            # Activate the virtual environment we installed to
            source /proj/nobackup/support-hpc2n/bbrydsoe/vpyenv/bin/activate
            
            # Run your Python script
            python <tf-example>.py
            

The recommended TensorFlow version for this course is 2.6.0 on Kebnekaise. The module is compatible with Python 3.9.5 (automatically loaded when you load TensorFlow and its other prerequisites).            

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
          
          .. code-block:: sh
 
            #!/bin/bash
            #SBATCH -A SNIC2022-22-641 # Change to your own after the course
            #SBATCH --time=00:05:00 # Asking for 5 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python 3.9.5
            module load GCC/10.3.0  OpenMPI/4.1.1 Python/3.9.5
            
            # Run your Python script 
            python sum-2args.py 2 3 

.. solution:: Solution for UPPMAX
    :class: dropdown
    
          This batch script is for Kebnekaise. Adding the numbers 2 and 3. 
          
          .. code-block:: sh
 
            #!/bin/bash
            #SBATCH -A SNIC2022-22-641 # Change to your own after the course
            #SBATCH --time=00:05:00 # Asking for 5 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python 3.9.5
            module load GCC/10.3.0  OpenMPI/4.1.1 Python/3.9.5
            
            # Run your Python script 
            python sum-2args.py 2 3 

.. keypoints::

   - The SLURM scheduler handles allocations to the calculation nodes
   - Interactive sessions was presented in last slide
   - Batch jobs runs without interaction with user
   - A batch script consists of a part with SLURM parameters describing the allocation and a second part describing the actual work within the job, for instance one or several Python scripts.
      - Remember to include possible input arguments to the Python script in the batch script.
    
