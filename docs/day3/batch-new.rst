Running Python in batch mode
============================

.. questions::

   - Can I run more advanced Python scripts as batch jobs?
   - Can you do matplotlib or pandas as batch jobs?
   - What about using virtual environments in batch jobs? 
 
.. objectives:: 

   - Use a virtual environment in a batch script. 
   - Show how a Python code with pandas and matplotlib can be transformed to run in a batch script. 
   - Some examples to try. 

.. admonition:: Compute allocations, storage space, reservations in this workshop 

   - This was covered Friday, during the section on basic batch and Slurm
   - You can find an overview in the common section NAISS projects overview: https://uppmax.github.io/HPC-python/common/naiss_projects_overview.html 

Running your programs and scripts on UPPMAX, HPC2N, LUNARC, C3SE, NSC, and PDC 
------------------------------------------------------------------------------

As mentioned Friday, in the introduction to Slurm and batch jobs: 

- Any longer, resource-intensive, or parallel jobs must be run through a **batch script** or in an interactive session on allocated compute nodes.
- A batch job is **not** interactive, so you cannot make changes to the job while it is running. 
- In order to run a batch job, you need to create and submit a SLURM submit file (also called a batch submit file, a batch script, or a job script).

.. admonition:: "Recap: Useful commands to the batch system" 

   - Submit job: ``sbatch <jobscript.sh>``
   - Get list of your jobs: ``squeue -u <username>`` or ``squeue --me``
   - Check on a specific job: ``scontrol show job <job-id>``
   - Delete a specific job: ``scancel <job-id>``
   - Useful info about a job: ``sacct -l -j <job-id> | less -S``
         
Example Python batch scripts
---------------------------- 
       
Friday we looked at a simple Python serial code run as a batch script. There are many other situations: 

- Python code needing self-installed packages in virtual environment
- Python code requiring tweaking before running as a batch job
- Python code that is parallel
- Python code that needs GPUs 

Today we will look at some of these situations. The GPU example will be covered tomorrow where we will also talk about parallelism (today that will only be shown with a small batch script template).   

Serial code + self-installed package in virt. env.
##################################################

.. hint::

   Don't type along! This just shows how you would activate and use a virtual environment in a batch script. 

.. tabs::

   .. tab:: UPPMAX

        Short serial example for running on Pelle. We are loading Python 3.11.5 and a compatible SciPy-bundle and Python-bundle-PyPi. This gives us access to packages like scipy, numpy, pandas, seaborn. PyTorch and matplotlib are their own modules and only available for Python 3.12.3. 
        
        The important thing is to load the SAME modules you used in the virtual environment you have installed the needed packages in.   

        .. code-block:: bash
        
            #!/bin/bash -l 
            #SBATCH -A uppmax2025-2-393 # Change to your own after the course
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python 3.11.5 and a compatible SciPy-bundle and a compatible Python-bundle-PyPi. 
            module load Python/3.11.5-GCCcore-13.2.0
            module load SciPy-bundle/2023.11-gfbf-2023b 
            module load Python-bundle-PyPI/2023.10-GCCcore-13.2.0
            
            # Activate your virtual environment, which you previously created with the above modules loaded. 
            source /proj/hpc-python-uppmax/<user-dir>/<path-to-virtenv>/<virtenv>/bin/activate  
            
            # Run your Python script (remember to add the path to it 
            # or change to the directory with it first)
            python <my_program.py>


   .. tab:: HPC2N

        Short serial example for running on Kebnekaise. We are loading Python 3.11.5 and a compatible SciPy-bundle and Python-bundle-PyPi. This gives us access to packages like scipy, numpy, pandas, seaborn. PyTorch and matplotlib are their own modules and are available for most Python versions. PyTorch requires you to also load OpenMPI. 
        
        The important thing is to load the SAME modules you used in the virtual environment you have installed the needed packages in.
       
        .. code-block:: bash

            #!/bin/bash
            #SBATCH -A hpc2n2025-151 # Change to your own 
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python/3.11.5 and compatible SciPy-bundle and Python-bundle-PyPi
            module load GCC/13.2.0 Python/3.11.5 
            module load SciPy-bundle/2023.11
            module load Python-bundle-PyPI/2023.10
            
            # Activate your virtual environment, which you previously created with the above modules loaded. 
            source /proj/nobackup/fall-courses/<user-dir>/<path-to-virt-env>/bin/activate
            
            # Run your Python script  (remember to add the path to it 
            # or change to the directory with it first)
            python <my_program.py>

   .. tab:: LUNARC

        Short serial example for running on Cosmos. Loading Python/3.11.5 and compatible SciPy-bundle and Python-bundle-PyPi. 

        The important thing is to load the SAME modules you used in the virtual environment you have installed the needed packages in.  
       
        .. code-block:: bash

            #!/bin/bash
            #SBATCH -A lu2025-7-106 # Change to your own 
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python/3.11.5 and compatible SciPy-bundle and Python-bundle-PyPi
            module load GCC/13.2.0 Python/3.11.5
            module load SciPy-bundle/2023.11
            module load Python-bundle-PyPI/2023.10
            
            # Activate your virtual environment which was created with the same modules as above. 
            source <path-to-virt-env>/bin/activate
            
            # Run your Python script  (remember to add the path to it 
            # or change to the directory with it first)
            python <my_program.py>

   .. tab:: NSC

        Short serial example for running on Tetralith. Loading Python/3.11.5, a compatible SciPy-bundle and prerequisites, as well as JupyterLab (containing some extra packages).  
       
        .. code-block:: bash

            #!/bin/bash
            #SBATCH -A naiss2025-22-934 # Change to your own 
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python/3.11.5 and compatible SciPy-bundle, etc. 
            module load Python/3.11.5
            module load buildtool-easybuild/4.9.4-hpc71cbb0050 GCC/13.2.0
            module load SciPy-bundle/2023.11 Python-bundle-PyPI/2023.10
            module load JupyterLab/4.2.0
            
            # Activate your virtual environment, which was built with the same modules as above. 
            source /proj/courses-fall-2025/<user-dir>/<path-to-virt-env>/bin/activate
            
            # Run your Python script  (remember to add the path to it 
            # or change to the directory with it first)
            python <my_program.py>
            
   .. tab:: PDC 

        Short serial example for running on Dardel. Loading Python/3.11.x + using any Python packages you have installed yourself with virtual environment. The module cray-python is recommended and contains numpy, scipy, pandas, mpi4py and more 
       
        .. code-block:: bash

            #!/bin/bash
            #SBATCH -A naiss2025-22-934 # Change to your own 
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            #SBATCH -p shared

            # Load any modules you need, here for Python/3.11.x 
            module load cray-python/3.11.7 
            
            # Activate your virtual environment, which was built with the same modules as above. 
            source /cfs/klemming/projects/supr/courses-fall-2025/<user-dir>/<path-to-virt-env>/bin/activate
            
            # Run your Python script  (remember to add the path to it 
            # or change to the directory with it first)
            python <my_program.py>

   .. tab:: C3SE 

        Short serial example for running on Alvis. Loading Python/3.13.5 + a compatible SciPy-bundle + Python-bundle-PyPI and using any Python packages you have installed yourself with virtual environment. 
               
        .. code-block:: bash

            #!/bin/bash
            #SBATCH -A naiss2025-22-934 # Change to your own 
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            # You need to ask for a GPU to run on alvis.
            # This is a CPU job. Do not do things like this normally!
            # Only use alvis for GPU jobs!
            #SBATCH --gpus-per-node=T4:1
            #SBATCH -n 1 -c 1 # Asking for 1 core    # one core per task
            
            # Load any modules you need, here for Python/3.11.5 etc 
            module load Python/3.11.5-GCCcore-13.2.0
            module load SciPy-bundle/2025.07-gfbf-2025b
            module load Python-bundle-PyPI/2025.07-GCCcore-14.3.0
            
            # Activate your virtual environment, which was built with the same modules as above. 
            source /mimer/NOBACKUP/groups/courses-fall-2025/<user-dir>/<path-to-virt-env>/bin/activate
            
            # Run your Python script  (remember to add the path to it 
            # or change to the directory with it first)
            python <my_program.py>

MPI code
########

We will talk more about parallel code in the session "Parallel computing with Python" tomorrow. This is a simple example of a batch script to run an MPI code. 

.. tabs::

   .. tab:: NSC

        Short MPI example for running on Tetralith. 
        
        .. code-block:: 

           #!/bin/bash
           # Change to your own project account after the course 
           #SBATCH -A naiss2025-22-934
           # Asking for 10 min 
           #SBATCH -t 00:10:00
           # ask for 32 cores here, modify for your needs.
           # Aim to use multiples of 32 for larger jobs
           #SBATCH -n 32
           # name output and error file
           #SBATCH -o mpi_process_%j.out
           #SBATCH -e mpi_process_%j.err

           # Load Python and mpi4py
           ml purge > /dev/null 2>&1
           ml buildtool-easybuild/4.8.0-hpce082752a2  GCC/13.2.0  OpenMPI/4.1.6 mpi4py/3.1.5

           # Run your mpi_executable
           mpirun -np 32 python integration2d_mpi.py

   .. tab:: PDC

        Short MPI example for running on Dardel. 
        
        .. code-block:: 

           #!/bin/bash
           # Change to your own project account after the course 
           #SBATCH -A naiss2025-22-934
           # Asking for 10 min 
           #SBATCH -t 00:10:00
           # Using the Dardel shared partition
           #SBATCH -p shared
           # ask for 16 core on one node, modify for your needs.
           #SBATCH -N 1
           #SBATCH --ntasks-per-node=16
           # name output and error file
           #SBATCH -o mpi_process_%j.out
           #SBATCH -e mpi_process_%j.err
           # Loading a suitable module. Here for cray-python
           module load cray-python/3.11.7

           # Run your mpi_executable
           mpirun -np 16 python integration2d_mpi.py

   .. tab:: C3SE

        Short MPI example for running on Alvis. 
        
        .. code-block:: 

           #!/bin/bash
           # Change to your own project account after the course 
           #SBATCH -A naiss2025-22-934
           # Asking for 10 min 
           #SBATCH -t 00:10:00
           #SBATCH -p alvis
           # You need to ask for a GPU to run on alvis.
           # This is a CPU job. Do not do things like this normally!
           # Only use for GPU jobs!
           #SBATCH -N 1 --gpus-per-node=T4:1
           # Number of tasks - default is 1 core per task. Here 4 
           #SBATCH -n 4

           # Time in HHH:MM:SS - at most 168 hours. 
           #SBATCH --time=00:05:00

           # It is always a good idea to do ml purge before loading other modules 
           ml purge > /dev/null 2>&1
           # Load a suitable Python module 
           ml add Python/3.13.5-GCCcore-14.3.0 SciPy-bundle/2025.07-gfbf-2025b Python-bundle-PyPI/2025.07-GCCcore-14.3.0 mpi4py/4.1.0-gompi-2025b

           # Run the program. Remember to use "srun" unless the program handles parallelizarion itself
           srun python integration2d_mpi.py  

   .. tab:: UPPMAX

        Short MPI example for running on Pelle. 
        
        .. code-block:: 

           #!/bin/bash -l
           #SBATCH -A uppmax2025-2-393
           #SBATCH -t 00:05:00
           #SBATCH -n 4
           #SBATCH -o output_%j.out   # output file
           #SBATCH -e error_%j.err    # error messages

           module load Python/3.11.5-GCCcore-13.2.0 SciPy-bundle/2023.11-gfbf-2023b Python-bundle-PyPI/2023.10-GCCcore-13.2.0 mpi4py/4.0.1-gompi-2024a                                                                                                                   
           mpirun -np 4 python integration2d_mpi.py

   .. tab:: HPC2N

        Short MPI example for running on Kebnekaise. 
        
        .. code-block:: 

           #!/bin/bash
           # Change to your own project id! 
           #SBATCH -A hpc2n2025-151
           #SBATCH -t 00:05:00
           #SBATCH -n 4
           #SBATCH -o output_%j.out   # output file
           #SBATCH -e error_%j.err    # error messages

           ml purge > /dev/null 2>&1
           ml GCC/12.3.0 Python/3.11.3
           ml OpenMPI/4.1.5
           ml SciPy-bundle/2023.07 mpi4py/3.1.4 

           mpirun -np 4 python integration2d_mpi.py

   .. tab:: LUNARC

        Short MPI example for running on Cosmos. 
        
        .. code-block:: 

           #!/bin/bash
           # The name of the account you are running in, mandatory.
           #SBATCH -A lu2025-7-106 
           # Request resources - here for eight MPI tasks
           #SBATCH -n 8
           # Request runtime for the job (HHH:MM:SS) where 168 hours is the maximum. Here asking for 15 min. 
           #SBATCH --time=00:15:00 

           # Clear the environment from any previously loaded modules
           module purge > /dev/null 2>&1

           # Load the module environment suitable for the job, it could be more or
           # less, depending on other package needs. This is for a simple job needing 
           # mpi4py.  
                                          
           ml GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 OpenMPI/4.1.6 mpi4py/3.1.5 

           # run the job - use srun for MPI jobs, but not for serial jobs 
           srun ./integration2d_mpi.py 

   .. tab:: integration2d_mpi.py 

        .. code-block:: 
 
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


Tweak for batch
############### 

Some codes need to be tweaked a little bit to run under a batch job instead of interactively, for instance. Examples could be: 

- They are querying for input during running 
- They are creating plots and open them 

In both cases the codes need to be rewritten (more or less), depending on what is needed: 

- Rewrite so they can take input from a file/dataset, or from arguments given when starting the run 
- Rewrite so the plots are saved to file instead of being opened directly 

.. challenge:: Simple example of option 1 

    Run the serial example script from Friday (https://uppmax.github.io/HPC-python/day2/basic_batch_slurm.html#simple-example-batch-script - the one that was used to run mmmult.py) but with this code (sum-2args.py) instead 
    
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
            module load GCC/12.3.0  Python/3.11.3
            
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
            module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11

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
            #SBATCH -p shared

            # Load any modules you need, here for Python 3.11.x
            module load cray-python/3.11.7

            # Run your Python script
            python sum-2args.py 2 3
            
.. solution:: Solution for C3SE 
    :class: dropdown 

          This barch script is for Alvis. Adding the numbers 2 and 3. 

          .. code-block:: bash 

             #!/bin/bash
            #SBATCH -A naiss2025-22-934 # Change to your own
            #SBATCH --time=00:05:00 # Asking for 5 minutes
            #SBATCH -p alvis
            # You need to ask for a GPU to run on alvis.
            # This is a CPU job. Do not do things like this normally!
            # Only use for GPU jobs!
            #SBATCH -N 1 --gpus-per-node=T4:1
            # Number of tasks - default is 1 core per task. 
            #SBATCH -n 1 -c 1
            # Load any modules you need, here for Python 3.11.3
            module purge  > /dev/null 2>&1
            module load Python/3.11.3-GCCcore-12.3.0 OpenMPI/4.1.5-GCC-12.3.0 SciPy-bundle/2023.07-gfbf-2023a

            # Run your Python script
            python sum-2args.py 2 3 

.. challenge:: Simple example of option 2 

    How to run a Pandas and matplotlib example as a batch job.  

    **Let us first see how you might do it interactively, from the command line** 

    You need to open a terminal window either in ThinLinc, on a DesktopOnDemand, or with regular ``ssh -Y <username|domain>`` first! 

    1. Load Python and prerequisites (and activate any needed virtual environments)
        - UPPMAX: ml Python/3.12.3-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a Python-bundle-PyPI/2024.06-GCCcore-13.3.0 matplotlib/3.9.2-gfbf-2024a 
        - HPC2N: ml GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2 Tkinter/3.11.3 
        - LUNARC: ml GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 matplotlib/3.8.2 Tkinter/3.11.5 
        - NSC: ml buildtool-easybuild/4.8.0-hpce082752a2  GCC/11.3.0  OpenMPI/4.1.4 matplotlib/3.5.2 SciPy-bundle/2022.05 Tkinter/3.10.4  
        - PDC: 
            - ml cray-python/3.11.7  
            - python -m venv --system-site-packages mymatplotlib
            - source mymatplotlib/bin/activate
            - pip install matplotlib
        - C3SE: matplotlib/3.10.5-gfbf-2025b (Loads Python/3.13.5, SciPy-buncle, Python-bundle-PyPi, Tkinter, etc.) 
            
   2. Start Python (``python``) in the ``<path-to>/Exercises/examples/programs`` directory
   3. Run these lines: 

       - At PDC  

       .. code-block:: python

          import pandas as pd
          import matplotlib.pyplot as plt
          dataframe = pd.read_csv("scottish_hills.csv")
          x = dataframe.Height
          y = dataframe.Latitude
          plt.scatter(x, y)
          plt.show()

       - At UPPMAX, HPC2N, LUNARC, NSC, and C3SE 

       .. code-block:: python 
         
          import pandas as pd 
          import matplotlib
          import matplotlib.pyplot as plt
          matplotlib.use('TkAgg')
          dataframe = pd.read_csv("scottish_hills.csv")
          x = dataframe.Height
          y = dataframe.Latitude
          plt.scatter(x, y)
          plt.show()

   **CHALLENGE: How would you do it so you could run as a batch script?** 
   
   - Hint: The main difference is that here we cannot open the plot directly, but have to save to a file instead, for instance with ``plt.savefig("myplot.png")``.
    
   - Make the change to the Python script and then make a batch script to run it! You can find solutions in the exercises directory, for each centre. 

   **NOTE** We will not talk about pandas and matplotlib otherwise. You already learned about them earlier.

   Submit with ``sbatch <batch-script.sh>``.

   The batch scripts can be found in the exercises directories for day3 for hpc2n, uppmax, lunarc, nsc, pdc, and c3se, and is named ``pandas_matplotlib-batch.sh`` .

.. keypoints::

    - Remember to include possible input arguments to the Python script in the batch script.
    - We saw an example of a batch script where we activated a virtual environment and used our own installed packages 
    - We saw a brief example of a parallel batch job 
    - We saw something about how to tweak interactive jobs to run them as batch jobs 

