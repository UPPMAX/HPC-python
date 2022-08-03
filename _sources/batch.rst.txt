Running Python in batch mode
============================

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
- Get list of your jobs: ``squeue -u <username>
- Check on a specific job: ``scontrol show job <job-id>``
- Delete a specific job: ``scancel <job-id>``
- Useful info about a job: sacct -l -j <job-id> | less -S``
- Url to a page with info about the job (Kebnekaise only): job-usage <job-id>``
         
Example Python batch scripts
---------------------------- 

.. admonition:: Running on Kebnekaise, SciPy-bundle/2021.05 and Python/3.9.5, serial code 
    :class: dropdown
   
        .. code-block:: sh

            #!/bin/bash
            #SBATCH -A SNIC2022-22-641 # Change to your own after the course
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python 3.9.5 and compatible SciPy-bundle
            module load GCC/10.3.0  OpenMPI/4.1.1 Python/3.9.5 SciPy-bundle/2021.05
            
            # Run your Python script 
            python <my_program.py>
            
            
.. admonition:: Running on Kebnekaise, SciPy-bundle/2021.05, Python/3.9.5 + Python package you have installed yourself with virtual environment. Serial code
    :class: dropdown
   
        .. code-block:: sh

            #!/bin/bash
            #SBATCH -A SNIC2022-22-641 # Change to your own after the course
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python 3.9.5 and compatible SciPy-bundle
            module load GCC/10.3.0  OpenMPI/4.1.1 Python/3.9.5 SciPy-bundle/2021.05
            
            # Activate your virtual environment. Note that you either need to have added the location to your path, or give the full path
            source <path-to-virt-env>/bin/activate
 
            # Run your Python script 
            python <my_program.py>
            

.. admonition:: Running on Kebnekaise, SciPy-bundle/2021.05, Python/3.9.5 + TensorFlow/2.6.0-CUDA-11.3.1, GPU code
    :class: dropdown
   
        .. code-block:: sh

            #!/bin/bash
            #SBATCH -A SNIC2022-22-641 # Change to your own after the course
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            # Asking for one K80 card
            #SBATCH --gres=gpu:k80:1
            
            # Load any modules you need 
            module load GCC/10.3.0  OpenMPI/4.1.1 Python/3.9.5 TensorFlow/2.6.0-CUDA-11.3.1
          
            # Run your Python script 
            python <my_tf_program.py>
            

The recommended TensorFlow version for this course is 2.6.0. The module is compatible with Python 3.9.5 (automatically loaded when you load TensorFlow and its other prerequisites).            
