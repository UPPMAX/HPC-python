#!/bin/bash -l
# This is a very simple example of how to run a Python script with a job array
#SBATCH -A naiss2025-22-934 # Change to your own!
#SBATCH --time=00:05:00 # Asking for 5 minutes
#SBATCH --array=1-10   # how many tasks in the array 
#SBATCH -c 1 # Asking for 1 core    # one core per task 
# Setting the name of the output file 
#SBATCH -o hello-world-%j-%a.out

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/cfs/klemming/projects/snic/hpc-python-spring-naiss/<your-dir>/HPC-python/Exercises/examples/programs/

# Load any modules you need, here for Python 3.11.x
ml load cray-python/3.11.7

# Run your Python script
srun python $MYPATH/hello-world-array.py $SLURM_ARRAY_TASK_ID
