#!/bin/bash -l
# This is a very simple example of how to run a Python script with a job array
#SBATCH -A hpc2nXXXX-YYY # Change to your own!
#SBATCH --time=00:05:00 # Asking for 5 minutes
#SBATCH --array=1-10   # how many tasks in the array 
#SBATCH -c 1 # Asking for 1 core    # one core per task 
#SBATCH -o hello-world-%j-%a.out

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/<your-proj-id>/<your-dir>/HPC-python/Exercises/examples/programs/

# Load any modules you need, here for Python 3.9.5
ml GCC/10.3.0 Python/3.9.5

# Run your Python script
srun python $MYPATH/hello-world-array.py $SLURM_ARRAY_TASK_ID
