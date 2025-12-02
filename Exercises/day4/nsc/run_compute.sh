#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A naiss2025-22-934
# We are asking for 5 minutes
#SBATCH --time=00:05:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/courses-fall-2025/users/<mydir-name>/Exercises/day4/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load buildenv-gcccuda/12.2.2-gcc11-hpc1 Python/3.10.4-env-hpc2-gcc-2022a-eb

# Activate the virtual environment we created earlier
source /proj/hpc-python-spring-naiss/venvNSC-numba/bin/activate

# Run your Python script
python $MYPATH/compute.py
