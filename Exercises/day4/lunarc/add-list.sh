#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A lu2025-7-106
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# Asking for one A100 GPU 
#SBATCH -p gpua100
#SBATCH --gres=gpu:1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/lunarc/nobackup/projects/lu2025-17-52/<mydir-name>/Exercises/day4/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load GCC/12.2.0  OpenMPI/4.1.4 Python/3.10.8 SciPy-bundle/2023.02 CUDA/12.1.1 numba/0.58.0 

# Run your Python script
python $MYPATH/add-list.py
