#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A lu2025-7-34
# We are asking for 5 minutes
#SBATCH --time=00:05:00
#SBATCH --ntasks-per-node=1
# Asking for one A100 GPU
#SBATCH -p gpua100
#SBATCH --gres=gpu:1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/lunarc/nobackup/projects/lu2024-17-44/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load GCC/12.2.0  OpenMPI/4.1.4 SciPy-bundle/2023.02 numba/0.58.0  

# Run your Python script
python $MYPATH/compute.py
