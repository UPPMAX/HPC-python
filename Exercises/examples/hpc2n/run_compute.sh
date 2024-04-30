#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A hpc2n2024-052
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# Asking for one V100
#SBATCH --gres=gpu:v100:1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/<your-proj-id>/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load GCC/10.3.0  OpenMPI/4.1.1 Python/3.9.5 SciPy-bundle/2021.05 CUDA/11.4.1 

# Run your Python script
python $MYPATH/compute.py
