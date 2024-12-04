#!/bin/bash
# Remember to change this to your own project ID after the course!
#SBATCH -A lu2024-2-88
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# The following two lines splits the output in a file for any errors and a file for other output.
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --ntasks-per-node=1
# Asking for one A100
#SBATCH -p gpua100
#SBATCH --gres=gpu:1

# Set a path where the example programs are installed.
# Change the below to your own path to where you placed the example programs
MYPATH=/lunarc/nobackup/projects/lu2024-17-44/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/1.12.1-CUDA-11.7.0

srun python pytorch_fitting_gpu.py
