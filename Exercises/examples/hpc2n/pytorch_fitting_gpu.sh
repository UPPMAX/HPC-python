#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A hpc2n2024-142
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# The following two lines splits the output in a file for any errors and a file for other output.
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
# Asking for one L40s GPU card
#SBATCH --gpus=1
#SBATCH -C l40a

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/hpc-python-fall-hpc2n/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load GCC/12.3.0  OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1

srun python $MYPATH/pytorch_fitting_gpu.py
