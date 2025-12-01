#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A hpc2n2025-151
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# The following two lines splits the output in a file for any errors and a file for other output.
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=T4:4

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/mimer/NOBACKUP/groups/courses-fall-2025/<mydir-name>/Exercises/day4/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
Python/3.11.3-GCCcore-12.3.0 OpenMPI/4.1.5-GCC-12.3.0 SciPy-bundle/2023.07-gfbf-2023a PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 i

srun python $MYPATH/pytorch_fitting_gpu.py
