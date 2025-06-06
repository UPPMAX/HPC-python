#!/bin/bash -l
# Remember to change this to your own project ID after the course!
#SBATCH -A uppmax2025-2-296
# We are asking for 15 minutes
#SBATCH --time=00:15:00
# We want to run on Snowy
#SBATCH -M snowy
# The following two lines splits the output in a file for any errors and a file for other output.
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
# Asking for one GPU
#SBATCH --gres=gpu:1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/hpc-python-uppmax/<mydir-name>/HPC-python/Exercises/examples/programs/

module load uppmax
module load python_ML_packages/3.11.8-gpu python/3.11.8

srun python $MYPATH/pytorch_fitting_gpu.py
