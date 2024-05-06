#!/bin/bash -l
# Remember to change this to your own project ID after the course!
#SBATCH -A naiss2024-22-415
# We want to run on Snowy
#SBATCH -M snowy
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# Asking for one GPU
#SBATCH --gres=gpu:1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/hpc-python/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load uppmax
module load python/3.11.8
module load python_ML_packages/3.11.8-gpu

# Run your Python script
python $MYPATH/compute.py

