#!/bin/bash
# Remember to change this to your own project ID after the course!
#SBATCH -A naiss2024-22-1442
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# Running on Snowy and asking for 1 GPU 
#SBATCH -M snowy
#SBATCH --gres=gpus:1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/hpc-python-fall/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load python/3.11.8 python_ML_packages-gpu

# Run your Python script
python $MYPATH/scikit-learn.py

