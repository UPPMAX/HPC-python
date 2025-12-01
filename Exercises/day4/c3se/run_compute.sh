#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A hpc2n2025-151
# We are asking for 5 minutes
#SBATCH --time=00:05:00
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=T4:4

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/mimer/NOBACKUP/groups/courses-fall-2025/<mydir-name>/Exercises/day4/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
ml numba-cuda/0.20.0-foss-2025b-CUDA-12.9.1

# Run your Python script
python $MYPATH/compute.py
