#!/bin/bash -l
# Remember to change this to your own project ID after the course!
#SBATCH -A uppmax2025-2-393
#SBATCH -p gpu                                                                  
#SBATCH --gpus=1    
#SBATCH --gres=gpu:l40s:1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/hpc-python-uppmax/<mydir-name>/Exercises/day4/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load numba/0.60.0-foss-2024a

# Run your Python script
python $MYPATH/compute.py

