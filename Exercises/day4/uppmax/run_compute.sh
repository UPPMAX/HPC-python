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
module load Python/3.13.5 foss/2025b CUDA/13.0.2

# Load a virtual environment we already created
source /sw/arch/local/software/python/venvs/numba-gpu/bin/activate
# If you are doing this later and need to create it yourself, do the following 
# module load Python/3.13.5 foss/2025b CUDA/13.0.2
# python3 -m venv numba-gpu
# source numba-gpu/bin/activate
# pip3 install --upgrade pip setuptools wheel
# pip3 install numba-cuda\[cu13\] numpy

# Run your Python script
python $MYPATH/compute.py

