#!/bin/bash -l
#SBATCH -A naiss2023-22-1126 # Change to your own after the course
#SBATCH --time=00:01:00 # Asking for 1 minute
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/naiss2023-22-1126/<mydir-name>/HPC-python/Exercises/examples/programs/

# Load any modules you need, here for Python 3.9.5 
module load uppmax
module load python_ML_packages/3.9.5-cpu
module load python/3.9.5

# Activate your virtual environment. 
# CHANGE <path-to-virt-env> to the full path where you installed your virtual environment
# Example: /proj/naiss2023-22-1126/mrspock/pythonUPPMAX 
source <path-to-virt-env>/bin/activate

# Run your Python script
python $MYPATH/<my_program.py>
