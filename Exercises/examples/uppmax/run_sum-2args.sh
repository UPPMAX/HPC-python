#!/bin/bash
#SBATCH -A naiss2023-22-500 # Change to your own after the course
#SBATCH --time=00:05:00 # Asking for 5 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/naiss2023-22-500/<mydir-name>/HPC-python/Exercises/examples/uppmax/

# Load any modules you need, here for Python 3.9.5
module load GCC/10.3.0  OpenMPI/4.1.1 Python/3.9.5

# Run your Python script
python $MYPATH/sum-2args.py 2 3
