#!/bin/bash -l
#SBATCH -A naiss2024-22-1442 # Change to your own after the course
#SBATCH --time=00:05:00 # Asking for 5 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/hpc-python-fall/<mydir-name>/HPC-python/Exercises/examples/programs/

# Load any modules you need, here for Python 3.11.8
ml python/3.11.8

# Run your Python script
python $MYPATH/sum-2args.py 2 3
