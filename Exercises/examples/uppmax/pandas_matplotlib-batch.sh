#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A naiss2024-22-415 
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# Asking for one core
#SBATCH -n 1 

# Change to the directory where the data files and program are located
# Change the below to your own path to where you placed the example programs
cd /proj/hpc-python/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load python/3.11.8   

# Run your Python script
python pandas_matplotlib-batch-rackham.py
