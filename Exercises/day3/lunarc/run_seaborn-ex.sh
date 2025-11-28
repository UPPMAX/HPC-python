#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A lu2025-7-106 
# We are asking for 5 minutes; change if needed
#SBATCH --time=00:05:00
# Asking for one core
#SBATCH -n 1 

# Change to the directory where the data files and program are located
# Change the below to your own path to where you placed the example programs
cd /lunarc/nobackup/projects/lu2025-17-52/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load GCC/13.2.0 Seaborn/0.13.2
# this loads SciPy-bundle and Matplotlib

# Run your Python script
python seaborn-ex1.py
# change script name as needed
