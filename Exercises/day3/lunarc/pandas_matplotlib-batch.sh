#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A lu2025-7-106 
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# Asking for one core
#SBATCH -n 1 

# Change to the directory where the data files and program are located
# Change the below to your own path to where you placed the example programs
cd /lunarc/nobackup/projects/lu2025-17-52/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2  

# Run your Python script
python pandas_matplotlib-batch.py
