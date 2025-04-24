#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A naiss2025-22-403 
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# Asking for one core
#SBATCH -n 1 

# Change to the directory where the data files and program are located
# Change the below to your own path to where you placed the example programs
cd /proj/hpc-python-spring-naiss/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load buildtool-easybuild/4.8.0-hpce082752a2  GCC/11.3.0  OpenMPI/4.1.4 Python/3.10.4 SciPy-bundle/2022.05 matplotlib/3.5.2 Tkinter/3.10.4

# Run your Python script
python pandas_matplotlib-batch-tetralith.py
