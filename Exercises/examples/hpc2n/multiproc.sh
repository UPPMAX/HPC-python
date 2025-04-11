#!/bin/bash
# Batch script for running the program "multiproc.py" on Kebnekaise
#SBATCH -A hpc2n2025-076 # Remember to change this to your own project ID
# We are asking for 5 minutes
#SBATCH --time=00:05:00
#SBATCH -n 4

# Set a path where the example programs are installed.
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/hpc-python-spring/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load GCC/12.3.0 Python/3.11.3

# Run your Python script
python $MYPATH/multiproc.py
