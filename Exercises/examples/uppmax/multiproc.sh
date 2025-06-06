#!/bin/bash
# Batch script for running the program "multiproc.py" on Rackham
#SBATCH -A uppmax2025-2-296 # Remember to change this to your own project ID
# We are asking for 5 minutes
#SBATCH --time=00:05:00
#SBATCH -n 4

# Set a path where the example programs are installed.
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/hpc-python-uppmax/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load python/3.11.8

# Run your Python script
python $MYPATH/multiproc.py
