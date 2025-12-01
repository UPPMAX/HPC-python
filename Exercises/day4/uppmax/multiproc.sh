#!/bin/bash
# Batch script for running the program "multiproc.py" on Pelle
#SBATCH -A uppmax2025-2-393 # Remember to change this to your own project ID
# We are asking for 5 minutes
#SBATCH --time=00:05:00
#SBATCH -n 4

# Set a path where the example programs are installed.
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/hpc-python-uppmax/<mydir-name>/Exercises/day4/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load Python/3.11.5-GCCcore-13.2.0 SciPy-bundle/2023.11-gfbf-2023b Python-bundle-PyPI/2023.10-GCCcore-13.2.0 mpi4py/4.0.1-gompi-2024a

# Run your Python script
python $MYPATH/multiproc.py
