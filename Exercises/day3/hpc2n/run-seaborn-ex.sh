#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A hpc2n2025-151
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# Asking for one core
#SBATCH -n 1 

# Change to the directory where the data files and program are located
# Change the below to your own path to where you placed the example programs
cd /proj/nobackup/fall-courses/<mydir-name>/Exercises/day3/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load GCC/13.2.0 Seaborn/0.13.2
# this loads SciPy-bundle and Matplotlib

# Run your Python script
python seaborn-ex1.py
# change script name as needed
