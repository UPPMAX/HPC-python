#!/bin/bash
# Batch script for running the program "multiproc.py" on Kebnekaise
#SBATCH -A naiss2025-22-934 # Remember to change this to your own project ID
# We are asking for 5 minutes
#SBATCH --time=00:05:00
#SBATCH -n 4

# Set a path where the example programs are installed.
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/courses-fall-2025/<mydir-name>/Exercises/day4/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load buildtool-easybuild/4.8.0-hpce082752a2  GCC/13.2.0  OpenMPI/4.1.6 Python/3.11.5

# Run your Python script
python $MYPATH/multiproc.py
