#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A lu2025-7-106 
# We are asking for 5 minutes; change if needed
#SBATCH --time=00:05:00
# Asking for one core; change as needed for some Pandas exercises
#SBATCH -n 1 

# Change to the directory where the data files and program are located
# Change the below to your own path to where you placed the example programs
cd /lunarc/nobackup/projects/lu2025-17-52/<mydir-name>/Exercises/day3/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load GCC/13.2.0 SciPy-bundle/2023.11 matplotlib/3.8.2
# Same modules needed for Pandas

# Run your Python script
python matplotlib-intro.py
# change script name as needed
