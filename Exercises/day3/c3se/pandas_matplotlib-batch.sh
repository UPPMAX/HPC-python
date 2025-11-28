#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A naiss2025-22-934
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# Asking for one core
#SBATCH -n 1 

# Change to the directory where the data files and program are located
# Change the below to your own path to where you placed the example programs
cd /mimer/NOBACKUP/groups/courses-fall-2025/<mydir-name>/Exercises/day3/programs/


# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load matplotlib/3.10.5-gfbf-2025b   

# Run your Python script
python pandas_matplotlib-batch.py
