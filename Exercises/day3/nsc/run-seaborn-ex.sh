#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A naiss2025-22-934 
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# Asking for one core
#SBATCH -n 1 

# Change to the directory where the data files and program are located
# Change the below to your own path to where you placed the example programs
cd /proj/courses-fall-2025/users/<mydir-name>/Exercises/day3/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 matplotlib/3.8.2
# Seaborn installed under ~/.local/lib/python3.11 should be in $PYTHONPATH already

# Run your Python script (change as needed)
python seaborn-ex1.py
