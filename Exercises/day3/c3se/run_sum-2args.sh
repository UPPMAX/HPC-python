#!/bin/bash
#SBATCH -A naiss2025-22-934 # Change to your own
#SBATCH --time=00:05:00 # Asking for 5 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/mimer/NOBACKUP/groups/courses-fall-2025/<mydir-name>/Exercises/day3/programs/

# Load any modules you need, here for Python 3.11.3
module purge > /dev/null 2>&1
module load Python/3.11.5-GCCcore-13.2.0
module load SciPy-bundle/2025.07-gfbf-2025b
module load Python-bundle-PyPI/2025.07-GCCcore-14.3.0

# Run your Python script
python $MYPATH/sum-2args.py 2 3 
