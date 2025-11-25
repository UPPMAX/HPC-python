#!/bin/bash
#SBATCH -A lu2025-7-106 # Change to your own
#SBATCH --time=00:05:00 # Asking for 5 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/lunarc/nobackup/projects/lu2025-17-522024-17-44/<mydir-name>/HPC-python/Exercises/examples/programs/

# Load any modules you need, here for Python 3.11.5
module purge > /dev/null 2>&1
module load GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11

# Run your Python script
python $MYPATH/sum-2args.py 2 3 
