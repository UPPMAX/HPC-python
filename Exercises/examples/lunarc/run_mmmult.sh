#!/bin/bash
# Batch script for running mmmult.py on Kebnekaise
#SBATCH -A lu2025-7-106 # Change to your own project ID 
#SBATCH --time=00:10:00 # Asking for 10 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/lunarc/nobackup/projects/lu2025-17-522024-17-44/<mydir-name>/HPC-python/Exercises/examples/programs/

# Purge any loaded modules
ml purge > /dev/null 2>&1

# Load any modules you need, here Python 3.11.3, SciPy-bundle,  and prerequisites.
module load GCC/12.3.0  Python/3.11.3 SciPy-bundle/2023.07

# Run your Python script
python mmmult.py
