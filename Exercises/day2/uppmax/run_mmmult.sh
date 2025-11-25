#!/bin/bash -l
#SBATCH -A uppmax2025-2-393 # Change to your own after the course
#SBATCH --time=00:15:00 # Asking for 15 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs.
MYPATH=/proj/hpc-python-uppmax/<mydir-name>/HPC-python/Exercises/examples/programs/

# Load any modules you need, here Python 3.12.3 and a compatible SciPy-bundle for numpy
module load Python/3.12.3-GCCcore-13.3.0 
module load SciPy-bundle/2024.05-gfbf-2024a

# Run your Python script
python $MYPATH/mmmult.py
