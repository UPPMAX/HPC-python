#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A uppmax2025-2-393
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# Asking for one core
#SBATCH -n 1 

# Change to the directory where the data files and program are located
# Change the below to your own path to where you placed the example programs
cd /proj/hpc-python-uppmax/<mydir-name>/Exercises/day3/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load Python/3.12.3-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a Python-bundle-PyPI/2024.06-GCCcore-13.3.0 matplotlib/3.9.2-gfbf-2024a   

# Run your Python script
python pandas_matplotlib-batch.py
