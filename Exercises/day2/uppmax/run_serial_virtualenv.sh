#!/bin/bash -l
#SBATCH -A uppmax2025-2-393 # Change to your own after the course
#SBATCH --time=00:01:00 # Asking for 1 minute
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/hpc-python-uppmax/<mydir-name>/Exercises/day2/programs/

# Load any modules you need, here for Python 3.11.8
module load Python/3.12.3-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a

# Activate your virtual environment. 
# CHANGE <path-to-virt-env> to the full path where you installed your virtual environment
# Example: /proj/hpc-python/mrspock/pythonUPPMAX 
source <path-to-virt-env>/bin/activate

# Run your Python script
python $MYPATH/<my_program.py>
