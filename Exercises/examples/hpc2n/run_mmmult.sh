#!/bin/bash
# Change to your own project ID! 
#SBATCH -A hpc2n2024-142
#SBATCH --time=00:10:00 # Asking for 10 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/hpc-python-fall-hpc2n/<mydir-name>/HPC-python/Exercises/examples/programs/

# Load any modules you need, here for Python 3.11.3 and compatible SciPy-bundle
module purge  > /dev/null 2>&1
module load GCC/12.3.0 OpenMPI/4.1.5 Python/3.11.3 SciPy-bundle/2023.07

# Run your Python script
python $MYPATH/mmmult.py
