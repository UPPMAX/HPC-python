#!/bin/bash
#SBATCH -A hpc2n2023-089 # Change to your own after the course
#SBATCH --time=00:10:00 # Asking for 10 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed, provided you followed the suggestion. 
# In any case, change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/hpc2n2023-089/<mydir-name>/pythonHPC2N/examples/programs/

# Load any modules you need, here for Python 3.9.5 and compatible SciPy-bundle
module load GCC/10.3.0  OpenMPI/4.1.1 Python/3.9.5 SciPy-bundle/2021.05

# Run your Python script
python $MYPATH/mmmult.py
