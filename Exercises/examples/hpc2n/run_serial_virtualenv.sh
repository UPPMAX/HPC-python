#!/bin/bash
#SBATCH -A hpc2n2023-089 # Change to your own after the course
#SBATCH --time=00:10:00 # Asking for 10 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/hpc2n2023-089/<mydir-name>/HPC-python/Exercises/examples/programs/

# Load any modules you need, here for Python 3.9.5 and compatible SciPy-bundle
module load GCCcore/11.3.0 OpenMPI/4.1.4 Python/3.10.4 SciPy-bundle/2022.05

# Activate your virtual environment. 
# CHANGE <path-to-virt-env> to the full path where you installed your virtual environment
# Example: /proj/nobackup/hpc2n2023-089/<mydir-name>/pythonHPC2N 
source <path-to-virt-env>/bin/activate

# Run your Python script - remember to add the name of your script
python $MYPATH/<mycode>.py
