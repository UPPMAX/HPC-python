#!/bin/bash
#SBATCH -A hpc2n2025-151 # Change to your own
#SBATCH --time=00:10:00 # Asking for 10 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/fall-courses/<mydir-name>/Exercises/day3/programs/

# Load any modules you need
module load GCC/13.2.0 Python/3.11.5
module load SciPy-bundle/2023.11
module load Python-bundle-PyPI/2023.10

# Activate your virtual environment, which was created with the same modules loaded as bove. 
# CHANGE <path-to-virt-env> to the full path where you installed your 
# virtual environment.  
# Example: /proj/nobackup/hpc2nXXXX-YYY/<mydir-name>/pythonHPC2N 
source <path-to-virt-env>/bin/activate

# Run your Python script - remember to add the name of your script
python $MYPATH/virt-example.py
