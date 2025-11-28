#!/bin/bash
#SBATCH -A lu2025-7-106 # Change to your own
#SBATCH --time=00:10:00 # Asking for 10 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/lunarc/nobackup/projects/lu2025-17-52/<mydir-name>/Exercises/day3/programs/

# Load any modules you need, here for Python 3.11.5 and compatible SciPy-bundle, matplotlib 
module load GCC/13.2.0 SciPy-bundle/2023.11 matplotlib/3.8.2

# Activate your virtual environment. 
# CHANGE <path-to-virt-env> to the full path where you installed your 
# virtual environment. 
# Example: /lunarc/nobackup/projects/lu20YY-XX-ZZ/<mydir-name>/<myvirtenv>
source <path-to-virt-env>/bin/activate

# Run your Python script - remember to add the name of your script
python $MYPATH/seaborn-example.py
