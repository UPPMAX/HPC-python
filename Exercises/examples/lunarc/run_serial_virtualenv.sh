#!/bin/bash
#SBATCH -A lu2025-7-34 # Change to your own
#SBATCH --time=00:10:00 # Asking for 10 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/lunarc/nobackup/projects/lu2024-17-44/<your-projecct-storage>/<mydir-name>/HPC-python/Exercises/examples/programs/

# Load any modules you need, here for Python 3.11.3 and compatible SciPy-bundle, matplotlib 
module load GCC/12.3.0  Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2

# Activate your virtual environment. 
# CHANGE <path-to-virt-env> to the full path where you installed your 
# virtual environment. For instance, the vpyenv created in the course 
# would work with this example 
# Example: /lunarc/nobackup/projects/luXXXX-YY-ZZ/<mydir-name>/<myvirtenv>
source <path-to-virt-env>/bin/activate

# Run your Python script - remember to add the name of your script
python $MYPATH/seaborn-example.py
