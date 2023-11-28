#!/bin/bash
# Change to your own project ID! 
#SBATCH -A hpc2nXXXX-YYY
#SBATCH --time=00:10:00 # Asking for 10 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/<your-proj-id>/<mydir-name>/HPC-python/Exercises/examples/programs/

# Load any modules you need, here for Python 3.9.5 and compatible SciPy-bundle
module load GCCcore/10.3.0 OpenMPI/4.1.1 Python/3.9.5 SciPy-bundle/2021.05

# Run your Python script
python $MYPATH/mmmult.py
