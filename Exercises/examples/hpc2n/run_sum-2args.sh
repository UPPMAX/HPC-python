#!/bin/bash
#SBATCH -A hpc2n2023-089 # Change to your own after the course
#SBATCH --time=00:05:00 # Asking for 5 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/hpc2n2023-089/<mydir-name>/HPC-python/Exercises/examples/programs/

# Load any modules you need, here for Python 3.9.5
module purge
module load GCCcore/11.3.0 OpenMPI/4.1.4 Python/3.10.4 SciPy-bundle/2022.05

# Run your Python script
python $MYPATH/sum-2args.py 2 3 
