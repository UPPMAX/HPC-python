#!/bin/bash
# Change to your own project ID! 
#SBATCH -A naiss2024-22-1493
#SBATCH --time=00:10:00 # Asking for 10 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/hpc-python-fall-nsc/<mydir-name>/HPC-python/Exercises/examples/programs/

module purge  > /dev/null 2>&1
module load buildtool-easybuild/4.8.0-hpce082752a2  GCC/13.2.0 SciPy-bundle/2023.11

# Run your Python script
python $MYPATH/mmmult.py
