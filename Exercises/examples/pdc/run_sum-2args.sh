#!/bin/bash
#SBATCH -A naiss2025-22-403 # Change to your own
#SBATCH --time=00:05:00 # Asking for 5 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/cfs/klemming/projects/snic/hpc-python-spring-naiss/<mydir-name>/HPC-python/Exercises/examples/programs/

# Load any modules you need
module load cray-python/3.11.7 

# Run your Python script
python $MYPATH/sum-2args.py 2 3 
