#!/bin/bash
#SBATCH -A SNIC2022-22-641 # Change to your own after the course
#SBATCH --time=00:05:00 # Asking for 5 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed, provided you followed the suggestion. 
# In any case, change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/snic2022-22-641/<mydir-name>/pythonHPC2N/examples/programs/

# Load any modules you need, here for Python 3.9.5
module purge
module load GCC/10.3.0  Python/3.9.5

# Run your Python script
python $MYPATH/sum-2args.py 2 3 
