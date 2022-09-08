#!/bin/bash
#SBATCH -A SNIC2022-22-641 # Change to your own after the course
#SBATCH --time=00:10:00 # Asking for 10 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed, provided you followed the suggestion. 
# In any case, change the below to your own path to where you placed the example programs
MYPATH=/proj/snic2022-22-641/nobackup/<mydir-name>/pythonHPC2N/examples/programs/

# Load any modules you need, here for Python 3.9.5 
module load python/3.9.5

# Activate your virtual environment. 
# CHANGE <path-to-virt-env> to the full path where you installed your virtual environment
# Example: /proj/snic2022-22-641/nobackup/mrspock/pythonUPPMAX 
source <path-to-virt-env>/bin/activate

# Run your Python script
python $MYPATH/<my_program.py>
