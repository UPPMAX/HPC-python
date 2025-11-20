#!/bin/bash
#SBATCH -A naiss2025-22-934 # Change to your own
#SBATCH --time=00:10:00 # Asking for 10 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/cfs/klemming/projects/snic/hpc-python-spring-naiss/<mydir-name>/HPC-python/Exercises/examples/programs/

# Load any modules you need. This is an example 
ml cray-python/3.11.7 

# Activate your virtual environment that you installed needed stuff to. 
# CHANGE <path-to-virt-env> to the full path where you installed your 
# virtual environment.  
# Example: /proj/<storage-dir>/<mydir-name>/virtenv  
source <path-to-virt-env>/bin/activate

# Run your Python script - remember to add the name of your script
python $MYPATH/seaborn-example.py
