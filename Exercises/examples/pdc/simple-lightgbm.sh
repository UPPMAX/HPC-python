#!/bin/bash
# Change to your own project ID after the course!
#SBATCH -A naiss2025-22-403 
# We are asking for 10 minutes
#SBATCH --time=00:10:00
#SBATCH -n 1

# Go to where the programs and data files are installed.
# Change the below to your own path to where you placed the example programs
cd /cfs/klemming/projects/snic/hpc-python-spring-naiss/<mydir-name>/HPC-python/Exercises/examples/programs/

# Load the modules we need
module load cray-python/3.11.7 

# Prepare a virtual environment with lightgbm and sklearn - do this before
# running the batch script
# python -m venv --system-site-packages mylightgbm
# source mylightgbm/bin/activate
# pip install scikit-learn
# pip install lightgbm  

# Later, during the batch job, you would just activate
# the virtual environment
source <path-to>/mylightgbm

# Run your Python script
python simple_example.py
