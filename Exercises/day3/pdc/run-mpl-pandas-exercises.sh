#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A naiss2025-22-934 
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# Asking for one core (may need to add more for later pandas exercises)
#SBATCH -n 1 

# Change to the directory where the data files and program are located
# Change the below to your own path to where you placed the example programs
cd /cfs/klemming/projects/snic/hpc-python-spring-naiss/<mydir-name>/HPC-python/Exercises/examples/programs/

# Load the modules we need
module load cray-python/3.11.7
# If not using virtual environment, uncomment next module load
# statement and comment out source statement further down:
#module load PDCOLD/23.12 matplotlib/3.8.2-cpeGNU-23.12

# Prepare a virtual environment with matplotlib - do this before
# running the batch script
# python -m venv --system-site-packages mymatplotlib
# source mymatplotlib/bin/activate
# pip install matplotlib

# Later, during the batch job, you would just activate
# the virtual environment
source <path-to>/mymatplotlib 

# Run your Python script (change to pandas as desired)
python matplotlib-intro.py
