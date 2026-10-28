#!/bin/bash
# Remember to change this to your own project ID after the course!
#SBATCH -A naiss2025-22-934
# We are asking for 5 minutes
#SBATCH --time=00:05:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/cfs/klemming/projects/snic/hpc-python-spring-naiss/<mydir-name>/HPC-python/Exercises/examples/programs/

# Load the modules we need
ml load cray-python/3.11.7 
ml load rocm/5.7.0

# Prepare a virtual environment with scikit-learn and pytorch 
# and/or tensorflow - do this before
# running the batch script
# python -m venv --system-site-packages mysklearn
# source mysklearn/bin/activate
# pip install scikit-learn
# pip install torch
# pip install tensorflow 

# Later, during the batch job, you would just activate
# the virtual environment
source <path-to>/mysklearn

# Run your Python script
python $MYPATH/example-tf.py

