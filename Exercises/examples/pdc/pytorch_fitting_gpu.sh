#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A naiss2025-22-934
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# The following two lines splits the output in a file for any errors and a file for other output.
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/cfs/klemming/projects/snic/hpc-python-spring-naiss/<mydir-name>/HPC-python/Exercises/examples/programs/

# Load the modules we need
ml load cray-python/3.11.7
ml load rocm/5.7.0 

# Prepare a virtual environment with pytorch - do this before
# running the batch script
# python -m venv --system-site-packages mypytorch
# source mypytorch/bin/activate
# pip install torch

# Later, during the batch job, you would just activate
# the virtual environment - change to your actual path 
source <path-to>/mypytorch 

srun python $MYPATH/pytorch_fitting_gpu.py
