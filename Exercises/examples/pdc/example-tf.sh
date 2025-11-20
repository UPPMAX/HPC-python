#!/bin/bash
#SBATCH -A naiss2025-22-934 # Change to your own 
#SBATCH --time=00:10:00  # Asking for 10 minutes
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/cfs/klemming/projects/snic/hpc-python-spring-naiss/<mydir-name>/HPC-python/Exercises/examples/programs/

# Load the module we need
module load cray-python/3.11.7
module load rocm/5.7.0

# Prepare a virtual environment with tensorflow and scikit-learn 
# as well as numba - do this before
# running the batch script
# python -m venv --system-site-packages myTF
# source myTF/bin/activate
# pip install numba 
# pip install tensorflow
# pip install scikit-learn 

# Later, during the batch job, you would just activate
# the virtual environment
source <path-to>/myTF

# Run your Python script
python $MYPATH/example-tf.py
