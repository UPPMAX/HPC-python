#!/bin/bash
#SBATCH -A naiss2025-22-934 # Change to your own
#SBATCH --time=00:10:00  # Asking for 10 minutes
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu

# Set a path where the example programs are installed.
# Change the below to your own path to where you placed the example programs
MYPATH=/cfs/klemming/projects/snic/hpc-python-spring-naiss/<your-dir>/HPC-python/Exercises/examples/programs/

# Load the module we need
module load PDC/23.12
module load rocm/5.7.0
module load cray-python/3.11.5
module load craype-accel-amd-gfx90a

# Prepare a virtual environment with hip - do this before
# running the batch script
# python -m venv --system-site-packages myhip
# source myhip/bin/activate
# pip install hip-python

# Later, during the batch job, you would just activate
# the virtual environment
source <path-to>/myhip/bin/activate

# Run your Python script
python $MYPATH/hip-example.py
