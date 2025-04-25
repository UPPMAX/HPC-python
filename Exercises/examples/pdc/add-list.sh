#!/bin/bash
# Remember to change this to your own project ID after the course!
#SBATCH -A naiss2025-22-403
# We are asking for 10 minutes
#SBATCH --time=00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/cfs/klemming/projects/snic/hpc-python-spring-naiss/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load cray-python/3.11.7
module load rocm/5.7.0

# Prepare a virtual environment with numba - do this before
# running the batch script
# python -m venv --system-site-packages mynumba
# source mynumba/bin/activate
# pip install numba

# Later, during the batch job, you would just activate
# the virtual environmenti - remember to change the path to 
# the actual one you used 
source <path-to>/mynumba

# Run your Python script
python $MYPATH/add-list.py
