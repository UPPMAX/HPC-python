#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A naiss2025-22-403
#SBATCH -t 00:20:00
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu 

ml load cray-python/3.11.7 
ml load rocm/5.7.0

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/cfs/klemming/projects/snic/hpc-python-spring-naiss/<mydir-name>/HPC-python/Exercises/examples/programs/

# Prepare a virtual environment with TensorFlow and numba - do this before
# running the batch script. Or reuse a previous virtual environment 
# python -m venv --system-site-packages myTFnumba
# source myTFnumba/bin/activate
# pip install numba
# pip install tensorflow 

# Later, during the batch job, you would just activate
# the virtual environment - change the path to your actual one 
source <path-to>/myTFnumba

python $MYPATH/integration2d_gpu.py
python $MYPATH/integration2d_gpu_shared.py
