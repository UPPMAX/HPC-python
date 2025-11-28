#!/bin/bash -l
#SBATCH -A uppmax2025-2-393
#SBATCH -t 00:05:00
#SBATCH -n 4
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/hpc-python-uppmax/<mydir-name>/HPC-python/Exercises/examples/programs/

ml uppmax 
ml gcc/12.3.0
ml python/3.11.8
ml openmpi/4.1.5

# If you have not already, create a virtual environment and install mpi4py after
# loading the above modules
# python -m venv --system-site-packages <myvenv>
# pip install mpi4py 

# Activate the virtual environment you install mpi4py to
# CHANGE THE PATH BELOW TO YOUR OWN PATH 
source /proj/hpc-python-uppmax/<mydir-name>/<path-to-myvenv>/bin/activate

mpirun -np 4 python $MYPATH/integration2d_mpi.py

