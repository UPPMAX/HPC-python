#!/bin/bash -l
#SBATCH -A uppmax2025-2-393
#SBATCH -t 00:05:00
#SBATCH -n 4
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/hpc-python-uppmax/<mydir-name>/Exercises/day3/programs/

module load Python/3.11.5-GCCcore-13.2.0 SciPy-bundle/2023.11-gfbf-2023b Python-bundle-PyPI/2023.10-GCCcore-13.2.0 mpi4py/4.0.1-gompi-2024a

mpirun -np 4 python $MYPATH/integration2d_mpi.py

