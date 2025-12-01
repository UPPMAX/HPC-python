#!/bin/bash
# Change to your own project id! 
#SBATCH -A hpc2n2025-151
#SBATCH -t 00:05:00
#SBATCH -n 4
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/mimer/NOBACKUP/groups/courses-fall-2025/<mydir-name>/Exercises/day4/programs/

ml purge > /dev/null 2>&1
ml Python/3.13.5-GCCcore-14.3.0
ml SciPy-bundle/2025.07-gfbf-2025b Python-bundle-PyPI/2025.07-GCCcore-14.3.0
ml mpi4py/4.1.0-gompi-2025b

mpirun -np 4 python $MYPATH/integration2d_mpi.py
