#!/bin/bash
# Change to your own project id! 
#SBATCH -A naiss2025-22-934
#SBATCH -t 00:05:00
#SBATCH -n 4
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/hpc-python-spring-naiss/<mydir-name>/HPC-python/Exercises/examples/programs/

ml purge > /dev/null 2>&1
ml buildtool-easybuild/4.8.0-hpce082752a2  GCC/13.2.0  OpenMPI/4.1.6 mpi4py/3.1.5

mpirun -np 4 python $MYPATH/integration2d_mpi.py
