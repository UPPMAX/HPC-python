#!/bin/bash
#SBATCH -A naiss2023-22-500
#SBATCH -t 00:05:00
#SBATCH -n 4
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/naiss2023-22-500/<mydir-name>/HPC-python/Exercises/examples/programs/

ml purge > /dev/null 2>&1
ml python/3.9.5

# CHANGE THE PATH BELOW TO YOUR OWN PATH 
source /proj/naiss2023-22-500/<mydir-name>/vpyenv-python-course/bin/activate

mpirun -np 4 python $MYPATH/integration2d_mpi.py

