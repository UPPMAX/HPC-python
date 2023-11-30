#!/bin/bash -l
#SBATCH -A naiss2023-22-1126
#SBATCH -t 00:05:00
#SBATCH -n 4
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/naiss2023-22-1126/<mydir-name>/HPC-python/Exercises/examples/programs/

ml uppmax 
ml gcc/9.2.0
ml python/3.9.5
ml openmpi/4.0.2

# CHANGE THE PATH BELOW TO YOUR OWN PATH 
source /proj/naiss2023-22-1126/<mydir-name>/vpyenv-python-course/bin/activate

mpirun -np 4 python $MYPATH/integration2d_mpi.py

