#!/bin/bash
# Change to your own project id! 
#SBATCH -A naiss2025-22-934
#SBATCH -t 00:05:00
#SBATCH -n 4
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/cfs/klemming/projects/supr/courses-fall-2025/<mydir-name>/Exercises/day4/programs/

ml cray-python/3.11.7

mpirun -np 4 python $MYPATH/integration2d_mpi.py
