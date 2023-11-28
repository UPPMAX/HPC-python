#!/bin/bash
# Change to your own project id! 
#SBATCH -A hpc2nXXXX-YYY
#SBATCH -t 00:05:00
#SBATCH -n 4
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/<your-proj-id>/<mydir-name>/HPC-python/Exercises/examples/programs/

ml purge > /dev/null 2>&1
ml GCC/11.3.0 Python/3.9.5
ml OpenMPI/4.1.1
ml SciPy-bundle/2021.05
#ml Julia/1.7.1-linux-x86_64  # if Julia is needed

# CHANGE THE PATH BELOW TO YOUR OWN PATH 
source /proj/nobackup/<your-proj-id>/<mydir-name>/<path-to-vpyenv-python-course>/bin/activate

mpirun -np 4 python $MYPATH/integration2d_mpi.py

