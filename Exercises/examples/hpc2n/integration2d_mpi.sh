#!/bin/bash
# Change to your own project id! 
#SBATCH -A hpc2n2024-052
#SBATCH -t 00:05:00
#SBATCH -n 4
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/python-hpc/<mydir-name>/HPC-python/Exercises/examples/programs/

ml purge > /dev/null 2>&1
ml GCC/12.3.0 Python/3.11.3
ml OpenMPI/4.1.5
ml SciPy-bundle/2023.07 mpi4py/3.1.4 
#ml Julia/1.7.1-linux-x86_64  # if Julia is needed

# CHANGE THE PATH BELOW TO YOUR OWN PATH if you need a virtual environment 
#source /proj/nobackup/<your-proj-id>/<mydir-name>/<path-to-vpyenv-python-course>/bin/activate

mpirun -np 4 python $MYPATH/integration2d_mpi.py
