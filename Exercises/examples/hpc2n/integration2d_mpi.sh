#!/bin/bash
#SBATCH -A hpc2n2023-089
#SBATCH -t 00:05:00
#SBATCH -n 4
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages

# Set a path where the example programs are installed, provided you followed the suggestion. 
# In any case, change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/hpc2n2023-089/<mydir-name>/pythonHPC2N/examples/programs/

ml purge > /dev/null 2>&1
ml GCCcore/11.2.0 Python/3.9.6
ml GCC/11.2.0 OpenMPI/4.1.1
#ml Julia/1.7.1-linux-x86_64  # if Julia is needed

# CHANGE THE PATH BELOW TO YOUR OWN PATH 
source /proj/nobackup/hpc2n2023-089/<mydir-name>/vpyenv-python-course/bin/activate

mpirun -np 4 python $MYPATH/integration2d_mpi.py

