#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A hpc2n2025-151
#SBATCH -t 00:08:00
#SBATCH -N 1
#SBATCH -n 24
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages
#SBATCH --gpus=1
#SBATCH -C l40s 
#SBATCH --exclusive

ml purge > /dev/null 2>&1
ml GCC/12.3.0 Python/3.11.3 OpenMPI/4.1.5
ml SciPy-bundle/2023.07
ml CUDA/12.1.1 numba/0.58.1 

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/fall-courses/<mydir-name>/Exercises/day4/programs/

python $MYPATH/integration2d_gpu.py
python $MYPATH/integration2d_gpu_shared.py
