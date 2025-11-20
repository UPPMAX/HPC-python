#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A lu2025-7-106
#SBATCH -t 00:15:00
#SBATCH -n 24
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages
#SBATCH --ntasks-per-node=1
#SBATCH -p gpua100
#SBATCH --gres=gpu:1
#SBATCH --exclusive

ml purge > /dev/null 2>&1
module load GCC/12.2.0  OpenMPI/4.1.4 Python/3.10.8 SciPy-bundle/2023.02 CUDA/12.1.1 numba/0.58.0  

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/lunarc/nobackup/projects/lu2025-17-522024-17-44/<mydir-name>/HPC-python/Exercises/examples/programs/

python $MYPATH/integration2d_gpu.py
python $MYPATH/integration2d_gpu_shared.py
