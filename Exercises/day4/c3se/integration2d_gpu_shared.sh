#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A naiss2025-22-934
#SBATCH -t 00:15:00
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=T4:4
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/mimer/NOBACKUP/groups/courses-fall-2025/<mydir-name>/Exercises/day4/programs/

ml purge > /dev/null 2>&1
ml numba-cuda/0.20.0-foss-2025b-CUDA-12.9.1

python $MYPATH/integration2d_gpu.py
python $MYPATH/integration2d_gpu_shared.py
