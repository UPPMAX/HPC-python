#!/bin/bash
# Change to your own project id! 
#SBATCH -A naiss2025-22-934
#SBATCH -t 00:10:00
#SBATCH -p alvis
# You need to ask for a GPU to run on alvis.
# This is a CPU job. Do not do things like this normally!
# Only use for GPU jobs!
#SBATCH -N 1 --gpus-per-node=T4:1
# Number of tasks - default is 1 core per task. Here 4
#SBATCH -n 4
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/mimer/NOBACKUP/groups/courses-fall-2025/<mydir-name>/Exercises/day3/programs/

ml purge > /dev/null 2>&1
ml Python/3.13.5-GCCcore-14.3.0
ml SciPy-bundle/2025.07-gfbf-2025b Python-bundle-PyPI/2025.07-GCCcore-14.3.0
ml mpi4py/4.1.0-gompi-2025b

srun python $MYPATH/integration2d_mpi.py
