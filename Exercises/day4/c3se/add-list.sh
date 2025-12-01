#!/bin/bash
# Remember to change this to your own project ID after the course!
#SBATCH -A naiss2025-22-934
# We are asking for 10 minutes
#SBATCH -t 00:10:00
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=T4:2
# Writing output and error files
#SBATCH --output=output%J.out
#SBATCH --error=error%J.error

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/mimer/NOBACKUP/groups/courses-fall-2025/<mydir-name>/Exercises/day4/programs/

# Load any needed GPU modules and any prerequisites - on Alvis this module loads all
ml purge > /dev/null 2>&1
module load numba-cuda/0.20.0-foss-2025b-CUDA-12.9.1

python add-list.py
