#!/bin/bash -l
# Remember to change this to your own project ID after the course!
#SBATCH -A uppmax2025-2-296
#SBATCH -t 00:08:00
# We want to run on Snowy
#SBATCH -M snowy
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages
#SBATCH --gres=gpu:1

# Change this if you are using Python 3.9.5 and python_ML_packages/3.9.5-gpu instead 
module load uppmax
module load python_ML_packages/3.11.8-gpu python/3.11.8

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/hpc-python-uppmax/<mydir-name>/HPC-python/Exercises/examples/programs/

# Uncomment this if you are using Python 3.9.5 and a virtual environment with 
# numba etc. installed! 
# CHANGE TO YOUR OWN PATH AND THE NAME OF YOUR VIRTUAL ENVIRONMENT!
# source /proj/hpc-python/<mydir-name>/<vpyenv-python-course>/bin/activate

python $MYPATH/integration2d_gpu.py
python $MYPATH/integration2d_gpu_shared.py
