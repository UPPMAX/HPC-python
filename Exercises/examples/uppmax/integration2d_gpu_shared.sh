#!/bin/bash -l
# Remember to change this to your own project ID after the course!
#SBATCH -A naiss2023-22-1126
#SBATCH -t 00:08:00
# We want to run on Snowy
#SBATCH -M snowy
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages
#SBATCH --gres=gpu:2

module load python_ML_packages/3.9.5-gpu python/3.9.5

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/naiss2023-22-1126/<mydir-name>/HPC-python/Exercises/examples/programs/

# CHANGE TO YOUR OWN PATH AND THE NAME OF YOUR VIRTUAL ENVIRONMENT!
source /proj/nobackup/naiss2023-22-1126/<mydir-name>/<vpyenv-python-course>/bin/activate

python $MYPATH/integration2d_gpu.py
python $MYPATH/integration2d_gpu_shared.py
