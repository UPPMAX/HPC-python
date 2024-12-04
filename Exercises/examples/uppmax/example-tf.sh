#!/bin/bash -l
#SBATCH -A naiss2024-22-1442 # Change to your own after the course
# We want to run on Snowy
#SBATCH -M snowy
#SBATCH --time=00:10:00  # Asking for 10 minutes
# Asking for one GPU card
#SBATCH --gres=gpu:1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/hpc-python-fall/<mydir-name>/HPC-python/Exercises/examples/programs/

module load uppmax
module load python_ML_packages/3.11.8-gpu python/3.11.8 

# Run your Python script
python $MYPATH/example-tf.py
