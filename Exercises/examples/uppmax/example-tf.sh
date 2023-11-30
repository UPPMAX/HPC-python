#!/bin/bash -l
#SBATCH -A naiss2023-22-1126 # Change to your own after the course
# We want to run on Snowy
#SBATCH -M snowy
#SBATCH --time=00:10:00  # Asking for 10 minutes
# Asking for one GPU card
#SBATCH --gres=gpu:1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/naiss2023-22-1126/<mydir-name>/HPC-python/Exercises/examples/programs/

module load uppmax
module load python_ML_packages/3.9.5-gpu python/3.9.5 

# Run your Python script
python $MYPATH/example-tf.py
