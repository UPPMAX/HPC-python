#!/bin/bash
# Remember to change this to your own project ID after the course!
#SBATCH -A naiss2023-22-1126
# We want to run on Snowy
#SBATCH -M snowy
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# Asking for one GPU
#SBATCH --gres=gpu:1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/naiss2023-22-1126/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load python_ML_packages python/3.9.5 


# Activate the virtual environment we installed to
source /proj/naiss2023-22-1126/<mydir-name>/vpyenv/bin/activate

# Run your Python script
python $MYPATH/add-list.py

