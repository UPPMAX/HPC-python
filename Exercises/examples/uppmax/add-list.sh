#!/bin/bash -l
# Remember to change this to your own project ID after the course!
#SBATCH -A uppmax2025-2-296
# We want to run on Snowy
#SBATCH -M snowy
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# Asking for one GPU
#SBATCH --gres=gpu:1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/hpc-python-uppmax/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
# CHANGE if you used 3.9.5 and a virtual environment instead! 
module purge  > /dev/null 2>&1
module load uppmax
module load python_ML_packages/3.11.8-gpu python/3.11.8 


# Activate the virtual environment if you used Python 3.9.5! 
# source /proj/hpc-python/<mydir-name>/vpyenv/bin/activate

# Run your Python script
python $MYPATH/add-list.py

