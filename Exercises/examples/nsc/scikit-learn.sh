#!/bin/bash
# Remember to change this to your own project ID after the course!
#SBATCH -A naiss2025-22-403
# We are asking for 5 minutes
#SBATCH --time=00:05:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/hpc-python-spring-naiss/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
ml load buildtool-easybuild/4.8.0-hpce082752a2 GCCcore/13.2.0
ml load Python/3.11.5

source tf_env/bin/activate #unncomment this for tf env and comment torch env

# Run your Python script
python $MYPATH/scikit-learn.py

