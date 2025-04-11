#!/bin/bash
#SBATCH -A naiss2025-22-403 # Change to your own 
#SBATCH --time=00:10:00  # Asking for 10 minutes
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

#source torch_env/bin/activate
source tf_env/bin/activate #unncomment this for tf env and comment torch env

# Run your Python script
python $MYPATH/example-tf.py
