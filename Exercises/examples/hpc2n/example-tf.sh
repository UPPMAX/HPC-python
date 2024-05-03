#!/bin/bash
#SBATCH -A hpc2n2024-052 # Change to your own 
#SBATCH --time=00:10:00  # Asking for 10 minutes
# Asking for one V100 card
#SBATCH --gres=gpu:v100:1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/python-hpc/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load GCC/11.3.0  OpenMPI/4.1.4 TensorFlow/2.11.0-CUDA-11.7.0 scikit-learn/1.1.2

# Run your Python script
python $MYPATH/example-tf.py
