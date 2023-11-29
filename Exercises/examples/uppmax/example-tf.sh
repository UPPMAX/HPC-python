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

module load python_ML_packages/3.9.5-gpu python/3.9.5 
#module load TensorFlow/2.5.0-fosscuda-2020b 
module load scikit-learn/0.22.1

# If needed, activate the virtual environment we installed to (remove comment out)
# CHANGE <path-to-virt-env> to the full path where you installed your virtual environment
# Example: /proj/naiss2023-22-500/<mydir-name>/pythonHPC2N/vpyenv
source <path-to-virt-env>/bin/activate

# Run your Python script
python $MYPATH/example-tf.py
