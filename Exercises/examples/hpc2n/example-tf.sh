#!/bin/bash
#SBATCH -A hpc2nXXXX-YYY # Change to your own 
#SBATCH --time=00:10:00  # Asking for 10 minutes
# Asking for one V100 card
#SBATCH --gres=gpu:v100:1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/<your-proj-id>/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load GCC/10.3.0 OpenMPI/4.1.1 Python/3.9.5 SciPy-bundle/2021.05 TensorFlow/2.6.0-CUDA-11.3.1 

# Activate the virtual environment we installed to
# CHANGE <path-to-virt-env> to the full path where you installed your virtual environment
# Example: /proj/nobackup/hpc2nXXXX-YYY/<mydir-name>/<path-to-your-virt-env>/vpyenv
source <path-to-virt-env>/bin/activate

# Run your Python script
python $MYPATH/example-tf.py
