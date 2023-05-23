#!/bin/bash
#SBATCH -A hpc2n2023-089 # Change to your own after the course
#SBATCH --time=00:10:00  # Asking for 10 minutes
# Asking for one K80 card
#SBATCH --gres=gpu:k80:1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/hpc2n2023-089/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load GCC/11.2.0 OpenMPI/4.1.1 Python/3.9.6 SciPy-bundle/2021.10 TensorFlow/2.7.1

# Activate the virtual environment we installed to
# CHANGE <path-to-virt-env> to the full path where you installed your virtual environment
# Example: /proj/nobackup/hpc2n2023-089/<mydir-name>/pythonHPC2N/vpyenv
source <path-to-virt-env>/bin/activate

# Run your Python script
python $MYPATH/example-tf.py
