#!/bin/bash
#SBATCH -A hpc2n2023-089 # Change to your own after the course
#SBATCH --reservation=hpc-python  # Only valid during the course 
#SBATCH --time=00:10:00  # Asking for 10 minutes
# Asking for one K80 card
#SBATCH --gres=gpu:k80:1

# Set a path where the example programs are installed, provided you followed the suggestion. 
# In any case, change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/snic2022-22-641/<mydir-name>/pythonHPC2N/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load GCC/10.3.0  OpenMPI/4.1.1 TensorFlow/2.6.0-CUDA-11.3.1

# Activate the virtual environment we installed to
# CHANGE <path-to-virt-env> to the full path where you installed your virtual environment
# Example: /proj/nobackup/hpc2n2023-089/<mydir-name>/pythonHPC2N/vpyenv
source <path-to-virt-env>/bin/activate

# Run your Python script
python $MYPATH/example-tf.py
