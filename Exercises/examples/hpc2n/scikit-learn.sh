#!/bin/bash
# Remember to change this to your own project ID after the course!
#SBATCH -A hpc2n2024-114
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# Asking for one L40s GPU
#SBATCH --gpus=1
#SBATCH -C l40s

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/fall-courses/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load GCC/12.3.0  OpenMPI/4.1.5 TensorFlow/2.15.1-CUDA-12.1.1 SciPy-bundle/2023.07 scikit-learn/1.4.2

# Run your Python script
python $MYPATH/scikit-learn.py

