#!/bin/bash
#SBATCH -A naiss2024-22-1493 # Change to your own 
#SBATCH --time=00:10:00  # Asking for 10 minutes
#SBATCH -n 1
#SBATCH -c 32
# Asking for one T4 card
#SBATCH --gpus-per-task=1


# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/python-hpc-fall-nsc/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
#ml buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 julia/1.9.4-bdist OpenMPI/4.1.6 SciPy-bundle/2023.11
module load buildenv-gcccuda/12.2.2-gcc11-hpc1 Python/3.10.4-env-hpc2-gcc-2022a-eb 

source /proj/hpc-python-fall-nsc/venvNSC-TF2/bin/activate

# Run your Python script
python example-tf.py

