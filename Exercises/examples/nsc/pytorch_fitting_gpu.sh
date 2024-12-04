#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A naiss2024-22-1493
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# The following two lines splits the output in a file for any errors and a file for other output.
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/hpc-python-fall-nsc/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
ml load buildtool-easybuild/4.8.0-hpce082752a2 GCCcore/13.2.0
ml load Python/3.11.5

source torch_env/bin/activate

srun python $MYPATH/pytorch_fitting_gpu.py
