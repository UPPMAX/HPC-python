#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A naiss2025-22-403
#SBATCH -t 00:20:00
#SBATCH -n 24
#SBATCH --gpus-per-task=1
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages
#SBATCH --exclusive

ml purge > /dev/null 2>&1
ml load buildtool-easybuild/4.8.0-hpce082752a2 GCCcore/13.2.0
ml load Python/3.11.5

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/hpc-python-spring-naiss/<mydir-name>/HPC-python/Exercises/examples/programs/

source tf_env/bin/activate 

python $MYPATH/integration2d_gpu.py
python $MYPATH/integration2d_gpu_shared.py
