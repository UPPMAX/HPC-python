#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A hpc2nXXXX-YYY
#SBATCH -t 00:08:00
#SBATCH -N 1
#SBATCH -n 28
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages
#SBATCH --gres=gpu:v100:2
#SBATCH --exclusive

ml purge > /dev/null 2>&1
ml GCC/10.3.0 Python/3.9.5 OpenMPI/4.1.1
ml SciPy-bundle/2021.05 
ml CUDA/11.4.1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/<your-proj-id>/<mydir-name>/HPC-python/Exercises/examples/programs/

# CHANGE TO YOUR OWN PATH AND THE NAME OF YOUR VIRTUAL ENVIRONMENT!
source /proj/nobackup/<your-proj-id>/<mydir-name>/<path-to-vpyenv-python-course>/bin/activate

python $MYPATH/integration2d_gpu.py
python $MYPATH/integration2d_gpu_shared.py
