#!/bin/bash
# Remember to change this to your own project ID after the course!
#SBATCH -A uppmax2025-2-393
# We are asking for 10 minutes
#SBATCH --time=00:10:00
# Asking for one GPU
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages

# Set a path where the example programs are installed.
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/hpc-python-uppmax/<mydir-name>/Exercises/day4/programs/

# If you are doing this later and need to create the virtual environment  yourself, 
# do the following:  
# module load Python/3.13.5 foss/2025b CUDA/13.0.2
# python3 -m venv numba-gpu
# source numba-gpu/bin/activate
# pip3 install --upgrade pip setuptools wheel
# pip3 install numba-cuda\[cu13\] numpy

# Remove any loaded modules and load the ones we need, then load the 
# virtual environment created for you
ml purge > /dev/null 2>&1
module load Python/3.13.5 foss/2025b CUDA/13.0.2
source /sw/arch/local/software/python/venvs/numba-gpu/bin/activate

# Run your Python script

python integration2d_gpu.py
python integration2d_gpu_shared.py
