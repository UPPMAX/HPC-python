#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A naiss2025-22-934
#SBATCH -t 00:20:00
#SBATCH -n 24
#SBATCH --gpus-per-task=1
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages
#SBATCH --exclusive

ml purge > /dev/null 2>&1
ml load Python/3.11.5-bare-hpc1-gcc-2023b-eb
ml load buildenv-gcccuda/12.9.1-gcc11-hpc1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/courses-fall-2025/<mydir-name>/Exercises/day4/programs/

# Load a virtual environment where numba is installed
# Use the one we created for you if during the course
source /proj/courses-fall-2025/numba-gpu/bin/activate 
# or else comment out the above line and you can create it with the following steps: 
# cd /proj/courses-fall-2025/users/<mydir>
# module load Python/3.11.5-bare-hpc1-gcc-2023b-eb
# module load buildenv-gcccuda/12.9.1-gcc11-hpc1
# python3 -m venv numba-gpu
# source numba-gpu/bin/activate
# pip3 install --upgrade pip setuptools wheel
# pip3 install numba-cuda\[cu13\] numpy
# Then in the batch script, you load it 
# source /proj/courses-fall-2025/users/<mydir>/numba-gpu/bin/activate
# Remove the comment of the above line if you created your own venv 

python $MYPATH/integration2d_gpu.py
python $MYPATH/integration2d_gpu_shared.py
