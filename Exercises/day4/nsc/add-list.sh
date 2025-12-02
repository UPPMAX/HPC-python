#!/bin/bash
# Remember to change this to your own project ID after the course!
#SBATCH -A naiss2025-22-934
# We are asking for 5 minutes
#SBATCH --time=00:05:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/courses-fall-2025/users/<mydir-name>/Exercises/day4/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load Python/3.11.5-bare-hpc1-gcc-2023b-eb 
module load buildenv-gcccuda/12.9.1-gcc11-hpc1 

# If you are running this during the course, you can use the venv we have created for you
source /proj/courses-fall-2025/numba-gpu/bin/activate
# Otherwise, you have to create a virtual environment to use, then comment out the above 
# Create a virtual environment to use. Do this before submitting the job  
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

# Run your Python script
python $MYPATH/add-list.py

