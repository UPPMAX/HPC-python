#!/bin/bash
# Change to your own project ID after the course!
#SBATCH -A naiss2025-22-934 
# We are asking for 10 minutes
#SBATCH --time=00:10:00
#SBATCH -n 1

# Go to where the programs and data files are installed.
# Change the below to your own path to where you placed the example programs
cd /proj/hpc-python-spring-naiss/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
ml buildtool-easybuild/4.8.0-hpce082752a2  GCC/11.3.0  OpenMPI/4.1.4 matplotlib/3.5.2

# Activate the course environment (assuming it was called vpyenv)
source /proj/hpc-python-spring-naiss/<mydir-name>/<path-to-my-venv>/vpyenv/bin/activate

# Run your Python script
python simple_example.py
