#!/bin/bash -l
# Change to your own project ID after the course!
#SBATCH -A uppmax2025-2-393
# We are asking for 10 minutes
#SBATCH --time=00:10:00
#SBATCH -n 1

# Go to where the example programs and data are installed.
# Change the below to your own path to where you placed the example programs
cd proj/hpc-python-uppmax/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load uppmax
module load python/3.11.8

# Activate the course environment (assuming it was called vpyenv)
source /proj/hpc-python-uppmax/<mydir-name>/<path-to-my-venv>/vpyenv/bin/activate

# Run your Python script
python simple_example.py
