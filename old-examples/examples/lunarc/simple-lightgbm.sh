#!/bin/bash
# Change to your own project ID after the course!
#SBATCH -A lu2025-7-106
# We are asking for 10 minutes
#SBATCH --time=00:10:00
#SBATCH -n 1

# Go to where the programs and data files are installed.
# Change the below to your own path to where you placed the example programs
cd /lunarc/nobackup/projects/lu2025-17-522024-17-44/<mydir-name>/HPC-python/Exercises/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2

# Activate the course environment (assuming it was called vpyenv)
source /lunarc/nobackup/projects/lu2025-17-522024-17-44/<mydir-name>/<path-to-my-venv>/vpyenv/bin/activate

# Run your Python script
python simple_example.py
