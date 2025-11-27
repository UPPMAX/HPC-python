#!/bin/bash
# Change to your own project ID!
#SBATCH -A naiss2025-22-934
#SBATCH --time=00:30:00 # Asking for 30 minutes
# You need to ask for a GPU to run on alvis.
# This is a CPU job. Do not do things like this normally!
# Only use alvis for GPU jobs!
#SBATCH --gpus-per-node=T4:1
#SBATCH -n 1 -c 1 # Asking for 1 core    # one core per task

MYPATH=/mimer/NOBACKUP/groups/courses-fall-2025/<mydir>/Exercises/day2/programs

# Load any modules you need, here for Python 3.11.3 and compatible SciPy-bundle
module purge  > /dev/null 2>&1
module load Python/3.11.3-GCCcore-12.3.0 OpenMPI/4.1.5-GCC-12.3.0 SciPy-bundle/2023.07-gfbf-2023a

# Run your Python script
python $MYPATH/mmmult.py
