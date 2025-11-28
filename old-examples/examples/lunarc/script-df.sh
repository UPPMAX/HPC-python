#!/bin/bash
#SBATCH -A lu2025-7-106       # your project_ID
#SBATCH -J job-serial        # name of the job
#SBATCH -n 1                 # nr. tasks
#SBATCH --time=00:20:00      # requested time
#SBATCH --error=job.%J.err   # error file
#SBATCH --output=job.%J.out  # output file

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/lunarc/nobackup/projects/lu2025-17-522024-17-44/<mydir-name>/HPC-python/Exercises/examples/programs/

# Load any modules you need, here for Python 3.11.3 and compatible SciPy-bundle
module load GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07

python $MYPATH/script-df.py
