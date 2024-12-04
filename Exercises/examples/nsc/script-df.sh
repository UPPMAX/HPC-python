#!/bin/bash
#SBATCH -A naiss2024-22-1493 # your project_ID
#SBATCH -J job-serial        # name of the job
#SBATCH -n 1                 # nr. tasks
#SBATCH --time=00:20:00      # requested time
#SBATCH --error=job.%J.err   # error file
#SBATCH --output=job.%J.out  # output file

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/hpc-python-fall-nsc/<mydir-name>/HPC-python/Exercises/examples/programs/

# Load any modules you need, here for Python 3.11.5 and compatible SciPy-bundle
ml buildtool-easybuild/4.8.0-hpce082752a2  GCC/13.2.0 SciPy-bundle/2023.11

python $MYPATH/script-df.py
