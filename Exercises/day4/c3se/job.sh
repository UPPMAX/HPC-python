#!/bin/bash
#SBATCH -A hpc2n202X-XYZ     # your project_ID
#SBATCH -J job-serial        # name of the job
#SBATCH -n *FIXME*           # nr. tasks
#SBATCH --time=00:20:00      # requested time
#SBATCH --error=job.%J.err   # error file
#SBATCH --output=job.%J.out  # output file

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/mimer/NOBACKUP/groups/courses-fall-2025/<mydir-name>/Exercises/day4/programs/

# Do a purge and load any modules you need, here for Python
ml purge > /dev/null 2>&1
ml Python/3.13.5-GCCcore-14.3.0
ml SciPy-bundle/2025.07-gfbf-2025b Python-bundle-PyPI/2025.07-GCCcore-14.3.0
ml mpi4py/4.1.0-gompi-2025b

python $MYPATH/integration2d_multiprocessing.py
