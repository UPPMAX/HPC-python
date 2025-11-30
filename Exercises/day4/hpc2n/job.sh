#!/bin/bash
#SBATCH -A hpc2n202X-XYZ     # your project_ID
#SBATCH -J job-serial        # name of the job
#SBATCH -n *FIXME*           # nr. tasks
#SBATCH --time=00:20:00      # requested time
#SBATCH --error=job.%J.err   # error file
#SBATCH --output=job.%J.out  # output file

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/fall-courses/<mydir-name>/Exercises/day3/programs/

# Do a purge and load any modules you need, here for Python
ml purge > /dev/null 2>&1
ml GCCcore/11.2.0 Python/3.9.6
python $MYPATH/integration2d_multiprocessing.py
