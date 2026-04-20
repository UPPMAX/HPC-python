#!/bin/bash
#SBATCH -A naiss2026-4-66 # Change to your own
#SBATCH --time=00:10:00 # Asking for 10 minutes
#SBATCH -n 1 # Asking for 1 core
#SBATCH -p shared 

MYPATH=/cfs/klemming/projects/supr/spring-courses-naiss/<mydir>/Exercises/day2/programs

# Load any modules you need, here for cray-python/3.11.7.
module load cray-python/3.11.7

# Run your Python script
python $MYPATH/mmmult.py
