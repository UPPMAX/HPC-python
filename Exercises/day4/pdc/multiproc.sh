#!/bin/bash
# Batch script for running the program "multiproc.py" on Kebnekaise
#SBATCH -A naiss2026-4-66 # Remember to change this to your own project ID
# We are asking for 5 minutes
#SBATCH --time=00:05:00
#SBATCH -n 4

# Set a path where the example programs are installed.
# Change the below to your own path to where you placed the example programs
MYPATH=/cfs/klemming/projects/supr/spring-courses-naiss/<mydir-name>/Exercises/day4/programs/

# Load the modules we need
module load cray-python/3.11.7  

# Run your Python script
python $MYPATH/multiproc.py
