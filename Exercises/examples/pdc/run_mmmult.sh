#!/bin/bash
#SBATCH -A naiss2025-22-403 # Change to your own
#SBATCH --time=00:10:00 # Asking for 10 minutes
#SBATCH -n 1 # Asking for 1 core

# Load any modules you need, here for cray-python/3.11.7.
module load cray-python/3.11.7

# Run your Python script
python mmmult.py
