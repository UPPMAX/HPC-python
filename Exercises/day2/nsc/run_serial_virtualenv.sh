#!/bin/bash
#SBATCH -A naiss2025-22-934 # Change to your own
#SBATCH --time=00:10:00 # Asking for 10 minutes
#SBATCH -n 1 # Asking for 1 core

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/<your-projecct-storage>/<mydir-name>/HPC-python/Exercises/examples/programs/

# Load any modules you need. This is an example 
ml buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 JupyterLab/4.2.0

# Activate your virtual environment. 
# CHANGE <path-to-virt-env> to the full path where you installed your 
# virtual environment. For instance, the vpyenv created in the course 
# would work with this example 
# Example: /proj/<storage-dir>/<mydir-name>/virtenv  
source <path-to-virt-env>/bin/activate

# Run your Python script - remember to add the name of your script
python $MYPATH/virt-example.py
