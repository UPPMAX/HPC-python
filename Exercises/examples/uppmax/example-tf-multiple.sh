#!/bin/bash -l
# Remember to change this to your own project ID after the course!
#SBATCH -A naiss2024-22-415
# Running on Snowy
#SBATCH -M snowy
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# Asking for one GPU
#SBATCH --gres=gpu:1

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/hpc-python/<mydir-name>/HPC-python/Exercises/examples/programs/

module load uppmax
module load python_ML_packages/3.11.8-gpu python/3.11.8

# Output to file - not needed if your job creates output in a file directly
# In this example I also copy the output somewhere else and then run another executable (or you could just run the same executable for different parameters).

python $MYPATH/<my_tf_program.py> <param1> <param2> > myoutput1 2>&1
cp myoutput1 mydatadir
python $MYPATH/<my_tf_program.py> <param3> <param4> > myoutput2 2>&1
cp myoutput2 mydatadir
python $MYPATH/<my_tf_program.py> <param5> <param6> > myoutput3 2>&1
cp myoutput3 mydatadir

