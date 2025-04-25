#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A naiss2025-22-403
# We are asking for 5 minutes
#SBATCH --time=00:05:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu

# Load the modules we need
ml load cray-python/3.11.7
ml load rocm/5.7.0

# Prepare a virtual environment with numba and TensorFlow - do this before
# running the batch script
# python -m venv --system-site-packages myTFnumba
# source myTFnumba/bin/activate
# pip install numba
# pip install tensorflow 
# pip install scikit-learn 

# Later, during the batch job, you would just activate
# the virtual environment - change to your path 
source <path-to>/myTFnumba

# Output to file - not needed if your job creates output in a file directly
# In this example I also copy the output somewhere else and then run another executable (or you could just run the same executable for different parameters).

python <my_tf_program.py> <param1> <param2> > myoutput1 2>&1
cp myoutput1 mydatadir
python <my_tf_program.py> <param3> <param4> > myoutput2 2>&1
cp myoutput2 mydatadir
python <my_tf_program.py> <param5> <param6> > myoutput3 2>&1
cp myoutput3 mydatadir
