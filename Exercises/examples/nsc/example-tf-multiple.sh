#!/bin/bash
# Remember to change this to your own project ID!
#SBATCH -A naiss2025-22-934
# We are asking for 5 minutes
#SBATCH --time=00:05:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
ml load buildtool-easybuild/4.8.0-hpce082752a2 GCCcore/13.2.0
ml load Python/3.11.5 

#source torch_env/bin/activate
source tf_env/bin/activate #unncomment this for tf env and comment torch env

# Output to file - not needed if your job creates output in a file directly
# In this example I also copy the output somewhere else and then run another executable (or you could just run the same executable for different parameters).

python <my_tf_program.py> <param1> <param2> > myoutput1 2>&1
cp myoutput1 mydatadir
python <my_tf_program.py> <param3> <param4> > myoutput2 2>&1
cp myoutput2 mydatadir
python <my_tf_program.py> <param5> <param6> > myoutput3 2>&1
cp myoutput3 mydatadir
