#!/bin/bash
# Remember to change this to your own project ID after the course!
#SBATCH -A hpc2n2023-089
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# Asking for one K80
#SBATCH --gres=gpu:k80:1

# Set a path where the example programs are installed, provided you followed the suggestion. 
# In any case, change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/hpc2n2023-089/<mydir-name>/pythonHPC2N/examples/programs/

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load GCC/10.3.0  OpenMPI/4.1.1 Python/3.9.5 SciPy-bundle/2021.05 CUDA/11.3.1 


# Activate the virtual environment we installed to
source /proj/nobackup/hpc2n2023-089/<mydir-name>/vpyenv/bin/activate

# Run your Python script
python $MYPATH/add-list.py
