#!/bin/bash
# Remember to change this to your own project ID after the course!
#SBATCH -A SNIC2022-22-641
#SBATCH -t 00:08:00
#SBATCH -N 1
#SBATCH -n 28
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages
#SBATCH --gres=gpu:k80:2
#SBATCH --exclusive

ml purge > /dev/null 2>&1
ml GCCcore/11.2.0 Python/3.9.6
ml GCC/11.2.0 OpenMPI/4.1.1
ml CUDA/11.4.1

# Set a path where the example programs are installed, provided you followed the suggestion. 
# In any case, change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/snic2022-22-641/<mydir-name>/pythonHPC2N/examples/programs/

# CHANGE TO YOUR OWN PATH AND THE NAME OF YOUR VIRTUAL ENVIRONMENT!
source /proj/nobackup/<your-project-storage>/<vpyenv-python-course>/bin/activate

python $MYPATH/integration2d_gpu.py
python $MYPATH/integration2d_gpu_shared.py