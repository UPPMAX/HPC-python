!/bin/bash
# Remember to change this to your own project ID after the course!
#SBATCH -A uppmax2025-2-393
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# Asking for one L40s GPU
#SBATCH -p gpu
#SBATCH --gpus=l40s:1

module load numba/0.60.0-foss-2024a

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/hpc-python-uppmax/<mydir-name>/Exercises/day4/programs/

python $MYPATH/integration2d_gpu.py
python $MYPATH/integration2d_gpu_shared.py
