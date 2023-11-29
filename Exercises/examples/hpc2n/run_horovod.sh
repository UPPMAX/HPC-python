#!/bin/bash
# Change to your own project ID!
#SBATCH -A hpc2nXXXX-YYY
#SBATCH -t 00:05:00
#SBATCH -N X               # nr. nodes - CHANGE TO ACTUAL NUMBER!
#SBATCH -n Y               # nr. MPI ranks - CHANGE TO ACTUAL NUMBER!
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages
#SBATCH --gres=gpu:v100:2
#SBATCH --exclusive

# Set a path where the example programs are installed. 
# Change the below to your own path to where you placed the example programs
MYPATH=/proj/nobackup/<your-proj-id>/<your-user-dir>/HPC-python/Exercises/examples/programs/

# Since Horovod is not installed for version 3.9.5 of Python, we are using 
# different versions of Python and other prerequisites for this example. 
ml purge > /dev/null 2>&1
ml GCC/10.2.0 CUDA/11.1.1 OpenMPI/4.0.5
ml TensorFlow/2.4.1
ml Horovod/0.21.1-TensorFlow-2.4.1

list_of_nodes=$( scontrol show hostname $SLURM_JOB_NODELIST | sed -z 's/\n/\:4,/g' )
list_of_nodes=${list_of_nodes%?}
mpirun -np $SLURM_NTASKS -H $list_of_nodes python Transfer_Learning_NLP_Horovod.py --epochs 10 --batch-size 64
