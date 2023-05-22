#!/bin/bash
#SBATCH -A hpc2n2023-089
#SBATCH -t 00:05:00
#SBATCH -N X               # nr. nodes
#SBATCH -n Y               # nr. MPI ranks
#SBATCH -o output_%j.out   # output file
#SBATCH -e error_%j.err    # error messages
#SBATCH --gres=gpu:k80:2
#SBATCH --exclusive

ml purge > /dev/null 2>&1
ml GCC/10.2.0 CUDA/11.1.1 OpenMPI/4.0.5
ml TensorFlow/2.4.1
ml Horovod/0.21.1-TensorFlow-2.4.1

source /proj/nobackup/hpc2n2023-089/<mydir-name>/env-horovod/bin/activate

list_of_nodes=$( scontrol show hostname $SLURM_JOB_NODELIST | sed -z 's/\n/\:4,/g' )
list_of_nodes=${list_of_nodes%?}
mpirun -np $SLURM_NTASKS -H $list_of_nodes python Transfer_Learning_NLP_Horovod.py --epochs 10 --batch-size 64
