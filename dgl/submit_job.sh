#!/bin/bash
#SBATCH --output=./output.txt
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=4
#SBATCH --ntasks=2
#SBATCH --exclusive

mpirun hostname > hosts
cat hosts

mpirun bash launch.sh $SLURM_JOB_NUM_NODES 
