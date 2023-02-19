#!/bin/bash
#SBATCH -p gpu 
#SBATCH -A cis220117-gpu 
#SBATCH -t 03:00:00 
#SBATCH --nodes 8
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32

# TODO: change time limit and number of nodes

hostname
nvidia-smi
module list

cd build
make -j
cd ..

date
python ./icml2023/overall_performance/run.py 
date

