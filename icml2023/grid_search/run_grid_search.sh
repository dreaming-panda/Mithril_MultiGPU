#!/bin/bash
#SBATCH -p gpu 
#SBATCH -A cis220117-gpu 
#SBATCH -t 48:00:00 
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32

hostname
nvidia-smi
module list

cd build
make -j
cd ..

graph=$1

date
echo "Running grid search on ${graph}"
python ./icml2023/grid_search/run.py ${graph} /anvil/projects/x-cis220117/checkpointed_weights/checkpointed_weights_${graph}
date

