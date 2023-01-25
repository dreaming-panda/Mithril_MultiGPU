#!/bin/bash
#SBATCH -p gpu 
#SBATCH -A cis220117-gpu 
#SBATCH -t 48:00:00 
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --output grid_search_products_3e-3.txt

hostname
nvidia-smi
module list

cd build
make -j
cd ..

date
echo "Running grid search on ogbn-products"
python ./icml2023/grid_search/run.py ogbn_products checkpointed_weights_ogbn_products_3e-3 3e-3
date
