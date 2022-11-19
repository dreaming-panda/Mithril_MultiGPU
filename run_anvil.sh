#!/bin/bash
#SBATCH -p gpu 
#SBATCH -A cis220117-gpu 
#SBATCH -t 00:15:00 
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1 
#SBATCH --cpus-per-task 32

hostname
nvidia-smi
module list

cd build 
make -j

mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/single_gpu/gcn --graph $PROJECT/gnn_datasets/reordered/reddit --layers 2 --hunits 256 --epoch 1000 --lr 1e-2 --decay 0  | tee ../results/test_acc_large_graphs/sync/reddit_gcn.txt
mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/single_gpu/gcn --graph $PROJECT/gnn_datasets/reordered/ogbn_products --layers 3 --hunits 256 --epoch 1000 --lr 5e-3 --decay 0  | tee ../results/test_acc_large_graphs/sync/products_gcn.txt
mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/single_gpu/gcn --graph $PROJECT/gnn_datasets/reordered/ogbn_arxiv --layers 2 --hunits 256 --epoch 1000 --lr 1e-2 --decay 0  | tee ../results/test_acc_large_graphs/sync/arxiv_gcn.txt

