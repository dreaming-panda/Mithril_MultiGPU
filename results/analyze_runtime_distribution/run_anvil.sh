#!/bin/bash
#SBATCH -p gpu 
#SBATCH -A cis220117-gpu 
#SBATCH -t 01:00:00 
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1 
#SBATCH --cpus-per-task 32

hostname
nvidia-smi
module list

RESULT_DIR=$PWD

cd ~/Mithril_MultiGPU/build 
make -j

cd $RESULT_DIR

## run single-node without async
mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ~/Mithril_MultiGPU/build/applications/single_gpu/gcn --graph $PROJECT/gnn_datasets/reordered/reddit --layers 2 --hunits 64 --epoch 1000 --lr 1e-3 --decay 0 > ./gcn_reddit.txt
mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ~/Mithril_MultiGPU/build/applications/single_gpu/gcn --graph $PROJECT/gnn_datasets/reordered/ogbn_products --layers 2 --hunits 64 --epoch 1000 --lr 1e-3 --decay 0 > ./gcn_products.txt
mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ~/Mithril_MultiGPU/build/applications/single_gpu/gcn --graph $PROJECT/gnn_datasets/reordered/ogbn_arxiv --layers 2 --hunits 64 --epoch 1000 --lr 1e-3 --decay 0 > ./gcn_arxiv.txt

