#!/bin/bash
#SBATCH -p gpu 
#SBATCH -A cis220117-gpu 
#SBATCH -t 10:00:00  
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1 
#SBATCH --cpus-per-task 32

RESULT_DIR=$PWD

cd ../../../build/applications/async_multi_gpus
make -j 

for chunks in 1 2 8 32 128
do
    RES_DIR=$RESULT_DIR/$chunks
    mkdir -p $RES_DIR
    echo "Running with $chunks chunks"
    mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./gcn --graph $PROJECT/gnn_datasets/reordered/reddit --layers 2 --hunits 256 --lr 1e-3 --decay 0 --epoch 3000 --chunks $chunks > $RES_DIR/reddit_gcn.txt
    mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./gcn --graph $PROJECT/gnn_datasets/reordered/ogbn_products --layers 2 --hunits 256 --lr 1e-3 --decay 0 --epoch 3000 --chunks $chunks > $RES_DIR/products_gcn.txt
    mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./gcn --graph $PROJECT/gnn_datasets/reordered/ogbn_arxiv --layers 2 --hunits 256 --lr 1e-3 --decay 0 --epoch 3000 --chunks $chunks > $RES_DIR/arxiv_gcn.txt
done


