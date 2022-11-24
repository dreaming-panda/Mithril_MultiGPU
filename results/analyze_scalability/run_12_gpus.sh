#!/bin/bash
#SBATCH -p gpu 
#SBATCH -A cis220117-gpu 
#SBATCH -t 03:00:00  
#SBATCH --gpus-per-node 4
#SBATCH --ntasks-per-node 4
#SBATCH --nodes 3
#SBATCH --cpus-per-task 32

RESULT_DIR=$PWD
NUM_EPOCH=3000

cd ../../build/applications/async_multi_gpus
make -j 

RES_DIR=$RESULT_DIR/$SLURM_NTASKS
mkdir -p $RES_DIR

#mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./gcn --graph $PROJECT/gnn_datasets/reordered/reddit --layers 2 --hunits 256 --lr 1e-3 --decay 0 --epoch $NUM_EPOCH > $RES_DIR/reddit_gcn.txt
mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./gcn --graph $PROJECT/gnn_datasets/reordered/ogbn_products --layers 2 --hunits 256 --lr 1e-3 --decay 0 --epoch $NUM_EPOCH > $RES_DIR/products_gcn.txt
mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./gcn --graph $PROJECT/gnn_datasets/reordered/ogbn_arxiv --layers 2 --hunits 256 --lr 1e-3 --decay 0 --epoch $NUM_EPOCH > $RES_DIR/arxiv_gcn.txt


