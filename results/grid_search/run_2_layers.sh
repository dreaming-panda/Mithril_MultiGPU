#!/bin/bash
#SBATCH -p gpu 
#SBATCH -A cis220117-gpu 
#SBATCH -t 24:00:00  
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1 
#SBATCH --cpus-per-task 32

NUM_EPOCH=-1

echo "Grid search to discover the optimal parameter"
hostname
nvidia-smi
module list

RESULT_DIR=$PWD
echo "The path of the current directory is $RESULT_DIR"

# compile the project
MITHRIL_HOME=$HOME/Mithril_MultiGPU
echo "The Mithril project is located in $MITHRIL_HOME"
echo "Going to compile the project"
cd $MITHRIL_HOME
mkdir -p build
cd build 
make -j

cd $RESULT_DIR

APP=$MITHRIL_HOME/build/applications/single_gpu/gcn
for NUM_LAYERS in 2
do
    for HUNITS in 16 64 256
    do
        for LEARNING_RATE in 0.03 0.01 0.003 0.001 0.0003
        do
            for DECAY in 0 0.00001 0.0001 0.001
            do
                echo "Runing $NUM_LAYERS-layer GCN with $HUNITS hidden units, $LEARNING_RATE learning rate, and $DECAY decay"
                mkdir -p $NUM_LAYERS/$HUNITS/$LEARNING_RATE/$DECAY
                RESULT_FILE=$NUM_LAYERS/$HUNITS/$LEARNING_RATE/$DECAY/reddit_gcn.txt
                mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK $APP --graph $PROJECT/gnn_datasets/reordered/reddit --layers $NUM_LAYERS --hunits $HUNITS --epoch $NUM_EPOCH --lr $LEARNING_RATE --decay $DECAY > $RESULT_FILE
                RESULT_FILE=$NUM_LAYERS/$HUNITS/$LEARNING_RATE/$DECAY/products_gcn.txt
                mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK $APP --graph $PROJECT/gnn_datasets/reordered/ogbn_products --layers $NUM_LAYERS --hunits $HUNITS --epoch $NUM_EPOCH --lr $LEARNING_RATE --decay $DECAY > $RESULT_FILE
                RESULT_FILE=$NUM_LAYERS/$HUNITS/$LEARNING_RATE/$DECAY/arxiv_gcn.txt
                mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK $APP --graph $PROJECT/gnn_datasets/reordered/ogbn_arxiv --layers $NUM_LAYERS --hunits $HUNITS --epoch $NUM_EPOCH --lr $LEARNING_RATE --decay $DECAY > $RESULT_FILE
            done
        done
    done
done




