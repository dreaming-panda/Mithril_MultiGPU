#!/bin/bash
#SBATCH -p gpu 
#SBATCH -A cis220117-gpu 
#SBATCH -t 15:00:00  
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1 
#SBATCH --cpus-per-task 32

NUM_EPOCH=5000 

echo "Tuning the number of startup epoches..."
hostname
nvidia-smi
module list


# compile the project
MITHRIL_HOME=$HOME/Mithril_MultiGPU
echo "The Mithril project is located in $MITHRIL_HOME"
echo "Going to compile the project"
cd $MITHRIL_HOME
mkdir -p build
cd build 
make -j

RESULT_DIR=$MITHRIL_HOME/results/tune_startup_epoches
echo "The path of the current directory is $RESULT_DIR"
cd $RESULT_DIR

## run single-node without any asynchrony
echo "Running the single-node GCN applications without any asynchrony..."
APP=$MITHRIL_HOME/build/applications/single_gpu/gcn
mkdir -p sync
echo "  Training GCN on reddit..."
mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK $APP --graph $PROJECT/gnn_datasets/reordered/reddit --layers 2 --hunits 256 --epoch $NUM_EPOCH --lr 1e-2 --decay 0  > ./sync/reddit_gcn.txt
echo "  Training GCN on ogbn-products..."
mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK $APP --graph $PROJECT/gnn_datasets/reordered/ogbn_products --layers 3 --hunits 256 --epoch $NUM_EPOCH --lr 5e-3 --decay 0 > ./sync/products_gcn.txt
echo "  Training GCN on ogbn-arxiv..."
mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK $APP --graph $PROJECT/gnn_datasets/reordered/ogbn_arxiv --layers 3 --hunits 256 --epoch $NUM_EPOCH --lr 1e-2 --decay 0  > ./sync/arxiv_gcn.txt

# run single-node with async
echo "Runing the single-node GCN applications with asynchrony..."
for NUM_STARTUP_EPOCHES in 0 10 20 30 40 50 100 200 300
do
    echo "*** Runing applications with $NUM_STARTUP_EPOCHES startup epoches... ***"
    APP=$MITHRIL_HOME/build/applications/async_multi_gpus/gcn
    RES_PATH=./async/$NUM_STARTUP_EPOCHES
    mkdir -p $RES_PATH
    echo "  Training GCN on reddit..."
    mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK $APP --graph $PROJECT/gnn_datasets/reordered/reddit --layers 2 --hunits 256 --epoch $NUM_EPOCH --lr 1e-2 --decay 0 --part hybrid --startup $NUM_STARTUP_EPOCHES > $RES_PATH/reddit_gcn.txt
    echo "  Training GCN on ogbn-products..."
    mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK $APP --graph $PROJECT/gnn_datasets/reordered/ogbn_products --layers 3 --hunits 256 --epoch $NUM_EPOCH --lr 5e-3 --decay 0 --part hybrid --startup $NUM_STARTUP_EPOCHES > $RES_PATH/products_gcn.txt
    echo "  Training GCN on ogbn-arxiv..."
    mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK $APP --graph $PROJECT/gnn_datasets/reordered/ogbn_arxiv --layers 3 --hunits 256 --epoch $NUM_EPOCH --lr 1e-2 --decay 0 --part hybrid --startup $NUM_STARTUP_EPOCHES > $RES_PATH/arxiv_gcn.txt
done



