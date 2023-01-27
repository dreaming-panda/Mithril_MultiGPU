#!/bin/bash
#SBATCH -p gpu 
#SBATCH -A cis220117-gpu 
#SBATCH -t 00:45:00 
#SBATCH --nodes 4
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --output output.txt

hostname
nvidia-smi
module list

cd build
make -j

# setting up the hyper-parameters
num_layers=8
hunits=48
lr=1e-3
graph=ogbn_products
epoch=100
decay=0
dropout=0.5

mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/async_multi_gpus/gcn_graph_parallel --graph $PROJECT/gnn_datasets/metis_4_gpu/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --dropout $dropout --weight_file saved_weights
#mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/single_gpu/gcn --graph $PROJECT/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --dropout $dropout --weight_file saved_weights

./applications/single_gpu/gcn_inference --graph $PROJECT/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --weight_file saved_weights

