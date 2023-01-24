#!/bin/bash
#SBATCH -p gpu 
#SBATCH -A cis220117-gpu 
#SBATCH -t 00:15:00 
#SBATCH --nodes 1
#SBATCH --gpus-per-node 4
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --output output.txt

hostname
nvidia-smi
module list

cd build
make -j

# setting up the hyper-parameters
num_layers=4
hunits=128
lr=1e-2
graph=ogbn_products
epoch=1000
decay=1e-5
chunks=32
dropout=0.5

mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/async_multi_gpus/gcn --graph $PROJECT/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file saved_weights --dropout $dropout --seed 8888

$HOME/baseline/Mithril_MultiGPU/build/applications/single_gpu/gcn_inference --graph $PROJECT/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --weight_file saved_weights

