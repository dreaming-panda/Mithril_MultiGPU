#!/bin/bash
#SBATCH -p gpu 
#SBATCH -A cis220117-gpu 
#SBATCH -t 00:45:00 
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
hunits=256
lr=3e-3
graph=ogbn_mag
epoch=3000
decay=0
chunks=32
dropout=0.3
seed=1234

mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/async_multi_gpus/gcn --graph $PROJECT/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file saved_weights --dropout $dropout --seed $seed

$HOME/baseline/Mithril_MultiGPU/build/applications/single_gpu/gcn_inference --graph $PROJECT/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --weight_file saved_weights

