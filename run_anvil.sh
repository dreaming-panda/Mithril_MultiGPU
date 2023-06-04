#!/bin/bash
#SBATCH -p gpu 
#SBATCH -A cis220117-gpu 
#SBATCH -t 00:15:00 
#SBATCH --nodes 3
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
# products: {"hunit": 64, "lr": 0.003, "decay": 1e-05, "dropout": 0.3}
# arxiv: {"hunit": 256, "lr": 0.003, "decay": 0, "dropout": 0.3}
# reddit: {"hunit": 256, "lr": 0.003, "decay": 0, "dropout": 0.5}

num_layers=3
hunits=128
lr=3e-3
graph=ogbn_products
epoch=100
decay=0
chunks=16
dropout=0.5
seed=1
model=gcnii
eval_freq=-1
exact_inference=0
num_dp_ways=1
enable_compression=1

mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/async_multi_gpus/$model --graph $PROJECT/gnn_datasets/partitioned_graphs/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file $PROJECT/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference --num_dp_ways $num_dp_ways --enable_compression $enable_compression

#mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/async_multi_gpus/$model --graph $PROJECT/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file $PROJECT/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference

#$HOME/baseline/Mithril_MultiGPU/build/applications/single_gpu/gcn_inference --graph $PROJECT/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --weight_file $PROJECT/saved_weights_pipe
#
#mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/async_multi_gpus/gcn --graph $PROJECT/gnn_datasets/metis_32_chunks/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file saved_weights_pipe --dropout $dropout --seed $seed
#$HOME/baseline/Mithril_MultiGPU/build/applications/single_gpu/gcn_inference --graph $PROJECT/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --weight_file saved_weights_pipe
#
