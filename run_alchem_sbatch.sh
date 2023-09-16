#!/bin/bash
#SBATCH --output=./output.txt
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --gpus-per-task=4
#SBATCH --exclusive

nvidia-smi
mpirun -n 4 hostname

cd build
make -j

#dataset_path=/shared_hdd_storage/jingjichen/gnn_datasets/weighted_shuffled_partitioned_graphs
dataset_path=/shared_hdd_storage/jingjichen/gnn_datasets/partitioned_graphs

num_gpus=1
num_layers=12
hunits=128
lr=1e-3
graph=reddit
epoch=1500
decay=0
dropout=0.5
model=resgcn
eval_freq=-1
enable_compression=0
multi_label=0

chunks=2
num_dp_ways=$num_gpus

#chunks=$((num_gpus*4))
#num_dp_ways=2

exact_inference=1
seed=1

echo "Running experiments..."

mpirun -n $num_gpus --map-by node:PE=4 ./applications/async_multi_gpus/$model --graph $dataset_path/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference --num_dp_ways $num_dp_ways --enable_compression $enable_compression --multi_label $multi_label
