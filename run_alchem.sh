#!/bin/bash

hostname
nvidia-smi

cd build
make -j

#dataset_path=/shared_hdd_storage/jingjichen/gnn_datasets/weighted_partitioned_graphs 
dataset_path=/shared_hdd_storage/jingjichen/gnn_datasets/weighted_shuffled_partitioned_graphs
#dataset_path=/shared_hdd_storage/jingjichen/gnn_datasets/partitioned_graphs

num_gpus=8
num_layers=32
hunits=100
lr=1e-3
graph=flickr
epoch=100
decay=0
dropout=0.5
model=gcn
eval_freq=-1
enable_compression=0

#chunks=$num_gpus
#num_dp_ways=$num_gpus

chunks=$((num_gpus*4))
num_dp_ways=2

exact_inference=0
seed=1

echo "Running experiments..."

mpirun -n $num_gpus --map-by node:PE=8 --host gnerv2:4,gnerv3:4 hostname

mpirun -n $num_gpus --map-by node:PE=8 --host gnerv2:4,gnerv3:4 ./applications/async_multi_gpus/$model --graph $dataset_path/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference --num_dp_ways $num_dp_ways --enable_compression $enable_compression 

