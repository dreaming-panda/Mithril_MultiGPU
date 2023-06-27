#!/bin/bash

hostname
nvidia-smi

cd build
make -j

dataset_path=/shared_hdd_storage/jingjichen/gnn_datasets
num_gpus=4
num_layers=8
hunits=128
lr=1e-3
graph=reddit
epoch=100
decay=1e-5
chunks=16
dropout=0.5
seed=5
model=gcnii
eval_freq=10
exact_inference=0
num_dp_ways=1
enable_compression=0

mpirun -n $num_gpus --map-by node:PE=8 -host gnerv1:4,gnerv2:4 ./applications/async_multi_gpus/$model --graph $dataset_path/partitioned_graphs/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference --num_dp_ways $num_dp_ways --enable_compression $enable_compression

