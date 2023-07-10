#!/bin/bash

hostname
nvidia-smi

cd build
make -j

dataset_path=/shared_hdd_storage/jingjichen/gnn_datasets/weighted_partitioned_graphs
num_gpus=8
num_layers=16
hunits=128
lr=1e-3
graph=reddit
epoch=50
decay=1e-5
chunks=32
dropout=0.5
seed=5
model=gcnii
eval_freq=10
exact_inference=0
num_dp_ways=1
enable_compression=0

mpirun -n $num_gpus --map-by node:PE=8 --host gnerv2:4,gnerv1:4 ./applications/async_multi_gpus/$model --graph $dataset_path/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference --num_dp_ways $num_dp_ways --enable_compression $enable_compression

