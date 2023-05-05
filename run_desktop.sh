#!/bin/bash

hostname
nvidia-smi
module list

cd build
make -j

num_layers=4
hunits=450
lr=3e-3
graph=ogbn_arxiv
epoch=10000
decay=1e-5
chunks=8
dropout=0.6
seed=42
model=graphsage

mpirun -n 1 -N 1 ./applications/async_multi_gpus/$model --graph /home/amadeus/ssd512/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed 

