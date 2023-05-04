#!/bin/bash

hostname
nvidia-smi
module list

cd build
make -j

num_layers=6
hunits=200
lr=3e-3
graph=ogbn_arxiv
epoch=100
decay=1e-5
chunks=8
dropout=0.5
seed=42
model=gcn

mpirun -n 2 -N 2 ./applications/async_multi_gpus/$model --graph /home/amadeus/ssd512/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed 

