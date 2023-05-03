#!/bin/bash

hostname
nvidia-smi
module list

cd build
make -j

num_layers=4
hunits=128
lr=1e-3
graph=ogbn_arxiv
epoch=5000
decay=1e-5
chunks=16
dropout=0.5
seed=42
scaledown=0.1
model=gcnii

mpirun -n 2 -N 2 ./applications/async_multi_gpus/$model --graph /home/amadeus/ssd512/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --scaledown $scaledown 

